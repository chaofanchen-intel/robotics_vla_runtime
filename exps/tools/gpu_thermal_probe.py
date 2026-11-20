#!/usr/bin/env python3
"""Does the card (and the host) still throttle or drift under sustained load? A ~10 min probe with a CPU-only analysis.

Background (exps/core/thermal.py): this H200 NVL is passively cooled in a chassis whose fans decide its temperature.
Historic measured requests show where it hurt: every T2S / A2T request was clean (1,785 MHz, <= 200 W) but 58% of the
BAGEL requests were not, with the SM clock down to 1,050-1,400 MHz at 384^2 and above while `sw_power_cap` (and, at
768^2, `sw_thermal_slowdown`) was active. After the fan policy was changed to "performance" the question is whether that is
gone, so this probe puts the same kind of load on the card and records what the harness's request monitor would see.

    envs/mstar/bin/python exps/tools/gpu_thermal_probe.py run --out exps/runs/THERMAL_PROBE_01 --note "fan=performance"
    envs/mstar/bin/python exps/tools/gpu_thermal_probe.py analyze exps/runs/THERMAL_PROBE_01 \\
        [--markdown out.md] [--baseline exps/runs/THERMAL_PROBE_00]
    envs/mstar/bin/python exps/tools/gpu_thermal_probe.py run --dry-run          # prints the plan, touches no GPU

Phases (seconds, all adjustable): idle (baseline) -> burn (continuous bf16 GEMM, the highest sustained power) -> mixed
(50 ms slots, 60% GEMM: the duty of a BAGEL denoise step) -> cool (idle; how long to get back under 70 C and 60 C).
GPU is sampled twice a second through exps/core/gpu.py (temperature, memory temperature, SM clock, power, enforced
power limit, clock-event reasons); the host once a second (CPU MHz of the busiest cores, package temperature, kernel thermal
throttle counters at start and end). Needs an otherwise idle GPU; it does not touch any run directory but its own.

Reading rules, fixed before the run:
* "clean" = in the burn phase no clock-event reason is active for more than 1% of samples and the median SM clock is >= 98% of nominal.
* "throttled" = a reason is active for more than 10% of burn samples, or the median clock is < 90% of nominal.
* anything between is "marginal" and the numbers are reported without a verdict.
A passing probe says the card can hold clocks under a synthetic load; the BAGEL cells themselves are the real test.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PHASES = ("idle", "burn", "mixed", "cool")
DEFAULT_SECONDS = {"idle": 60, "burn": 240, "mixed": 120, "cool": 240}
GPU_FIELDS = ["utilization_percent", "temperature_c", "memory_temperature_c", "clocks_sm_mhz", "clocks_max_sm_mhz",
              "power_draw_w", "enforced_power_limit_w", "any_event_reason_active", "reasons"]
CPU_FIELDS = ["cpu_mhz_top8_mean", "cpu_mhz_max", "cpu_pkg_temp_c_max"]


# ----- host side (plain /proc and /sys reads, no GPU) -------------------------------------------------------------------

def cpu_snapshot() -> dict:
    mhz = []
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("cpu MHz"):
                mhz.append(float(line.split(":")[1]))
    except OSError:
        pass
    top = sorted(mhz, reverse=True)[:8]
    temps = []
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (hwmon / "name").read_text().strip() != "coretemp":
                continue
            for label in hwmon.glob("temp*_label"):
                if label.read_text().startswith("Package"):
                    temps.append(int((hwmon / label.name.replace("_label", "_input")).read_text()) / 1000)
        except (OSError, ValueError):
            continue
    return {"cpu_mhz_top8_mean": statistics.fmean(top) if top else None, "cpu_mhz_max": max(mhz) if mhz else None,
            "cpu_pkg_temp_c_max": max(temps) if temps else None}


def cpu_throttle_counters() -> dict:
    """Kernel counters of thermal throttle events since boot (a reboot resets them)."""
    out = {"core": 0, "package": 0}
    for cpu in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        for kind in out:
            try:
                out[kind] += int((cpu / "thermal_throttle" / f"{kind}_throttle_count").read_text())
            except (OSError, ValueError):
                pass
    return out


def host_meta() -> dict:
    def read(path: str) -> str | None:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    cpu = "/sys/devices/system/cpu"
    return {"uptime_s": float(read("/proc/uptime").split()[0]) if read("/proc/uptime") else None,
            "governor": read(f"{cpu}/cpu0/cpufreq/scaling_governor"),
            "energy_performance_preference": read(f"{cpu}/cpu0/cpufreq/energy_performance_preference"),
            "intel_pstate": read(f"{cpu}/intel_pstate/status"), "no_turbo": read(f"{cpu}/intel_pstate/no_turbo"),
            "cpu_throttle_counters": cpu_throttle_counters()}


# ----- load generators --------------------------------------------------------------------------------------------------

def make_load(size: int):
    import torch
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)

    def gemm(n: int = 1) -> None:
        for _ in range(n):
            torch.mm(a, b)

    def sync() -> None:
        torch.cuda.synchronize()

    return gemm, sync


def run_phase(name: str, seconds: float, gemm, sync) -> None:
    end = time.monotonic() + seconds
    if name in ("idle", "cool"):
        time.sleep(seconds)
    elif name == "burn":
        while time.monotonic() < end:
            gemm(20)
            sync()
    elif name == "mixed":
        while time.monotonic() < end:           # 30 ms of GEMM then 20 ms idle, like a launch-bound denoise step
            slot = time.monotonic()
            while time.monotonic() - slot < 0.030:
                gemm(1)
                sync()
            time.sleep(0.020)


class Sampler(threading.Thread):
    def __init__(self, writer: csv.DictWriter, period: float, gpu_module) -> None:
        super().__init__(daemon=True)
        self.writer, self.period, self.gpu = writer, period, gpu_module
        self.stop = threading.Event()
        self.phase = "idle"
        self.t0 = time.monotonic()
        self.rows = 0
        self.last_cpu: dict = {}
        self.last_cpu_at = 0.0

    def run(self) -> None:
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                s = self.gpu.sample()
            except Exception as exc:  # a failed query is a gap in the record, not the end of the probe
                print(f"gpu sample failed: {exc}", file=sys.stderr)
                s = None
            if started - self.last_cpu_at >= 1.0:
                self.last_cpu, self.last_cpu_at = cpu_snapshot(), started
            if s:
                row = {"t_s": round(started - self.t0, 2), "phase": self.phase,
                       **{k: s.get(k) for k in GPU_FIELDS if k != "reasons"},
                       "reasons": "|".join(sorted(k for k, v in (s.get("event_reasons") or {}).items() if v)),
                       **self.last_cpu}
                self.writer.writerow(row)
                self.rows += 1
            self.stop.wait(max(0.0, self.period - (time.monotonic() - started)))


def run(args: argparse.Namespace) -> int:
    seconds = {p: getattr(args, p) for p in PHASES}
    plan = " -> ".join(f"{p} {seconds[p]:.0f}s" for p in PHASES)
    if args.dry_run:
        print(f"would run {plan}\n  GEMM {args.size}x{args.size} bf16, GPU sample every {args.period}s, output {args.out}")
        print(f"  host now: {json.dumps(host_meta())}")
        return 0
    from exps.core import gpu
    out = args.out
    if (out / "samples.csv").exists():
        print(f"{out}/samples.csv exists; refusing to overwrite (use a new directory)", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": args.note, "plan": seconds, "size": args.size,
            "host_before": host_meta(), "gpu_probe": gpu.probe(), "apps_on_gpu": gpu.compute_apps()}
    if meta["apps_on_gpu"]:
        print(f"GPU is not idle (pids {meta['apps_on_gpu']}); stop them first", file=sys.stderr)
        return 2
    gemm, sync = make_load(args.size)
    gemm(5)
    sync()
    with open(out / "samples.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["t_s", "phase", *[k for k in GPU_FIELDS], *CPU_FIELDS])
        writer.writeheader()
        sampler = Sampler(writer, args.period, gpu)
        sampler.start()
        try:
            for phase in PHASES:
                sampler.phase = phase
                print(f"phase {phase} ({seconds[phase]:.0f}s)", flush=True)
                run_phase(phase, seconds[phase], gemm, sync)
        finally:
            sampler.stop.set()
            sampler.join(timeout=10)
            handle.flush()
    meta["host_after"] = host_meta()
    meta["samples"] = sampler.rows
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{sampler.rows} samples -> {out}/samples.csv")
    return 0


# ----- analysis (CPU only) ----------------------------------------------------------------------------------------------

def load(directory: Path) -> tuple[list[dict], dict]:
    with open(directory / "samples.csv") as handle:
        rows = list(csv.DictReader(handle))
    meta = json.loads((directory / "meta.json").read_text()) if (directory / "meta.json").exists() else {}
    return rows, meta


def num(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def series(rows: list[dict], key: str) -> list[float]:
    return [v for v in (num(r, key) for r in rows) if v is not None]


def first_time_at_or_below(rows: list[dict], key: str, limit: float) -> float | None:
    """Seconds since the phase started until `key` first is <= limit (None if it never is)."""
    if not rows:
        return None
    start = num(rows[0], "t_s")
    for r in rows:
        v = num(r, key)
        if v is not None and v <= limit:
            return num(r, "t_s") - start
    return None


def phase_stats(rows: list[dict]) -> dict:
    clocks, nominal = series(rows, "clocks_sm_mhz"), (series(rows, "clocks_max_sm_mhz") or [None])[0]
    reasons: dict[str, int] = {}
    for r in rows:
        for reason in filter(None, (r.get("reasons") or "").split("|")):
            reasons[reason] = reasons.get(reason, 0) + 1
    n = len(rows)
    ordered = sorted(clocks)
    return {
        "n": n, "seconds": (num(rows[-1], "t_s") - num(rows[0], "t_s")) if n > 1 else 0.0,
        "temp_start": (series(rows, "temperature_c") or [None])[0], "temp_end": (series(rows, "temperature_c") or [None])[-1],
        "temp_max": max(series(rows, "temperature_c"), default=None),
        "mem_temp_max": max(series(rows, "memory_temperature_c"), default=None),
        "clock_median": statistics.median(clocks) if clocks else None,
        "clock_p5": ordered[int(0.05 * len(ordered))] if ordered else None,
        "clock_min": min(clocks) if clocks else None, "nominal": nominal,
        "power_mean": statistics.fmean(series(rows, "power_draw_w")) if series(rows, "power_draw_w") else None,
        "power_max": max(series(rows, "power_draw_w"), default=None),
        "limit": (series(rows, "enforced_power_limit_w") or [None])[0],
        "limit_min": min(series(rows, "enforced_power_limit_w"), default=None),
        "reason_share": {k: v / n for k, v in reasons.items()} if n else {},
        "cpu_mhz": statistics.fmean(series(rows, "cpu_mhz_top8_mean")) if series(rows, "cpu_mhz_top8_mean") else None,
        "cpu_temp_max": max(series(rows, "cpu_pkg_temp_c_max"), default=None),
    }


def verdict(burn: dict) -> str:
    if not burn["n"] or burn["clock_median"] is None or not burn["nominal"]:
        return "no verdict (no burn samples with a clock reading)"
    share = max(burn["reason_share"].values(), default=0.0)
    frac = burn["clock_median"] / burn["nominal"]
    if share > 0.10 or frac < 0.90:
        return f"THROTTLED (reason active in {share:.0%} of burn samples, median clock {frac:.0%} of nominal)"
    if share <= 0.01 and frac >= 0.98:
        return f"CLEAN (reason active in {share:.1%} of burn samples, median clock {frac:.0%} of nominal)"
    return f"MARGINAL (reason active in {share:.0%} of burn samples, median clock {frac:.0%} of nominal)"


def fmt(v, spec="{:.0f}") -> str:
    return "n/a" if v is None else spec.format(v)


def report(rows: list[dict], meta: dict, label: str) -> tuple[str, dict]:
    by_phase = {p: [r for r in rows if r["phase"] == p] for p in PHASES}
    stats = {p: phase_stats(v) for p, v in by_phase.items() if v}
    lines = [f"## {label}", "", f"note: {meta.get('note') or '-'}; started {meta.get('started', '?')}; {len(rows)} samples", "",
             "| phase | s | temp start -> end / max (C) | memory max (C) | SM clock median / p5 / min (MHz) | power mean / max (W) | "
             "enforced limit (W) | clock-event reasons (share of samples) | CPU MHz top-8 / package max C |",
             "|---|---:|---|---:|---|---|---:|---|---|"]
    for p, s in stats.items():
        reasons = ", ".join(f"{k} {v:.0%}" for k, v in sorted(s["reason_share"].items(), key=lambda kv: -kv[1])) or "none"
        lines.append(f"| {p} | {s['seconds']:.0f} | {fmt(s['temp_start'])} -> {fmt(s['temp_end'])} / {fmt(s['temp_max'])} | "
                     f"{fmt(s['mem_temp_max'])} | {fmt(s['clock_median'])} / {fmt(s['clock_p5'])} / {fmt(s['clock_min'])} "
                     f"(nominal {fmt(s['nominal'])}) | {fmt(s['power_mean'])} / {fmt(s['power_max'])} | "
                     f"{fmt(s['limit'])} (min {fmt(s['limit_min'])}) | {reasons} | {fmt(s['cpu_mhz'])} / {fmt(s['cpu_temp_max'])} |")
    lines.append("")
    burn = stats.get("burn")
    if burn:
        throttled = first_time_at_or_below(by_phase["burn"], "clocks_sm_mhz", (burn["nominal"] or 0) * 0.98)
        lines.append(f"- burn verdict: **{verdict(burn)}**")
        lines.append(f"- first time the SM clock fell below 98% of nominal during burn: "
                     f"{'never' if throttled is None else f'{throttled:.0f} s after the phase started'}")
        for kind in ("core", "package"):
            before = (meta.get("host_before") or {}).get("cpu_throttle_counters", {}).get(kind)
            after = (meta.get("host_after") or {}).get("cpu_throttle_counters", {}).get(kind)
            if before is not None and after is not None:
                lines.append(f"- CPU {kind} thermal-throttle counter: {before} -> {after} ({after - before:+d} during the probe)")
    cool = by_phase["cool"]
    if cool and stats.get("burn") and stats["cool"]["temp_start"] is not None:
        for limit in (70, 60):
            t = first_time_at_or_below(cool, "temperature_c", limit)
            lines.append(f"- cooling after load: back to <= {limit} C "
                         f"{'not within the phase' if t is None else f'after {t:.0f} s'} (gate target is 70 C; thermal.py estimated ~10 s/C)")
    host = meta.get("host_before") or {}
    lines += ["", f"host: governor {host.get('governor')}, EPP {host.get('energy_performance_preference')}, intel_pstate "
              f"{host.get('intel_pstate')}, no_turbo {host.get('no_turbo')}, uptime {fmt(host.get('uptime_s'))} s", ""]
    return "\n".join(lines), stats


def analyze(args: argparse.Namespace) -> int:
    rows, meta = load(args.directory)
    text, stats = report(rows, meta, f"{args.directory.name}")
    out = ["# GPU / host thermal probe", "", text]
    if args.baseline:
        brows, bmeta = load(args.baseline)
        btext, bstats = report(brows, bmeta, f"baseline {args.baseline.name}")
        out += [btext, "## Change", ""]
        for key, name in (("clock_median", "burn median SM clock (MHz)"), ("temp_max", "burn max temp (C)"),
                          ("power_mean", "burn mean power (W)")):
            a, b = stats.get("burn", {}).get(key), bstats.get("burn", {}).get(key)
            out.append(f"- {name}: {fmt(b)} -> {fmt(a)}")
    out.append("Limits: a synthetic GEMM load is not BAGEL (its power and memory traffic differ); sampling is 2 Hz; the "
               "probe says what the card does under load now, and says nothing about the runs that already happened.")
    result = "\n".join(out) + "\n"
    print(result)
    if args.markdown:
        args.markdown.write_text(result)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="apply load and record (GPU)")
    r.add_argument("--out", type=Path, default=ROOT / "exps" / "runs" / "THERMAL_PROBE_01")
    r.add_argument("--note", default="", help="free text, e.g. 'fan=performance, 2026-10-27'")
    r.add_argument("--size", type=int, default=8192, help="GEMM edge (bf16)")
    r.add_argument("--period", type=float, default=0.5, help="GPU sampling period in seconds")
    for phase in PHASES:
        r.add_argument(f"--{phase}", type=float, default=DEFAULT_SECONDS[phase], help=f"{phase} phase seconds")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=run)
    a = sub.add_parser("analyze", help="summarise a probe directory (CPU only)")
    a.add_argument("directory", type=Path)
    a.add_argument("--baseline", type=Path, help="an earlier probe to compare with")
    a.add_argument("--markdown", type=Path)
    a.set_defaults(func=analyze)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
