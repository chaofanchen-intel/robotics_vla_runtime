#!/usr/bin/env python3
"""What machine the harness is running on, and what that changes.

Nothing about the GPU is hard-coded in the measurement code any more. A *hardware profile*
(`profiles/hardware/<name>.json`) carries what differs between machines:

* ``visible_devices``       which GPUs the services use (`--gpus 0,1`); the first one is the *primary* GPU whose
                            temperature / clock the gates read (`nvidia-smi --id` takes the physical index);
* ``allowed_devices``       optional physical GPU allowlist; an explicit `--gpus` may select only from this list;
* ``auto_select_gpus`` / ``gpus_per_task``
                            a shared machine: without `--gpus`, every cell picks the *free* GPUs itself (no compute process,
                            memory <= ``empty_gpu_max_memory_mib``), as many as ``gpus_per_task`` says that task/framework needs
                            ("task" or "task:framework", default 1), waiting for them if others hold the cards;
* ``thermal_stable`` and ``default_protocol``
                            a machine that never throttles does not need the per-request cooldown gate, so its default
                            protocol is back-to-back (override with `--protocol gated`);
* ``empty_gpu_max_memory_mib`` / ``loaded_gpu_max_memory_mib``
                            the idle-memory gate thresholds (an empty card, a card holding a model);
* ``config_overlay``        `configs/hardware/<overlay>/…` is searched before `configs/…` for every serve config (under the default's
                            own name, or the one ``config_overlay_files`` maps it to), so a
                            machine can change topology (multi-GPU, memory budgets) without touching the defaults;
* ``needs_overlay_for_tasks`` tasks whose default configs do not fit this machine: planning them without an overlay is
                            an error, not a silent OOM two hours in.
* ``config_variants``       the catalogue names a counterfactual serve config by a hardware-neutral key
                            (`vllm_omni/bagel_two_stage_prefix_cache`, written `{variant:KEY}` in an arm); this maps the key to the
                            machine's own file (path relative to `configs/`). A machine without an entry for a key cannot run the
                            experiments that use it: planning them is an error, never another machine's topology.
* ``tokens``                short machine-specific texts an arm can splice in as `{hw:NAME}` (an option, a log line to demand or forbid);
                            an undefined or empty token drops out, so an arm can carry a check only one machine needs.

A serve config under `configs/hardware/<other>/` is refused (`check_config_path`): another machine's overlay is a topology this
machine does not have, and the cell would only fail late (or, worse, run something else).

`--hardware NAME` selects a profile; without it the GPU name decides (`gpu_name_pattern`), then `generic`.
`python -m exps.hardware` prints what was detected and what to put into a new profile.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from exps.core import paths

PROFILE_DIR = paths.PROFILES / "hardware"


#: GPUs a named topology needs when the profile says nothing (BAGEL CFG parallelism: one rank per guidance branch).
TOPOLOGY_GPUS = {"cfg-parallel": 3}


@dataclass(frozen=True)
class Hardware:
    name: str
    description: str = ""
    gpu_name_pattern: str = ""
    gpu_count: int = 1
    memory_gib: float = 0.0
    thermal_stable: bool = False
    default_protocol: str = "gated"
    visible_devices: str = "0"
    allowed_devices: tuple[int, ...] = ()
    #: pick free GPUs per cell instead of using ``visible_devices`` (an explicit `--gpus` turns this off)
    auto_select_gpus: bool = False
    #: tensor-parallel size of the Qwen3-Omni Thinker in this machine's overlays (the Talker is always TP=1). It decides the
    #: fused-MoE shape the Thinker looks up (per-rank N = intermediate / TP), so the tuners must be run for it, not for TP=1.
    qwen_thinker_tp: int = 1
    #: "task" or "task:framework" -> number of GPUs the deployment needs (the fewest that run it); default 1
    gpus_per_task: dict = field(default_factory=dict)
    empty_gpu_max_memory_mib: float = 256.0
    loaded_gpu_max_memory_mib: float = 0.0
    #: seconds the idle / loaded-idle gates must hold continuously; None keeps the harness default (60 s, sized for a chassis that
    #: needs time to shed heat). A machine with stable cooling does not, and the gates are paid twice per cell.
    service_gate_continuous_s: float | None = None
    #: whether the per-request cooldown gate waits for a full SM clock and no event reason (True), or waits only for
    #: temperature and records the clock and event reasons (False). rtx5880 is False: its idle clock (210 MHz) never reaches
    #: nominal (3,105 MHz), so a clock condition blocks every cell without measuring anything. H200 keeps True.
    cooldown_gates_frequency: bool = True
    config_overlay: str | None = None
    #: default config (relative to configs/) -> overlay file (relative to configs/hardware/<overlay>/), for overlays whose name
    #: says what they are ("qwen3omni_3gpu_thinker_tp2.yaml") instead of repeating the default's name ("qwen3omni_1gpu.yaml");
    #: a config without an entry is looked up under its own name
    config_overlay_files: dict = field(default_factory=dict)
    needs_overlay_for_tasks: tuple[str, ...] = ()
    #: hardware-neutral variant key -> config file relative to configs/ (see the module docstring)
    config_variants: dict = field(default_factory=dict)
    #: `{hw:NAME}` texts (see the module docstring)
    tokens: dict = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def gpu_indices(self) -> list[int]:
        return [int(part) for part in self.visible_devices.split(",") if part.strip() != ""]

    @property
    def primary_gpu(self) -> int:
        return self.gpu_indices[0]

    def loaded_limit_mib(self) -> float:
        """Memory ceiling for a card holding a model; 0 in the profile means 97% of the card's own total."""
        if self.loaded_gpu_max_memory_mib:
            return self.loaded_gpu_max_memory_mib
        total = _query("memory.total", self.primary_gpu)
        return float(total) * 0.97 if total else 1.0e9

    def with_gpus(self, gpus: str | None) -> "Hardware":
        if not gpus:
            return self
        if not re.fullmatch(r"\d+(,\d+)*", gpus):
            raise ValueError(f"--gpus wants a comma separated list of GPU indices like 0,1; got {gpus!r}")
        indices = [int(part) for part in gpus.split(",")]
        if len(indices) != len(set(indices)):
            raise ValueError(f"--gpus cannot contain duplicate indices: {gpus!r}")
        if self.allowed_devices and not set(indices).issubset(self.allowed_devices):
            allowed = ",".join(str(index) for index in self.allowed_devices)
            raise ValueError(f"--gpus {gpus!r} includes a device outside the {self.name} allowlist ({allowed})")
        return Hardware(**{**self.__dict__, "visible_devices": gpus, "auto_select_gpus": False})

    def gpus_needed(self, task: str, framework: str | None = None) -> int:
        """GPUs the deployment of `task` (on `framework`) needs on this machine; "task:framework" wins over "task".

        A multi-GPU topology chosen with `--opt <task>.topology=NAME` (BAGEL `cfg-parallel`) looks up
        "task:framework:NAME", then "task:NAME", before the plain keys; with neither entry it needs `TOPOLOGY_GPUS[NAME]`.
        """
        table = self.gpus_per_task or {}
        from exps.core import overrides

        topology = overrides.current().option(task, "topology")
        if topology:
            for key in (f"{task}:{framework}:{topology}", f"{task}:{topology}"):
                if key in table:
                    return int(table[key])
            if topology in TOPOLOGY_GPUS:
                return TOPOLOGY_GPUS[topology]
        if framework and f"{task}:{framework}" in table:
            return int(table[f"{task}:{framework}"])
        return int(table.get(task, 1))


def _query(field_name: str, index: int = 0) -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", f"--id={index}", f"--query-gpu={field_name}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def gpu_name(index: int = 0) -> str | None:
    return _query("name", index)


def _all_gpu_state() -> dict[int, tuple[float, int]]:
    """{index: (memory used MiB, number of compute processes)} for every GPU, from two nvidia-smi calls."""
    try:
        mem = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,gpu_bus_id", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if mem.returncode != 0:
        return {}
    busy: dict[str, int] = {}
    for line in apps.stdout.splitlines() if apps.returncode == 0 else []:
        busy[line.strip().upper()] = busy.get(line.strip().upper(), 0) + 1
    state: dict[int, tuple[float, int]] = {}
    for line in mem.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit():
            state[int(parts[0])] = (float(parts[1]), busy.get(parts[2].upper(), 0))
    return state


def free_gpus(max_memory_mib: float | None = None) -> list[int]:
    """Physical GPUs nobody is using: no compute process and memory at the idle level, lowest index first."""
    limit = current().empty_gpu_max_memory_mib if max_memory_mib is None else max_memory_mib
    allowed = current().allowed_devices
    return [index for index, (used, procs) in sorted(_all_gpu_state().items())
            if used <= limit and procs == 0 and (not allowed or index in allowed)]


def select_gpus(task: str, framework: str | None = None, *, wait_s: float = 0.0, poll_s: float = 15.0) -> Hardware:
    """Point the run at the free GPUs this cell needs (`gpus_needed`), waiting up to `wait_s` for others to release them.

    No-op for a profile without `auto_select_gpus` or when `--gpus` was given. Takes exactly the number of cards the
    deployment needs, never "all that are free": a 3-GPU deployment on an 8-GPU machine uses 3.
    """
    import time

    global _CURRENT
    hw = current()
    if not hw.auto_select_gpus:
        return hw
    need = hw.gpus_needed(task, framework)
    deadline = time.monotonic() + wait_s
    while True:
        free = free_gpus()
        if len(free) >= need:
            _CURRENT = Hardware(**{**hw.__dict__, "visible_devices": ",".join(str(i) for i in free[:need])})
            return _CURRENT
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{task}/{framework or '*'} needs {need} free GPU(s) but only {len(free)} are free "
                               f"({free}); the other cards are in use by someone else. Retry later or raise --gpu-wait-s.")
        time.sleep(poll_s)


def _load(path: Path) -> Hardware:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["allowed_devices"] = tuple(raw.get("allowed_devices", ()))
    raw["needs_overlay_for_tasks"] = tuple(raw.get("needs_overlay_for_tasks", ()))
    raw["gpus_per_task"] = dict(raw.get("gpus_per_task", {}))
    raw["config_overlay_files"] = dict(raw.get("config_overlay_files", {}))
    raw["config_variants"] = dict(raw.get("config_variants", {}))
    raw["tokens"] = dict(raw.get("tokens", {}))
    raw["notes"] = tuple(raw.get("notes", ()))
    return Hardware(**raw)


def profiles() -> dict[str, Hardware]:
    return {p.stem: _load(p) for p in sorted(PROFILE_DIR.glob("*.json"))}


def detect() -> Hardware:
    name = gpu_name(0) or ""
    for profile in profiles().values():
        if profile.gpu_name_pattern and profile.gpu_name_pattern in name:
            return profile
    return profiles()["generic"]


_CURRENT: Hardware | None = None


def current() -> Hardware:
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = detect()
    return _CURRENT


def set_current(name: str | None = None, gpus: str | None = None) -> Hardware:
    """`name` None or `auto`: detect. `gpus`: override the visible devices (`--gpus 0,1`)."""
    global _CURRENT
    available = profiles()
    if name in (None, "auto"):
        base = detect()
    elif name in available:
        base = available[name]
    else:
        raise ValueError(f"unknown hardware profile {name!r}; available: {', '.join(sorted(available))}")
    _CURRENT = base.with_gpus(gpus)
    return _CURRENT


def reset() -> None:
    global _CURRENT
    _CURRENT = None


def visible_devices_env() -> str:
    # PCI_BUS_ID keeps CUDA's numbering equal to nvidia-smi's, which the free-GPU selection and the gates use
    return f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={current().visible_devices}"


def resolve_config(path: Path) -> Path:
    """The overlay copy of a config under `configs/` if this machine has one, else the path itself."""
    overlay = current().config_overlay
    if not overlay:
        return Path(path)
    try:
        relative = Path(path).resolve().relative_to(paths.CONFIGS.resolve())
    except ValueError:
        return Path(path)
    mapped = current().config_overlay_files.get(relative.as_posix())
    for name in (mapped, relative):
        if name and (paths.CONFIGS / "hardware" / overlay / name).exists():
            return paths.CONFIGS / "hardware" / overlay / name
    return Path(path)


def variant_path(key: str) -> str:
    """Repo-relative path (`exps/configs/...`) of this machine's file for the config variant `key`; an error if it has none."""
    hw = current()
    try:
        return "exps/configs/" + hw.config_variants[key].lstrip("/")
    except KeyError:
        raise SystemExit(
            f"hardware profile {hw.name!r} has no config variant {key!r}: the experiment needs a serve config that exists only "
            f"for other machines. Add it to `config_variants` in exps/profiles/hardware/{hw.name}.json (and the file under "
            f"exps/configs/hardware/{hw.config_overlay or hw.name}/), or leave the experiment out on this machine.") from None


def token(name: str) -> str:
    """The profile's `{hw:NAME}` text; empty when the profile does not define it."""
    return str(current().tokens.get(name, ""))


def check_config_path(path: Path | str) -> None:
    """Refuse a serve config inside another machine's `configs/hardware/<overlay>/` directory.

    The catalogue used to hard-code the rtx5880 overlay paths, so `--hardware h200nvl` planned (without a word) a three-GPU
    TP=2 deployment on the one card. A config of this machine's own overlay, or outside `configs/hardware/`, is fine.
    """
    try:
        relative = Path(path).resolve().relative_to((paths.CONFIGS / "hardware").resolve())
    except ValueError:
        return
    owner = relative.parts[0] if relative.parts else ""
    hw = current()
    if owner and owner != (hw.config_overlay or ""):
        raise SystemExit(
            f"{path}: a serve config of the {owner!r} overlay, but the hardware profile in use is {hw.name!r} "
            f"(overlay {hw.config_overlay or 'none'}). That topology is for another machine; use a config variant "
            f"(`config_variants` in the profile, `{{variant:KEY}}` in the catalogue) or run with --hardware {owner}.")


def check_task_fits(task: str, service_command: str | None) -> None:
    """Refuse to plan a task whose default configs do not fit this machine unless an overlay config is in use."""
    hw = current()
    if task not in hw.needs_overlay_for_tasks or not service_command:
        return
    marker = f"configs/hardware/{hw.config_overlay}/"
    if marker not in service_command:
        raise SystemExit(
            f"task {task!r} does not fit one {hw.name} GPU with its default config ({hw.memory_gib:g} GiB). Add a "
            f"multi-GPU config under exps/configs/hardware/{hw.config_overlay}/ (see exps/configs/hardware/README.md) "
            f"and pass --gpus; the single-GPU default was planned.")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Show the detected GPU(s) and the hardware profile in effect.")
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--gpus", default=None)
    args = parser.parse_args()
    hw = set_current(args.hardware, args.gpus)
    print(f"profile: {hw.name}  ({hw.description})")
    print(f"default protocol: {hw.default_protocol}; thermal_stable={hw.thermal_stable}; visible_devices={hw.visible_devices}")
    for index in range(max(hw.gpu_count, 1)):
        name = gpu_name(index)
        if name is None:
            break
        print(f"  GPU {index}: {name}, {_query('memory.total', index)} MiB total, {_query('memory.used', index)} MiB used now, "
              f"{_query('temperature.gpu', index)} C")
    print(f"idle-memory gate: empty <= {hw.empty_gpu_max_memory_mib:g} MiB, loaded <= {hw.loaded_limit_mib():,.0f} MiB "
          "(set empty_gpu_max_memory_mib from the idle 'used' value above, with a little headroom)")
    print(f"config overlay: {hw.config_overlay or '-'}; tasks needing an overlay: {list(hw.needs_overlay_for_tasks) or '-'}")
    for note in hw.notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
