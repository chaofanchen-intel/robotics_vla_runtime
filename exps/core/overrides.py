#!/usr/bin/env python3
"""Every knob a counterfactual or an instrumented run changes, in one explicit place.

The earlier harness steered ablations through process-wide environment variables
(`EXPERIMENTS_MSTAR_EXTRA_ENV`, `..._VLLM_EXTRA_ENV`, `..._NSYS_DIR`, a V-JEPA dtype switch, a Pi0.5
graph switch). A forgotten `export` silently changed every later cell, and nothing in the manifest said
which cell had been run under which override. Here the overrides are an object that the command line
builds, that the serve-command builders read, and that every manifest records. Nothing reads
`os.environ` for experiment settings any more.

Four kinds of override
----------------------
``env``        ``--env FRAMEWORK:KEY=VALUE``    environment of the serve process (``*`` = every framework)
``serve_args`` ``--serve-arg FRAMEWORK:ARG``    extra argument appended to the serve command
``options``    ``--opt TASK.KEY=VALUE``         task-level switch read by that task's cell (``pi05.vllm_graph=1``);
                                                ``--opt config.<family>=PATH`` swaps the serve config of a framework
``nsys``       ``--nsys-dir DIR``               wrap the serve command in ``nsys profile`` (profiling cells only)

FRAMEWORK is the runtime *family*: ``mstar``, ``vllm-omni``, ``sglang`` (SGLang-Omni and SGLang core),
``voxserve`` or ``*``. A BAGEL ``vllm-omni-two-stage`` cell reads the ``vllm-omni`` entries.
"""
from __future__ import annotations

import argparse
import shlex

from exps.core import paths
from dataclasses import dataclass, field
from typing import Any

FAMILIES = ("mstar", "vllm-omni", "sglang", "voxserve")


@dataclass
class Nsys:
    out_dir: str
    #: `node` shows per-kernel records inside CUDA graphs; `graph` is the fallback when `node` truncates a trace.
    graph_trace: str = "node"
    #: Multi-process engines lose their last unflushed CUDA buffers when the parent kills them; a short flush
    #: interval bounds that loss at the price of profiler overhead (busy / launch counts are unaffected).
    flush_ms: int | None = None


@dataclass
class Overrides:
    env: dict[str, dict[str, str]] = field(default_factory=dict)
    serve_args: dict[str, list[str]] = field(default_factory=dict)
    options: dict[str, str] = field(default_factory=dict)
    nsys: Nsys | None = None

    # --- reads -----------------------------------------------------------------------------------------
    def env_for(self, family: str) -> dict[str, str]:
        merged = dict(self.env.get("*", {}))
        merged.update(self.env.get(family, {}))
        return merged

    def args_for(self, family: str) -> list[str]:
        return list(self.serve_args.get("*", [])) + list(self.serve_args.get(family, []))

    def option(self, task: str, key: str, default: str | None = None) -> str | None:
        return self.options.get(f"{task}.{key}", default)

    def config_for(self, family: str, default):
        """`--opt config.<family>=PATH` replaces the serve config of every cell of that framework family (multi-GPU
        topologies, memory budgets). Repo-relative or absolute; must exist."""
        from pathlib import Path

        raw = self.options.get(f"config.{family}")
        if not raw:
            return default
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = paths.REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"--opt config.{family} points to a missing file: {path}")
        from exps import hardware  # local: hardware imports this module

        hardware.check_config_path(path)
        return path

    def is_default(self) -> bool:
        return not (self.env or self.serve_args or self.options or self.nsys)

    def env_words(self, family: str) -> str:
        """`K=V K=V` ready to splice into a shell command, each pair quoted."""
        return " ".join(shlex.quote(f"{k}={v}") for k, v in sorted(self.env_for(family).items()))

    # --- record ----------------------------------------------------------------------------------------
    def as_manifest(self) -> dict[str, Any]:
        return {
            "env": {k: dict(sorted(v.items())) for k, v in sorted(self.env.items())},
            "serve_args": {k: list(v) for k, v in sorted(self.serve_args.items())},
            "options": dict(sorted(self.options.items())),
            "nsys": None if self.nsys is None else vars(self.nsys).copy(),
        }


_CURRENT = Overrides()


def current() -> Overrides:
    return _CURRENT


def set_current(overrides: Overrides) -> None:
    global _CURRENT
    _CURRENT = overrides


def reset() -> None:
    set_current(Overrides())


def _split_family(text: str, what: str) -> tuple[str, str]:
    family, sep, rest = text.partition(":")
    if not sep or not rest:
        raise ValueError(f"{what} must look like FAMILY:VALUE (family one of *, {', '.join(FAMILIES)}); got {text!r}")
    if family != "*" and family not in FAMILIES:
        raise ValueError(f"unknown framework family {family!r} in {text!r}; use *, {', '.join(FAMILIES)}")
    return family, rest


def build(
    env: list[str] | None = None,
    serve_args: list[str] | None = None,
    options: list[str] | None = None,
    nsys_dir: str | None = None,
    nsys_graph_trace: str = "node",
    nsys_flush_ms: int | None = None,
) -> Overrides:
    result = Overrides()
    for item in env or []:
        family, pair = _split_family(item, "--env")
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--env wants FAMILY:KEY=VALUE, got {item!r}")
        result.env.setdefault(family, {})[key] = value
    for item in serve_args or []:
        family, arg = _split_family(item, "--serve-arg")
        result.serve_args.setdefault(family, []).append(arg)
    for item in options or []:
        key, sep, value = item.partition("=")
        if not sep or "." not in key:
            raise ValueError(f"--opt wants TASK.KEY=VALUE, got {item!r}")
        result.options[key] = value
    if nsys_dir:
        result.nsys = Nsys(out_dir=nsys_dir, graph_trace=nsys_graph_trace, flush_ms=nsys_flush_ms)
    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "overrides",
        "Counterfactual and instrumentation switches. Recorded in every manifest; nothing is read from the environment.",
    )
    group.add_argument("--env", action="append", default=[], metavar="FAMILY:KEY=VALUE",
                       help="Environment variable for the serve process of one framework family; repeatable.")
    group.add_argument("--serve-arg", action="append", default=[], metavar="FAMILY:ARG",
                       help="Extra serve-command argument for one framework family; repeatable.")
    group.add_argument("--opt", action="append", default=[], metavar="TASK.KEY=VALUE",
                       help="Task option, e.g. pi05.vllm_graph=1; repeatable.")
    group.add_argument("--nsys-dir", default=None,
                       help="Wrap serve commands in `nsys profile`, writing reports here (profiling cells only).")
    group.add_argument("--nsys-graph-trace", default="node", choices=("node", "graph"))
    group.add_argument("--nsys-flush-ms", type=int, default=None)


def from_args(args: argparse.Namespace) -> Overrides:
    return build(
        env=getattr(args, "env", None),
        serve_args=getattr(args, "serve_arg", None),
        options=getattr(args, "opt", None),
        nsys_dir=getattr(args, "nsys_dir", None),
        nsys_graph_trace=getattr(args, "nsys_graph_trace", "node"),
        nsys_flush_ms=getattr(args, "nsys_flush_ms", None),
    )
