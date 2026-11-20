"""Per-machine counterfactual configs: the `{variant:KEY}` / `{hw:NAME}` placeholders, the one-card H200 files and the overlay guard.

The catalogue used to hard-code the rtx5880 multi-card overlay paths, so `--hardware h200nvl` planned a three-GPU TP=2 deployment on the
one card without a word. These tests pin the replacement: every key resolves on each machine, an H200 variant is the H200 default plus the
same change the 5880 file makes, another machine's overlay is refused, and rtx5880 keeps expanding to exactly the strings its recorded
campaigns were validated against. CPU only.
"""
from __future__ import annotations

import re

import pytest
import yaml

from exps import hardware
from exps.core import overrides, paths
from exps.suite import analysis, catalogue, runner
from exps.tools import h200_variants

PLACEHOLDER = re.compile(r"\{(variant|variant_file|variant_abs):([\w/]+)\}")
#: M* TP=2 only: one rank opens the prefix cache by itself, so there is no `prefix_cache_tp` recipe to port (and its experiment needs 2 GPUs)
NO_ONE_CARD_VERSION = {"mstar/qwen_prefix_cache_chunk4096"}


def _arm_strings(arm):
    yield from arm.env
    yield from arm.serve_args
    yield from arm.opts
    for name in ("command_has", "command_lacks", "log_has", "log_lacks"):
        yield from getattr(arm.expect, name)


def _catalogue_keys(max_gpus: int | None = None) -> set[str]:
    """Variant keys the catalogue uses; `max_gpus` leaves out experiments that need more cards than a machine has (they are skipped there,
    so their keys need no file on that machine: transport-bagel-cfg / transport-qwen-t2s use the 3- and 4-card mooncake overlays)."""
    keys: set[str] = set()
    for experiment in catalogue.ALL.values():
        if max_gpus is not None and experiment.min_gpus > max_gpus:
            continue
        for arm in experiment.arms:
            for text in _arm_strings(arm):
                keys.update(key for _, key in PLACEHOLDER.findall(text))
    return keys


def test_the_catalogue_uses_variants_and_no_longer_names_an_overlay_directory():
    assert len(_catalogue_keys()) >= 24
    for experiment in catalogue.ALL.values():
        for arm in experiment.arms:
            for text in _arm_strings(arm):
                assert "configs/hardware/" not in text, (experiment.name, arm.name, text)


def test_every_variant_key_resolves_to_an_existing_file_on_each_machine():
    profiles = hardware.profiles()
    for machine, skipped, max_gpus in (("rtx5880", set(), None), ("h200nvl", NO_ONE_CARD_VERSION, profiles["h200nvl"].gpu_count)):
        keys = _catalogue_keys(max_gpus)
        variants = profiles[machine].config_variants
        assert set(keys) - skipped <= set(variants), (machine, sorted(set(keys) - skipped - set(variants)))
        for key, relative in variants.items():
            assert (paths.CONFIGS / relative).is_file(), (machine, key, relative)
            owner = relative.split("/")[1] if relative.startswith("hardware/") else None
            assert owner in (None, machine), f"{machine} maps {key} into another machine's overlay: {relative}"


def test_h200_one_card_files_are_current_and_complete():
    """A change to a 5880 counterfactual, or to an H200 default, makes the generated file stale: regenerate with exps/tools/h200_variants.py."""
    assert h200_variants.main(["--check"]) == 0
    generated = {path.relative_to(h200_variants.OUT).as_posix() for path in h200_variants.outputs()}
    on_disk = {path.relative_to(h200_variants.OUT).as_posix() for path in h200_variants.OUT.rglob("*") if path.is_file()}
    assert on_disk == generated, f"unexpected hand-written files: {sorted(on_disk - generated)}"


def _leaves(changes, path=()):
    """The leaf edits of a `delta`: the same change reads as `nest` into a block the 5880 recipe spells out and as `set` of the whole block
    where the one-card default leaves it out (SGLang's `engine:`), so compare the leaves, not the tree shape."""
    out = {}
    for key, (kind, value) in changes.items():
        here = path + (("stage", key[1]) if isinstance(key, tuple) else (key,))
        if kind == "nest":
            out.update(_leaves(value, here))
        elif kind == "set" and isinstance(value, dict):
            out.update(_leaves({k: ("set", v) for k, v in value.items()}, here))
        else:
            out[here] = (kind, repr(value))
    return out


@pytest.mark.parametrize("spec", h200_variants.SPECS, ids=lambda spec: spec.target)
def test_an_h200_variant_makes_the_same_change_as_the_5880_file_on_top_of_the_h200_default(spec):
    h200_default = yaml.safe_load((paths.CONFIGS / spec.h200_base).read_text())
    h200 = yaml.safe_load((h200_variants.OUT / spec.target).read_text())
    got = h200_variants.delta(h200_default, h200) or {}
    if spec.ada_variant is None:
        want = {}
    else:
        want = h200_variants.delta(yaml.safe_load((h200_variants.ADA / spec.ada_base).read_text()),
                                   yaml.safe_load((h200_variants.ADA / spec.ada_variant).read_text()))
        for key in spec.skip:
            want.pop(key, None)
    if spec.extra:
        # the extra keys are set on top; whatever the 5880 change says about the same top-level key does not apply
        for key in spec.extra:
            want.pop(key, None)
            got.pop(key, None)
    assert _leaves(got) == _leaves(want), spec.target


def test_h200_cache_off_recipes_turn_the_thinker_prefix_cache_off_and_the_default_leaves_it_on():
    default = yaml.safe_load((paths.CONFIGS / "mstar" / "qwen3omni_1gpu.yaml").read_text())
    assert "resources" not in default                            # one rank: M* opens the Thinker prefix cache on its own
    for spec in h200_variants.SPECS:
        if spec.target.startswith("mstar/qwen"):
            config = yaml.safe_load((h200_variants.OUT / spec.target).read_text())
            assert config["resources"]["thinker_kv"]["prefix_cache"] is False, spec.target
            assert "prefix_cache_tp" not in config["resources"]["thinker_kv"]     # a TP=2 switch with no meaning here


def test_no_h200_plan_names_another_machines_overlay():
    hardware.set_current("h200nvl")
    planned = 0
    for experiment in catalogue.ALL.values():
        if not experiment.task or experiment.min_gpus > 1 or experiment.tool:
            continue
        for run in runner.arm_runs(experiment, protocol="back-to-back"):
            planned += 1
            assert "configs/hardware/rtx5880" not in " ".join(run.argv), (experiment.name, run.arm.name)
    assert planned > 200


def test_experiments_that_need_more_cards_than_the_machine_has_are_skipped_not_planned():
    for name in ("baseline-bagel-cfgpar", "mstar-tp-cache-probe"):
        assert catalogue.get(name).min_gpus > 1


def test_rtx5880_still_expands_to_the_strings_its_recorded_campaigns_were_validated_against():
    hardware.set_current("rtx5880")
    X = lambda values: runner.expand_all(values, campaign="CAMP")
    prefix = "exps/configs/hardware/rtx5880"
    assert X(catalogue.get("qwen-fair-a2t").arm("vllm-on").opts) == [
        "qwen.sampling=lookahead-eligible", "qwen.rows=cycle", f"config.vllm-omni={prefix}/vllm_omni/qwen3_omni_3gpu_thinker_tp2_prefix_cache.yaml"]
    # the M* default arm carries no config option on rtx5880 (TP=2 already means a cold prefill), so its command is what it always was
    assert X(catalogue.get("qwen-prefill-chunk").arm("chunk-1024").opts) == ["qwen.sampling=lookahead-eligible"]
    assert X(catalogue.get("bagel-connector-timeline").arm("shm").env) == [
        f"vllm-omni:VLLM_LOGGING_CONFIG_PATH={paths.REPO_ROOT}/{prefix}/vllm_omni/logging/ms_timestamps.json"]
    off = analysis._expanded_expect(catalogue.get("qwen-fair-txt").arm("mstar-off").expect, _any_run())
    assert "prefix cache off at world size" in off.log_has and "" not in off.log_has and "" not in off.log_lacks
    assert not any("open for" in needle for needle in off.log_lacks + off.log_has)
    warm = analysis._expanded_expect(catalogue.get("qwen-fair-a2t").arm("mstar-on").expect, _any_run())
    assert "prefix cache off at world size" in warm.log_lacks and "prefix cache open for" not in warm.log_has
    assert warm.command_has == ("qwen3omni_3gpu_thinker_tp2_prefix_cache.yaml",)


def test_h200_default_m_star_arms_get_the_cache_off_recipe_and_expect_no_open_cache():
    hardware.set_current("h200nvl")
    X = lambda values: runner.expand_all(values, campaign="CAMP")
    cold = "config.mstar=exps/configs/hardware/h200nvl/mstar/qwen3omni_1gpu_prefix_cache_off.yaml"
    for experiment, arm in (("qwen-prefill-chunk", "chunk-1024"), ("qwen-fair-txt", "mstar-off"), ("qwen-fair-txt", "mstar-pinned"),
                            ("qwen-ttft-txt", "mstar"), ("qwen-prefill-nsys", "mstar"), ("qwen-fair-a2t", "mstar-pinned")):
        assert cold in X(catalogue.get(experiment).arm(arm).opts), (experiment, arm)
    off = analysis._expanded_expect(catalogue.get("qwen-fair-txt").arm("mstar-off").expect, _any_run())
    assert "prefix cache open for" in off.log_lacks
    assert not any("off at world size" in needle for needle in off.log_has)
    # the arm that wants the cache ON runs the one-rank default and demands the 'open for' line
    on_arm = catalogue.get("qwen-fair-a2t").arm("mstar-on")
    assert X(on_arm.opts)[-1] == "config.mstar=exps/configs/mstar/qwen3omni_1gpu.yaml"
    on = analysis._expanded_expect(on_arm.expect, _any_run())
    assert "prefix cache open for" in on.log_has and "off at world size" not in " ".join(on.log_lacks)


def test_an_unmapped_variant_is_refused_instead_of_planning_another_machines_topology():
    hardware.set_current("h200nvl")
    with pytest.raises(SystemExit, match="no config variant 'mstar/qwen_prefix_cache_chunk4096'"):
        runner.expand_text(catalogue.QWEN_MSTAR_PREFIX_CHUNK4096, campaign="C")
    assert runner.expand_text("{hw:no_such_token}", campaign="C") == ""         # an undefined text token just drops out


def test_a_config_from_another_machines_overlay_is_refused_and_one_from_its_own_is_accepted(tmp_path):
    ada = paths.CONFIGS / "hardware" / "rtx5880" / "mstar" / "qwen3omni_3gpu_thinker_tp2.yaml"
    own = h200_variants.OUT / "mstar" / "qwen3omni_1gpu_chunk4096.yaml"
    default = paths.CONFIGS / "mstar" / "qwen3omni_1gpu.yaml"
    hardware.set_current("h200nvl")
    with pytest.raises(SystemExit, match="another machine"):
        hardware.check_config_path(ada)
    hardware.check_config_path(own)
    hardware.check_config_path(default)
    hardware.check_config_path(tmp_path / "anything.yaml")                       # outside configs/hardware/: not this guard's business
    # the same path through the option that carries it into a serve command
    refused = overrides.build(options=[f"config.mstar={ada}"])
    with pytest.raises(SystemExit, match="another machine"):
        refused.config_for("mstar", default)
    hardware.set_current("rtx5880")
    hardware.check_config_path(ada)
    with pytest.raises(SystemExit, match="another machine"):
        hardware.check_config_path(own)


def _any_run():
    experiment = catalogue.get("qwen-fair-txt")
    return runner.arm_runs(experiment, protocol="back-to-back")[0]


def _nsys_flush_ms(machine: str, arm_name: str) -> str:
    hardware.set_current(machine)
    run = next(r for r in runner.arm_runs(catalogue.get("pi05-nsys"), protocol="back-to-back") if r.arm.name == arm_name)
    return run.argv[run.argv.index("--nsys-flush-ms") + 1]


def test_pi05_sglang_nsys_capture_flushes_every_100_ms_on_h200_and_keeps_the_default_on_5880():
    """The H200 SGLang worker exits before a 1,000 ms CUDA flush and its capture holds no kernels; a 100 ms flush records them
    (pi05-nsys-sglang-diag, 2026-11-16). The 5880 command is unchanged, and the other two arms keep the default on both machines."""
    assert _nsys_flush_ms("h200nvl", "sglang-core") == "100"
    assert _nsys_flush_ms("rtx5880", "sglang-core") == "1000"
    for machine in ("h200nvl", "rtx5880"):
        assert _nsys_flush_ms(machine, "mstar") == "1000"
        assert _nsys_flush_ms(machine, "vllm") == "1000"
    assert catalogue.get("pi05-nsys-sglang-diag").arm("flush-100ms").nsys_flush_ms == 100     # the diagnostic's own arm stays an int
