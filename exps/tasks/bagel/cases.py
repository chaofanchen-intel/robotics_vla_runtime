#!/usr/bin/env python3
"""BAGEL text-to-image, image editing and image-to-text.

Three tasks, three different ways the previous round failed
------------------------------------------------------------
**T2I** was measured over a 21-minute window while the card heat-soaked from
59 s to 140 s per request, with two warmups and a single start. Nothing about
the request was wrong; the measurement was. Fixed by the per-request cooldown
gate and by five starts instead of one.

**I2I** sent no width or height, so M* generated 1024x1024 while vLLM-Omni
generated 512x512 and the resulting 3.5x gap was reported as a speed ratio.
Fixed three ways at once: the source image is pre-resized to the exact target,
the size is sent explicitly to both engines, and a returned PNG whose
dimensions differ from the request voids the cell.

**I2T** sent M* only `max_tokens`, which M* ignores -- the key it reads is
`max_output_tokens`. M* also fell back to its config default temperature 0.6
while vLLM-Omni got greedy decoding, and the two-stage YAML allowed 2048
tokens at temperature 0.4. The cell was comparing a 2048-token completion
against a 128-token one. Fixed by sending both cap keys plus `ignore_eos` and
asserting the realised completion length per response.

Three recipes, not two engines
------------------------------
M*, vLLM-Omni single-stage, and vLLM-Omni two-stage. The last two are
different deployments of the same runtime and are kept separate, because
collapsing them would average two architectures into one number.

Resolutions
-----------
512 is the main operating point; 1024 is the paper's size and is available (explicit `--cases`, or
`baseline-bagel-1024`) for machines with stable cooling; 256 is kept for history and the host-bound regime. On the
throttling H200 NVL a single 1024 request is roughly four times the work of a 512 one and crosses the ~80 C
degradation knee partway through, so it measures the thermal limit, not the engine.

256 and 512 correspond to 256 and 1,024 latent tokens (side / 16, from the
VAE's 8x downsample and the DiT's patch size 2). DiT attention grows
quadratically in token count, so the two points give the scaling slope rather
than a single operating point.

Single-GPU topology
-------------------
The paper's main BAGEL figure uses 3xH100 with cross-device CFG parallelism.
Here there is one GPU, so `bagel_single_gpu.yaml` has no `LLM_cfg_text` or
`LLM_cfg_img` node group and the three CFG branches are merged into one batched
forward. This matches the paper's Appendix E single-GPU setting, not its
headline figure.
"""
from __future__ import annotations

from exps.core.registry import Case, TaskSpec

RESOLUTIONS = (256, 512, 1024)
TASKS = ("t2i", "i2i", "i2t")

#: Diffusion steps. Identical on both engines; asserted statically by preflight.
TIMESTEP_POINTS = 50
SERVED_MODEL_NAME = "bagel"

#: I2T completion length. Pinned with ignore_eos so every request decodes the
#: same number of tokens on every engine.
I2T_MAX_TOKENS = 128

#: Image seed. Affects pixels, not step count, so not a latency variable --
#: but pinned anyway so two runs of the same cell are comparable by eye.
SEED = 42

#: CFG settings per task, matching what M*'s own benchmark harness sets.
CFG_BY_TASK = {
    "t2i": {
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.4, 1.0],
        "cfg_renorm_type": "global",
        "cfg_renorm_min": 0.0,
    },
    "i2i": {
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 2.0,
        "cfg_interval": [0.0, 1.0],
        "cfg_renorm_type": "text_channel",
        "cfg_renorm_min": 0.0,
    },
}

TASK_LABELS = {
    "t2i": "text to image",
    "i2i": "image editing",
    "i2t": f"image to text, {I2T_MAX_TOKENS} tokens",
}


#: Decomposition ablations (`profiles=()`: only run via explicit `--case`, never in nightly/full).
#:
#: The headline BAGEL gap between engines could live in the per-step denoise cost (kernels, CFG
#: batching) or in work done once per request (prefill, KV hand-off between stages, VAE, image
#: encode). One E2E number cannot say which. Varying one knob and fitting ``E2E = a + b * knob``
#: per engine separates them: the intercept ``a`` is per-request fixed cost, the slope ``b`` is
#: per-step / per-token cost. Everything else (prompt, resolution, seed, CFG scales, source image)
#: stays at the baseline cell's values.
ABLATION_STEPS = (5, 10, 25)          # baseline is TIMESTEP_POINTS = 50
ABLATION_I2T_TOKENS = (16, 64, 256)   # baseline is I2T_MAX_TOKENS = 128
ABLATION_RESOLUTION = 512
#: Extra T2I resolutions for the launch-bound vs GPU-bound crossover (never in nightly/full).
SWEEP_RESOLUTIONS = (384, 448, 768)
#: CFG-branch ladder (T2I only). The harness sends cfg_text_scale=4.0, cfg_img_scale=1.0 to both engines. vLLM-Omni then runs two
#: branches (main + cfg_text: it adds the third only for cfg_img_scale > 1.0). Current M* does the same: LLMSubmodule.image_gen runs
#: two branches when image guidance is off (img scale <= 1). (An older M* repeated the packed sequence three times regardless and
#: discarded the cfg_img output; the RTX 5880 ladder, default 12.98 s vs cfg3 18.83 s at 512^2, shows that is no longer so.)
#: `-nocfg` is one branch on both, the default case is 2 on both, `-cfg3` sets cfg_img_scale=2.0 so both run three.
CFG_LADDER_RESOLUTIONS = (256, 512, 768)
CFG3_IMG_SCALE = 2.0
#: M* reads `num_timesteps` from the *server-side* model config (config.json), not from the request,
#: so the request-level `-steps{N}` cases are inert for M* (E2E stays at the 50-step value). These
#: cases start M* against a cache dir whose config.json carries `num_timesteps=N`
#: (see tools/make_bagel_steps_cache.py). Baseline (N=50) is the ordinary `t2i-256` / `i2i-256` case.
SERVER_STEPS = (5, 10, 25)
#: vLLM-Omni single-stage I2T runs at ~33 ms/token against ~5 ms for M* and two-stage. The
#: single-stage deploy config sets `enforce_eager: true`; these cases rerun the same token sweep
#: with `enforce_eager: false` to test whether the eager path explains it.
GRAPH_I2T_TOKENS = (16, 64, 128, 256)
#: Paper's I2T setting is single GPU, B=1,2,4,8,16 (B=1 is the baseline cell).
ABLATION_I2T_BATCH_SIZES = (2, 4, 8, 16)


def _ablation_cases() -> tuple[Case, ...]:
    cases: list[Case] = []
    for task in ("t2i", "i2i"):
        for steps in ABLATION_STEPS:
            cases.append(
                Case(
                    name=f"{task}-{ABLATION_RESOLUTION}-steps{steps}",
                    label=f"{TASK_LABELS[task]}, {steps} denoise steps (ablation)",
                    params={"task": task, "resolution": ABLATION_RESOLUTION, "timesteps": steps},
                    profiles=(),
                )
            )
        cases.append(
            Case(
                name=f"{task}-{ABLATION_RESOLUTION}-nocfg",
                label=f"{TASK_LABELS[task]}, CFG disabled: single branch (ablation)",
                params={"task": task, "resolution": ABLATION_RESOLUTION, "cfg_off": True},
                profiles=(),
            )
        )
    # vLLM-only (M* ignores request-level steps): 1024^2 with 5 steps exposes the Thinker -> DiT hand-off at four times the KV, cheaply.
    cases.append(
        Case(
            name="i2i-1024-steps5",
            label=f"{TASK_LABELS['i2i']} at 1024x1024, 5 denoise steps (hand-off scaling; request-level steps, so vLLM-Omni only)",
            params={"task": "i2i", "resolution": 1024, "timesteps": 5},
            profiles=(),
        )
    )
    for tokens in ABLATION_I2T_TOKENS:
        cases.append(
            Case(
                name=f"i2t-{ABLATION_RESOLUTION}-tokens{tokens}",
                label=f"image to text, {tokens} tokens (ablation)",
                params={"task": "i2t", "resolution": ABLATION_RESOLUTION, "max_tokens": tokens},
                profiles=(),
            )
        )
    for batch in ABLATION_I2T_BATCH_SIZES:
        cases.append(
            Case(
                name=f"i2t-{ABLATION_RESOLUTION}-b{batch}",
                label=f"image to text, {batch} concurrent requests per gated unit (concurrency)",
                params={"task": "i2t", "resolution": ABLATION_RESOLUTION, "batch_size": batch},
                profiles=(),
            )
        )
    # I2I concurrency at 5 steps (the step count where the hand-off and prefill are a large share): B concurrent requests per wave.
    # M* takes the step count from the server (srv-steps5), vLLM from the request (steps5).
    for batch in (4, 16):
        cases.append(
            Case(
                name=f"i2i-{ABLATION_RESOLUTION}-steps5-b{batch}",
                label=f"image editing, 5 denoise steps, {batch} concurrent requests (concurrency; request-level steps, vLLM-Omni only)",
                params={"task": "i2i", "resolution": ABLATION_RESOLUTION, "timesteps": 5, "batch_size": batch},
                profiles=(),
                not_applicable={"mstar": "M* ignores request-level steps; use the -srv-steps5 variant"},
            )
        )
        cases.append(
            Case(
                name=f"i2i-{ABLATION_RESOLUTION}-srv-steps5-b{batch}",
                label=f"image editing, M* server-side 5 denoise steps, {batch} concurrent requests (concurrency)",
                params={"task": "i2i", "resolution": ABLATION_RESOLUTION, "timesteps": 5, "server_timesteps": 5, "batch_size": batch},
                profiles=(),
                not_applicable={
                    "vllm-omni-single-stage": "request-level steps are honoured by vLLM; this case is the M* server-side variant",
                    "vllm-omni-two-stage": "request-level steps are honoured by vLLM; this case is the M* server-side variant",
                },
            )
        )
    # T2I concurrency at 5 steps: same two naming variants as the I2I block above.
    for batch in (4, 16):
        cases.append(
            Case(
                name=f"t2i-{ABLATION_RESOLUTION}-steps5-b{batch}",
                label=f"text to image, 5 denoise steps, {batch} concurrent requests (concurrency; request-level steps, vLLM-Omni only)",
                params={"task": "t2i", "resolution": ABLATION_RESOLUTION, "timesteps": 5, "batch_size": batch},
                profiles=(),
                not_applicable={"mstar": "M* ignores request-level steps; use the -srv-steps5 variant"},
            )
        )
        cases.append(
            Case(
                name=f"t2i-{ABLATION_RESOLUTION}-srv-steps5-b{batch}",
                label=f"text to image, M* server-side 5 denoise steps, {batch} concurrent requests (concurrency)",
                params={"task": "t2i", "resolution": ABLATION_RESOLUTION, "timesteps": 5, "server_timesteps": 5, "batch_size": batch},
                profiles=(),
                not_applicable={
                    "vllm-omni-single-stage": "request-level steps are honoured by vLLM; this case is the M* server-side variant",
                    "vllm-omni-two-stage": "request-level steps are honoured by vLLM; this case is the M* server-side variant",
                },
            )
        )
    vllm_na = {
        "vllm-omni-single-stage": "request-level steps are honoured by vLLM; this case is the M* server-side variant",
        "vllm-omni-two-stage": "request-level steps are honoured by vLLM; this case is the M* server-side variant",
    }
    for task in ("t2i", "i2i"):
        for steps in SERVER_STEPS:
            cases.append(
                Case(
                    name=f"{task}-{ABLATION_RESOLUTION}-srv-steps{steps}",
                    label=f"{TASK_LABELS[task]}, M* server-side {steps} denoise steps (ablation)",
                    params={"task": task, "resolution": ABLATION_RESOLUTION,
                            "timesteps": steps, "server_timesteps": steps},
                    profiles=(),
                    not_applicable=dict(vllm_na),
                )
            )
    graph_na = {
        "mstar": "the enforce_eager comparison only exists for the vLLM-Omni single-stage recipe",
        "vllm-omni-two-stage": "the enforce_eager comparison only exists for the vLLM-Omni single-stage recipe",
    }
    for tokens in GRAPH_I2T_TOKENS:
        cases.append(
            Case(
                name=f"i2t-{ABLATION_RESOLUTION}-graph-tokens{tokens}",
                label=f"image to text, {tokens} tokens, vLLM single-stage enforce_eager=false (ablation)",
                params={"task": "i2t", "resolution": ABLATION_RESOLUTION, "max_tokens": tokens,
                        "deploy_variant": "graph"},
                profiles=(),
                not_applicable=dict(graph_na),
            )
        )
    # Image tasks under the same enforce_eager=false recipe: with request-level launch overhead the
    # suspect for the per-step gap (nsys: equal GPU busy time per step, more launches in vLLM-Omni),
    # does torch.compile on the DiT path recover it?
    for task in ("t2i", "i2i"):
        cases.append(
            Case(
                name=f"{task}-{ABLATION_RESOLUTION}-graph",
                label=f"{TASK_LABELS[task]}, vLLM single-stage enforce_eager=false (ablation)",
                params={"task": task, "resolution": ABLATION_RESOLUTION, "deploy_variant": "graph"},
                profiles=(),
                not_applicable=dict(graph_na),
            )
        )
    for resolution in CFG_LADDER_RESOLUTIONS:
        if resolution != ABLATION_RESOLUTION:      # t2i-256-nocfg is already above
            cases.append(
                Case(
                    name=f"t2i-{resolution}-nocfg",
                    label=f"{TASK_LABELS['t2i']} at {resolution}x{resolution}, CFG disabled: single branch (CFG ladder)",
                    params={"task": "t2i", "resolution": resolution, "cfg_off": True},
                    profiles=(),
                )
            )
        cases.append(
            Case(
                name=f"t2i-{resolution}-cfg3",
                label=f"{TASK_LABELS['t2i']} at {resolution}x{resolution}, cfg_img_scale={CFG3_IMG_SCALE}: three branches on both "
                      "engines (CFG ladder)",
                params={"task": "t2i", "resolution": resolution, "cfg_img_scale": CFG3_IMG_SCALE},
                profiles=(),
            )
        )
    # Resolution sweep between the two headline points. At 256^2 the per-step cost is launch-bound
    # (GPU idle >=56%), at 512^2 it is GPU-bound and the engine order flips; 384^2 (576 latent
    # tokens) and 768^2 (2,304) locate the crossover. T2I only: no source image to re-render.
    for resolution in SWEEP_RESOLUTIONS:
        cases.append(
            Case(
                name=f"t2i-{resolution}",
                label=f"{TASK_LABELS['t2i']} at {resolution}x{resolution} (resolution sweep; default request: NOT equal work)",
                params={"task": "t2i", "resolution": resolution},
                profiles=(),
            )
        )
    return tuple(cases)


def _headline_cases() -> tuple[Case, ...]:
    """The cases a headline number may come from: equal work on every engine.

    T2I is run at one CFG branch (``-nocfg``: same work *and* same output semantics everywhere) and at three
    (``-cfg3``: vLLM-Omni gets ``cfg_img_scale=2.0`` so it also computes three branches, as M* always does).
    I2I is three branches on both engines by construction, I2T has no CFG. The plain default T2I request
    (``cfg_img_scale=1.0``) is two branches on both engines in the current M* (``-cfg3`` forces three), so it exists only as
    ``t2i-<res>-default``, outside every profile, as the two-branch reference.
    The nightly operating point is 256; ``full`` also measures 512 to expose the resolution slope.
    """
    cases: list[Case] = []
    for resolution in (512, 256, 1024):  # 512 first: the smoke profile takes the first case
        # 256 nightly (host-bound regime, kept for the H200 history), 512 the main operating point, 1024 explicit-only
        # (the paper's size; on a throttling chassis one request crosses the thermal knee, so it is meant for stable cooling)
        profiles = {256: ("nightly", "full"), 512: ("full",)}.get(resolution, ())
        side = f"{resolution}x{resolution}"
        cases += [
            Case(name=f"t2i-{resolution}-nocfg", profiles=profiles,
                 label=f"{TASK_LABELS['t2i']} at {side}, one CFG branch (equal work, equal output semantics)",
                 params={"task": "t2i", "resolution": resolution, "cfg_off": True}),
            Case(name=f"t2i-{resolution}-cfg3", profiles=profiles,
                 label=f"{TASK_LABELS['t2i']} at {side}, three CFG branches on every engine (equal work)",
                 params={"task": "t2i", "resolution": resolution, "cfg_img_scale": CFG3_IMG_SCALE}),
            Case(name=f"i2i-{resolution}", profiles=profiles,
                 label=f"{TASK_LABELS['i2i']} at {side} (three branches on every engine)",
                 params={"task": "i2i", "resolution": resolution}),
            Case(name=f"i2t-{resolution}", profiles=profiles,
                 label=f"{TASK_LABELS['i2t']} at {side}",
                 params={"task": "i2t", "resolution": resolution}),
            Case(name=f"t2i-{resolution}-default", profiles=(),
                 label=f"{TASK_LABELS['t2i']} at {side}, default request: NOT equal work "
                       "(M* computes 3 CFG branches, vLLM-Omni 2) -- reference only",
                 params={"task": "t2i", "resolution": resolution}),
        ]
    return tuple(cases)


def _all_cases() -> tuple[Case, ...]:
    headline = _headline_cases()
    taken = {case.name for case in headline}
    return headline + tuple(case for case in _ablation_cases() if case.name not in taken)


TASK = TaskSpec(
    name="bagel",
    title="BAGEL text-to-image / image editing / image-to-text",
    frameworks=("mstar", "vllm-omni-single-stage", "vllm-omni-two-stage"),
    cases=_all_cases(),
    runner_module="exps.tasks.bagel.cell",
    comparison_label="work_matched",
    timing_boundary=(
        "Anchored immediately before the HTTP send and closed at the end of the response body. "
        "Base64 decoding, PNG dimension validation, hashing and disk writes happen after the "
        "timer. For I2I the source image is read and encoded by RequestInput before the timer "
        "starts, so no engine is charged for the client's file I/O."
    ),
    notes=(
        "Every response is checked: generated images must be exactly the requested size, and I2T "
        "completions must be exactly the requested length. A mismatch voids the cell. "
        "1024x1024 is deliberately not measured this round."
    ),
)
