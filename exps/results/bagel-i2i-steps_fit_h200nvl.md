# BAGEL ablation: fixed vs marginal cost

## T2I — E2E p50 vs denoise_steps (median over starts, ms)

| framework | points | a (fixed ms) | b (ms/unit) | R² | usable |
|---|---|---:|---:|---:|---|

Gap to M* at baseline:

- status: reference_fit_unusable

## I2I — E2E p50 vs denoise_steps (median over starts, ms)

| framework | points | a (fixed ms) | b (ms/unit) | R² | usable |
|---|---|---:|---:|---:|---|
| mstar | 5:825 [825–826], 25:2,722 [2,720–2,724], 50:5,116 [5,114–5,117] | 345 | 95.4 | 1.000 | yes |
| vllm-omni-single-stage | 5:1,166 [1,145–1,187], 25:3,596 [3,575–3,616], 50:6,504 [6,502–6,505] | 595 | 118.5 | 1.000 | yes |
| vllm-omni-two-stage | 5:4,985 [4,956–5,013], 25:7,547 [7,500–7,593], 50:10,484 [10,411–10,558] | 4,418 | 122.0 | 0.999 | yes |

Gap to M* at baseline:

- vllm-omni-single-stage: gap 1,408 ms = fixed 250 (18%) + marginal 1,158 (82%) → **mostly_per_step_or_per_token_cost**
- vllm-omni-two-stage: gap 5,406 ms = fixed 4,073 (75%) + marginal 1,333 (25%) → **mostly_fixed_per_request_cost**

## I2T — E2E p50 vs completion_tokens (median over starts, ms)

| framework | points | a (fixed ms) | b (ms/unit) | R² | usable |
|---|---|---:|---:|---:|---|

Gap to M* at baseline:

- status: reference_fit_unusable

Limits: single GPU, C=1, 256x256; localises the gap, does not prove the graph abstraction causes it. See the module docstring for the pre-registered reading.
