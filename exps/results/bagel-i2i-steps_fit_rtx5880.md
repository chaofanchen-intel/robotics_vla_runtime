# BAGEL ablation: fixed vs marginal cost

## T2I — E2E p50 vs denoise_steps (median over starts, ms)

| framework | points | a (fixed ms) | b (ms/unit) | R² | usable |
|---|---|---:|---:|---:|---|

Gap to M* at baseline:

- status: reference_fit_unusable

## I2I — E2E p50 vs denoise_steps (median over starts, ms)

| framework | points | a (fixed ms) | b (ms/unit) | R² | usable |
|---|---|---:|---:|---:|---|
| mstar | 5:2,761 [2,759–2,763], 25:11,784 [11,783–11,785], 50:23,643 [23,624–23,661] | 344 | 464.5 | 1.000 | yes |
| vllm-omni-single-stage | 5:2,909 [2,907–2,911], 25:12,030 [12,026–12,033], 50:23,926 [23,907–23,945] | 491 | 467.4 | 1.000 | yes |
| vllm-omni-two-stage | 5:6,865 [6,856–6,874], 25:16,059 [16,034–16,084], 50:28,161 [28,158–28,165] | 4,397 | 473.7 | 1.000 | yes |

Gap to M* at baseline:

- vllm-omni-single-stage: gap 295 ms = fixed 147 (50%) + marginal 148 (50%) → **mixed**
- vllm-omni-two-stage: gap 4,515 ms = fixed 4,053 (90%) + marginal 462 (10%) → **mostly_fixed_per_request_cost**

## I2T — E2E p50 vs completion_tokens (median over starts, ms)

| framework | points | a (fixed ms) | b (ms/unit) | R² | usable |
|---|---|---:|---:|---:|---|

Gap to M* at baseline:

- status: reference_fit_unusable

Limits: single GPU, C=1, 256x256; localises the gap, does not prove the graph abstraction causes it. See the module docstring for the pre-registered reading.
