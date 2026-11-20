# BAGEL ablation: fixed vs marginal cost

## T2I — E2E p50 vs denoise_steps (median over starts, ms)

| framework | points | a (fixed ms) | b (ms/unit) | R² | usable |
|---|---|---:|---:|---:|---|
| mstar | 5:437 [410–465], 25:1,538 [1,515–1,561], 50:2,902 [2,885–2,920] | 165 | 54.8 | 1.000 | yes |
| vllm-omni-single-stage | 5:773 [771–775], 25:2,386 [2,358–2,414], 50:4,236 [4,228–4,244] | 416 | 76.8 | 0.999 | yes |
| vllm-omni-two-stage | 5:714 [712–716], 25:2,240 [2,207–2,272], 50:3,884 [3,770–3,998] | 406 | 70.3 | 0.998 | yes |

Gap to M* at baseline:

- vllm-omni-single-stage: gap 1,354 ms = fixed 250 (18%) + marginal 1,104 (82%) → **mostly_per_step_or_per_token_cost**
- vllm-omni-two-stage: gap 1,015 ms = fixed 240 (24%) + marginal 775 (76%) → **mostly_per_step_or_per_token_cost**

## I2I — E2E p50 vs denoise_steps (median over starts, ms)

| framework | points | a (fixed ms) | b (ms/unit) | R² | usable |
|---|---|---:|---:|---:|---|

Gap to M* at baseline:

- status: reference_fit_unusable

## I2T — E2E p50 vs completion_tokens (median over starts, ms)

| framework | points | a (fixed ms) | b (ms/unit) | R² | usable |
|---|---|---:|---:|---:|---|

Gap to M* at baseline:

- status: reference_fit_unusable

Limits: single GPU, C=1, 256x256; localises the gap, does not prove the graph abstraction causes it. See the module docstring for the pre-registered reading.
