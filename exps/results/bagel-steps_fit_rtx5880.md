# BAGEL ablation: fixed vs marginal cost

## T2I — E2E p50 vs denoise_steps (median over starts, ms)

| framework | points | a (fixed ms) | b (ms/unit) | R² | usable |
|---|---|---:|---:|---:|---|
| mstar | 5:1,224 [1,220–1,228], 25:6,262 [6,262–6,263], 50:13,009 [12,990–13,028] | -161 | 262.2 | 1.000 | yes |
| vllm-omni-single-stage | 5:1,319 [1,315–1,323], 25:6,379 [6,351–6,407], 50:13,043 [13,033–13,052] | -40 | 260.8 | 1.000 | yes |
| vllm-omni-two-stage | 5:1,393 [1,383–1,403], 25:6,589 [6,589–6,590], 50:13,465 [13,443–13,487] | -12 | 268.5 | 1.000 | yes |

Gap to M* at baseline:

- vllm-omni-single-stage: gap 48 ms = fixed 121 (251%) + marginal -73 (-151%) → **mostly_fixed_per_request_cost**
- vllm-omni-two-stage: gap 465 ms = fixed 149 (32%) + marginal 316 (68%) → **mostly_per_step_or_per_token_cost**

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
