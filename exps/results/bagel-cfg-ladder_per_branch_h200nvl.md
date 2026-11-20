| resolution | engine | variant | CFG branches | starts | E2E p50 ms [min-max] | mean SM clock MHz | ms per extra branch per step |
|---:|---|---|---:|---:|---:|---:|---:|
| 512 | mstar | nocfg | 1 | 2 | 2,096 [1,980–2,213] | 1,713 | - |
| 512 | mstar | default | 2 | 2 | 2,865 [2,830–2,899] | 1,583 | 15.4 |
| 512 | mstar | cfg3 | 3 | 2 | 3,865 [3,858–3,872] | 1,440 | 17.7 |
| 512 | vllm-omni-single-stage | nocfg | 1 | 2 | 2,993 [2,940–3,046] | 1,785 | - |
| 512 | vllm-omni-single-stage | default | 2 | 2 | 4,177 [4,077–4,277] | 1,761 | 23.7 |
| 512 | vllm-omni-single-stage | cfg3 | 3 | 2 | 5,341 [5,330–5,352] | 1,672 | 23.5 |
| 512 | vllm-omni-two-stage | nocfg | 1 | 2 | 2,636 [2,530–2,741] | 1,780 | - |
| 512 | vllm-omni-two-stage | default | 2 | 2 | 4,026 [3,849–4,204] | 1,759 | 27.8 |
| 512 | vllm-omni-two-stage | cfg3 | 3 | 2 | 5,139 [5,138–5,141] | 1,661 | 25.0 |

One branch (CFG off: same work AND same output semantics on both engines), vLLM / M*:

| resolution | vLLM deployment | M* ms | vLLM ms | vLLM / M* |
|---:|---|---:|---:|---:|
| 512 | vllm-omni-single-stage | 2,096 | 2,993 | 1.43x |
| 512 | vllm-omni-two-stage | 2,096 | 2,636 | 1.26x |

Three branches on both engines (vLLM with cfg_img_scale=2.0, so its output differs from the default request), vLLM / M*:

| resolution | vLLM deployment | M* ms | vLLM ms | vLLM / M* |
|---:|---|---:|---:|---:|
| 512 | vllm-omni-single-stage | 3,865 | 5,341 | 1.38x |
| 512 | vllm-omni-two-stage | 3,865 | 5,139 | 1.33x |

Both engines run two branches on the `default` request (cfg_img_scale=1.0: text guidance only) and three on `cfg3`, so `default` is equal work too; `ms per extra branch per step` should agree between the two variants of one engine. Clocks are the per-request means from the harness; an engine that draws more power per step sits lower on the same card, so compare the clock column before reading a ratio as the engine's.
