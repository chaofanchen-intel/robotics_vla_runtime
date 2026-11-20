| resolution | engine | variant | CFG branches | starts | E2E p50 ms [min-max] | mean SM clock MHz | ms per extra branch per step |
|---:|---|---|---:|---:|---:|---:|---:|
| 512 | mstar | nocfg | 1 | 2 | 7,412 [7,386–7,438] | 1,379 | - |
| 512 | mstar | default | 2 | 2 | 12,984 [12,966–13,003] | 1,283 | 111.4 |
| 512 | mstar | cfg3 | 3 | 2 | 18,834 [18,795–18,872] | 1,205 | 114.2 |
| 512 | vllm-omni-single-stage | nocfg | 1 | 2 | 7,437 [7,436–7,439] | 1,469 | - |
| 512 | vllm-omni-single-stage | default | 2 | 2 | 13,053 [13,039–13,066] | 1,371 | 112.3 |
| 512 | vllm-omni-single-stage | cfg3 | 3 | 2 | 18,821 [18,806–18,836] | 1,295 | 113.8 |
| 512 | vllm-omni-two-stage | nocfg | 1 | 2 | 7,642 [7,641–7,644] | 2,460 | - |
| 512 | vllm-omni-two-stage | default | 2 | 2 | 13,442 [13,414–13,470] | 2,460 | 116.0 |
| 512 | vllm-omni-two-stage | cfg3 | 3 | 2 | 19,335 [19,320–19,349] | 2,292 | 116.9 |

One branch (CFG off: same work AND same output semantics on both engines), vLLM / M*:

| resolution | vLLM deployment | M* ms | vLLM ms | vLLM / M* |
|---:|---|---:|---:|---:|
| 512 | vllm-omni-single-stage | 7,412 | 7,437 | 1.00x |
| 512 | vllm-omni-two-stage | 7,412 | 7,642 | 1.03x |

Three branches on both engines (vLLM with cfg_img_scale=2.0, so its output differs from the default request), vLLM / M*:

| resolution | vLLM deployment | M* ms | vLLM ms | vLLM / M* |
|---:|---|---:|---:|---:|
| 512 | vllm-omni-single-stage | 18,834 | 18,821 | 1.00x |
| 512 | vllm-omni-two-stage | 18,834 | 19,335 | 1.03x |

Both engines run two branches on the `default` request (cfg_img_scale=1.0: text guidance only) and three on `cfg3`, so `default` is equal work too; `ms per extra branch per step` should agree between the two variants of one engine. Clocks are the per-request means from the harness; an engine that draws more power per step sits lower on the same card, so compare the clock column before reading a ratio as the engine's.
