# BAGEL: is M*'s lead flash-attn? Attention call counts, flash_attn blocked, and the backends swapped

**Question.** M*'s README recommends flash-attn for BAGEL. Does M* use the pip `flash_attn` for the attention that matters, and does anyone else use flash attention? Source reading says the pip package is only in the ViT (T2I never runs it), M* LLM attention is FlashInfer and vLLM-Omni's generation step is FLASH_ATTN.

**Pre-registered reading.** (1) `mstar-probe`: T2I cases show vit_flash_attn_calls == 0 and the LLM attention managers are FlashInfer; I2T/I2I show > 0. (2) `mstar-no-flash-attn`: the ViT takes the SDPA fallback (vit_sdpa_fallback_calls > 0, flash calls == 0); T2I E2E unchanged vs mstar-probe within drift; I2T/I2I slower only by the ViT's share. (3) `vllm-sdpa`: vLLM-Omni's generation step on TORCH_SDPA instead of FLASH_ATTN; if its E2E barely moves, attention is not where the gap lives (kernel data already say <=4%). The ratio block pairs every vLLM arm with every M* arm; a ratio that survives all four pairings is not a flash-attn effect. On a GPU with no FlashAttention-3 kernel (RTX 5880 Ada is sm_89) check the vLLM log line for which backend it actually resolved.

**Caveats.** The probe patches imported M* modules in the worker (never external/); `proof` files show it applied.

Campaigns: `BAGEL_ATTN_BACKEND_MSTAR_PROBE_01_B2B` (mstar-probe, back-to-back), `BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01_B2B` (mstar-no-flash-attn, back-to-back), `BAGEL_ATTN_BACKEND_VLLM_DEFAULT_01_B2B` (vllm-default, back-to-back), `BAGEL_ATTN_BACKEND_VLLM_SDPA_01_B2B` (vllm-sdpa, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs mstar-probe |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-512-nocfg | mstar-probe | 2 | 10 | 7,423 [7,393–7,453] | 0% | 55 | 1,377 | - |
| mstar | t2i-512-nocfg | mstar-no-flash-attn | 2 | 10 | 7,436 [7,425–7,446] | 0% | 55 | 1,358 | 1.00x |
| mstar | i2i-512 | mstar-probe | 2 | 10 | 23,566 [23,543–23,588] | 0% | 65 | 1,254 | - |
| mstar | i2i-512 | mstar-no-flash-attn | 2 | 10 | 23,577 [23,565–23,589] | 0% | 65 | 1,244 | 1.00x |
| mstar | i2t-512 | mstar-probe | 2 | 10 | 2,299 [2,290–2,301] | 0% | 52 | 2,562 | - |
| mstar | i2t-512 | mstar-no-flash-attn | 2 | 10 | 2,290 [2,289–2,291] | 0% | 53 | 2,583 | 1.00x |
| vllm-omni-two-stage | t2i-512-nocfg | vllm-default | 2 | 10 | 7,641 [7,637–7,644] | 0% | 35 | 2,460 | - |
| vllm-omni-two-stage | t2i-512-nocfg | vllm-sdpa | 2 | 10 | 7,657 [7,642–7,672] | 0% | 34 | 2,460 | - |
| vllm-omni-two-stage | i2i-512 | vllm-default | 2 | 10 | 28,121 [28,075–28,166] | 0% | 34 | 1,818 | - |
| vllm-omni-two-stage | i2i-512 | vllm-sdpa | 2 | 10 | 31,412 [31,406–31,418] | 0% | 34 | 1,662 | - |
| vllm-omni-two-stage | i2t-512 | vllm-default | 2 | 10 | 2,976 [2,971–2,984] | 0% | 44 | 2,363 | - |
| vllm-omni-two-stage | i2t-512 | vllm-sdpa | 2 | 10 | 2,980 [2,979–2,981] | 0% | 43 | 2,354 | - |
