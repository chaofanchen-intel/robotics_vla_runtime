| campaign | framework | case | n | rows | stalls | audio s | E2E ms | ms / audio s | TTFA ms (steady) | after first audio, ms / audio s | fit a ms (all / steady) | fit b ms per audio s (all / steady) | R² (steady) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| QWEN_MSTAR_COMPILE_T2S_A1_01_B2B | mstar | verbatim-en | 20 | 10 | 0 | 3.66 | 299 | 81.4 | 153 | 38.7 | 95 / 95 | 54.7 / 54.7 | 0.96 |
| QWEN_MSTAR_COMPILE_T2S_COMPILE_ON_01_B2B | mstar | verbatim-en | 20 | 10 | 0 | 3.66 | 296 | 80.3 | 151 | 39.1 | 90 / 90 | 55.3 / 55.3 | 0.96 |
| QWEN_MSTAR_COMPILE_T2S_A2_01_B2B | mstar | verbatim-en | 20 | 10 | 0 | 3.66 | 294 | 81.0 | 151 | 39.3 | 91 / 91 | 55.0 / 55.0 | 0.98 |

stall = a request whose time-to-first-audio exceeds twice its cell's median. `steady` excludes stalls. a = fixed latency, b = ms per second of audio produced. Single GPU set, C=1, one fixture family; localises the gap, does not name the kernel.
