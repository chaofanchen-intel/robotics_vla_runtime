| campaign | framework | case | n | rows | stalls | audio s | E2E ms | ms / audio s | TTFA ms (steady) | after first audio, ms / audio s | fit a ms (all / steady) | fit b ms per audio s (all / steady) | R² (steady) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| QWEN_MSTAR_COMPILE_T2S_A1_02_B2B | mstar | verbatim-en | 20 | 10 | 0 | 3.86 | 366 | 95.7 | 154 | 55.3 | 92 / 92 | 71.2 / 71.2 | 1.00 |
| QWEN_MSTAR_COMPILE_T2S_COMPILE_ON_02_B2B | mstar | verbatim-en | 20 | 10 | 0 | 3.86 | 358 | 92.8 | 155 | 51.9 | 103 / 103 | 66.0 / 66.0 | 1.00 |
| QWEN_MSTAR_COMPILE_T2S_A2_02_B2B | mstar | verbatim-en | 20 | 10 | 0 | 3.86 | 367 | 95.1 | 154 | 55.8 | 95 / 95 | 70.3 / 70.3 | 0.99 |

stall = a request whose time-to-first-audio exceeds twice its cell's median. `steady` excludes stalls. a = fixed latency, b = ms per second of audio produced. Single GPU set, C=1, one fixture family; localises the gap, does not name the kernel.
