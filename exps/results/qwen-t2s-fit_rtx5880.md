| campaign | framework | case | n | rows | stalls | audio s | E2E ms | ms / audio s | TTFA ms (steady) | after first audio, ms / audio s | fit a ms (all / steady) | fit b ms per audio s (all / steady) | R² (steady) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| QWEN_T2S_FIT_MSTAR_02_B2B | mstar | verbatim-en | 40 | 20 | 0 | 3.54 | 335 | 96.0 | 154 | 50.9 | 101 / 101 | 66.1 / 66.1 | 1.00 |

stall = a request whose time-to-first-audio exceeds twice its cell's median. `steady` excludes stalls. a = fixed latency, b = ms per second of audio produced. Single GPU set, C=1, one fixture family; localises the gap, does not name the kernel.
