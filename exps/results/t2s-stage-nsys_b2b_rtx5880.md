# T2S: per-stage GPU timeline (nsys), three frameworks

**Question.** Where does each framework spend GPU time per stage (Thinker / Talker / Code2Wav)?

**Pre-registered reading.** Busy per audio second by stage. nsys inflates the timeline 1.6-2.2x, M* most: structure only, never E2E.

Campaigns: `T2S_STAGE_NSYS_NSYS_01_B2B` (nsys, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs nsys |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | verbatim-en | nsys | 1 | 3 | 408 [408–408] | 0% | 44 | 2,685 | - |
| sglang-omni | verbatim-en | nsys | 1 | 3 | 491 [491–491] | 0% | 38 | 2,460 | - |
| vllm-omni | verbatim-en | nsys | 1 | 3 | 654 [654–654] | 0% | 37 | 2,620 | - |
