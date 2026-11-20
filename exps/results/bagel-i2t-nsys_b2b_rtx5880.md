# BAGEL I2T: where do vLLM-Omni's extra ~650 ms before the first token go? (nsys)

**Question.** Per-token cost is equal (16.9 vs 17.1 ms) but time to first token is 146 ms on M* and 808 ms on vLLM-Omni two-stage, constant from 512^2 to 1024^2. Is the difference GPU work or idle time before the first kernel?

**Pre-registered reading.** Gap before each request's first kernel burst and the bursts' busy time (analyze_nsys_burst_gaps.py). A 600+ ms idle gap on vLLM with equal busy time is CPU / front end (preprocessing, stage hand-off); equal gaps and more vLLM busy time would be GPU work. Structure only: nsys inflates host time.

Campaigns: `BAGEL_I2T_NSYS_NSYS_01_B2B` (nsys, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs nsys |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | i2t-512 | nsys | 1 | 4 | 2,308 [2,308–2,308] | 0% | 48 | 2,608 | - |
| vllm-omni-two-stage | i2t-512 | nsys | 1 | 4 | 2,982 [2,982–2,982] | 0% | 47 | 2,286 | - |
