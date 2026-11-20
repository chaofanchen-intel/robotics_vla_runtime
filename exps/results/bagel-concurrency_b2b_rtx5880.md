# BAGEL I2T: B = 1..16 concurrent requests

**Question.** Throughput scaling of the AR path with batch (the paper's I2T batch claim).

**Pre-registered reading.** Wave time vs B per framework; scaling ~1.0 means serialised waves.

Campaigns: `BAGEL_CONCURRENCY_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | i2t-512 | main | 2 | 10 | 2,284 [2,276–2,285] | 0% | 51 | 2,536 | - |
| mstar | i2t-512-b4 | main | 2 | 10 | 2,661 [2,660–2,662] | 0% | 50 | 2,662 | - |
| mstar | i2t-512-b16 | main | 2 | 10 | 3,641 [3,634–3,647] | 0% | 52 | 2,487 | - |
| vllm-omni-two-stage | i2t-512 | main | 2 | 10 | 2,978 [2,977–2,978] | 0% | 45 | 2,369 | - |
| vllm-omni-two-stage | i2t-512-b4 | main | 2 | 10 | 5,535 [5,524–5,547] | 0% | 49 | 1,980 | - |
| vllm-omni-two-stage | i2t-512-b16 | main | 2 | 10 | 16,126 [16,070–16,181] | 0% | 60 | 1,513 | - |
| vllm-omni-single-stage | i2t-512 | main | 2 | 10 | 6,016 [6,015–6,041] | 0% | 48 | 2,512 | - |
| vllm-omni-single-stage | i2t-512-b4 | main | 2 | 10 | 24,226 [24,200–24,252] | 0% | 64 | 2,459 | - |
| vllm-omni-single-stage | i2t-512-b16 | main | 2 | 10 | 97,367 [97,327–97,509] | 0% | 73 | 2,411 | - |
