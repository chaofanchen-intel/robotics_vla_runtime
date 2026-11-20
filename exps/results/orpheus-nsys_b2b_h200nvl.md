# Orpheus TTS: GPU timeline of one request (b1) and one 16-request wave (b16) in M* and VoxServe (nsys)

**Question.** baseline-orpheus: M* leads VoxServe 1.09x at b1 and 1.38x at b16 (E2E 2,531 against 2,751 and 5,000 against 6,925 ms) with sampling matched. The lead grows with the batch. Is VoxServe's extra time GPU work (a slower decode step, a slower SNAC decoder) or gaps between launches, and does the gap per step grow with the batch?

**Pre-registered reading.** analyze_nsys_burst_gaps.py per framework and case (the last burst): span against busy and the launch count per decode step. Equal busy and a longer VoxServe span = scheduling / launch gaps; unequal busy = kernels (read the top-kernel list). Compare b1 against b16 for how the gap scales.

**Caveats.** No fairness audit of VoxServe's serve flags beyond the matched sampling (its CUDA graph / max batch settings are what `frameworks.voxserve_serve` passes). Two measured requests per cell: structure, not speed.

Campaigns: `ORPHEUS_NSYS_MSTAR_01_B2B` (mstar, back-to-back), `ORPHEUS_NSYS_VOXSERVE_01_B2B` (voxserve, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs mstar |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | b1 | mstar | 1 | 2 | 1,678 [1,678–1,678] | 100% | 59 | 1,785 | - |
| mstar | b16 | mstar | 1 | 2 | 2,581 [2,581–2,581] | 100% | 60 | 1,785 | - |
| voxserve | b1 | voxserve | 1 | 2 | 10,511 [10,511–10,511] | 100% | 61 | 1,785 | - |
| voxserve | b16 | voxserve | 1 | 2 | 4,601 [4,601–4,601] | 100% | 62 | 1,785 | - |
