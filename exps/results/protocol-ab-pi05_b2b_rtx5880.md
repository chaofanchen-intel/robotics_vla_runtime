# Measurement protocol A/B on pi05: gated vs back-to-back vs fixed 30 s gap

**Question.** Does the protocol itself change this task's latency, and for which framework? (Pi0.5: vLLM-Omni 333 ms gated vs 222 ms back-to-back at identical temperature and clock; BAGEL throttles without the gate.)

**Pre-registered reading.** gated is the reference. back-to-back differs: the cooldown gate / idle matters for this task. gap30 matches gated: the idle time matters but not the thermal condition; gap30 matches back-to-back: the gate's cool, full-clock card does. Compare the per-request temperature and SM clock columns before reading any ratio.

Campaigns: `PROTOCOL_AB_PI05_GATED_02` (gated, gated), `PROTOCOL_AB_PI05_B2B_02` (b2b, back-to-back), `PROTOCOL_AB_PI05_GAP30_02` (gap30, back-to-back, gap 30 s)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs gated |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | droid-46row | gated | 2 | 24 | 118 [116–122] | 0% | 30 | 304 | - |
| mstar | droid-46row | b2b | 2 | 24 | 99.8 [99.3–100.2] | 0% | 35 | 2,460 | 0.85x |
| mstar | droid-46row | gap30 | 2 | 24 | 117 [115–117] | 0% | 30 | 304 | 0.99x |
| sglang-core | droid-46row | gated | 2 | 24 | 174 [173–175] | 0% | 30 | 1,982 | - |
| sglang-core | droid-46row | b2b | 2 | 24 | 164 [162–164] | 0% | 34 | 2,058 | 0.94x |
| sglang-core | droid-46row | gap30 | 2 | 24 | 178 [177–178] | 0% | 30 | 1,998 | 1.02x |
| vllm-omni | droid-46row | gated | 2 | 24 | 242 [240–244] | 0% | 30 | 2,288 | - |
| vllm-omni | droid-46row | b2b | 2 | 24 | 211 [208–213] | 0% | 34 | 2,415 | 0.87x |
| vllm-omni | droid-46row | gap30 | 2 | 24 | 241 [240–246] | 0% | 30 | 2,346 | 0.99x |
