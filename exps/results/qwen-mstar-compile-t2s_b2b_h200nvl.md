# M* on Qwen3-Omni T2S: torch.compile off vs on (rtx5880: off for TXT / A2T, on for T2S since 2026-11-13)

**Question.** M* runs Qwen with TORCHDYNAMO_DISABLE=1 on the 5880 because the first smoke died with a Talker shared-memory OutOfResources. That kernel is M*'s own _cp_linear_kernel and SMOKE_5880_03 (with the flag already set) failed identically, so the flag was not the fix; the pinned M* now falls back to fewer stages / a shorter K step. Does M* compile on sm_89 now, what does it cost to start, and what is it worth?

**Pre-registered reading.** A1 / compile-on / A2: ratio beyond the A1-A2 spread. compile-on must have no OutOfResources / Traceback in server.log (the arm refuses to report otherwise). If it runs and is faster, M*'s Qwen numbers here understate it and the fair-profile arms should use `qwen.mstar_compile=on` (EXPS_QWEN_MSTAR_COMPILE=on); if it is slower or fails, the stale-workaround story is wrong and the failure is the thing to report. The arms pin the setting explicitly, because the per-task default is now on for T2S.

**Caveats.** Compile changes the first-seen-shape cost: the A2T / T2S arms use pre-warmed rows so a recompile does not land in the measured phase.

Campaigns: `QWEN_MSTAR_COMPILE_T2S_A1_01_B2B` (A1, back-to-back), `QWEN_MSTAR_COMPILE_T2S_COMPILE_ON_01_B2B` (compile-on, back-to-back), `QWEN_MSTAR_COMPILE_T2S_A2_01_B2B` (A2, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | verbatim-en | A1 | 2 | 20 | 299 [297–301] | 100% | 74 | 1,785 | - |
| mstar | verbatim-en | compile-on | 2 | 20 | 296 [294–299] | 100% | 74 | 1,785 | 0.99x † |
| mstar | verbatim-en | A2 | 2 | 20 | 294 [294–303] | 100% | 74 | 1,785 | 0.98x |

**A1/A2 drift** (the reference arm measured before and after the others): mstar verbatim-en 2%. `†` marks an arm whose ratio to A1 is no larger than that drift: by the pre-registered reading it cannot be attributed to the switch. If the drift is larger than every arm's effect the ablation is inconclusive for that case and must be re-measured, not interpreted.
