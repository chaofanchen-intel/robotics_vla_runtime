# Pi0.5 SGLang core under nsys: why did the H200 capture hold no kernels, and which variant records them?

**Question.** pi05-nsys on the H200 captured SGLang core's server and its worker process but no CUDA kernel at all (1,128 runtime calls, 564 per process, no CUPTI_ACTIVITY_KIND_KERNEL table; M* and vLLM-Omni captured 195k and 14k launches in the same harness; the rtx5880 capture of the same arm is fine; torch, CUDA and CUPTI are 13.0 / 13.0.85 in all three environments, so it is not a library mismatch). Is it node-level tracing of the graphs SGLang captures lazily, the buffers of a worker that is stopped by SIGTERM, or something the CUDA graphs have to do with at all?

**Pre-registered reading.** analyze_nsys_burst_gaps.py per arm: the arm whose capture has kernels names the cause. graph-trace-graph (`--cuda-graph-trace=graph`, no per-node tracing) recording kernels = node-level tracing of lazily captured graphs; flush-100ms recording them = buffers lost at shutdown (a shorter interval bounds the loss); no-cuda-graph recording them while the other two do not = the graphs are the cause, and that capture is a different workload (SGLang with graphs off, 10 ms of launch gaps per step), so read it as structure only. None recording = not the capture options; read server.log and try another nsys build (the rtx5880 box captures with ~/.local/opt/nsys-2026.6.1, this one with /usr/local/bin/nsys 2026.4.1). Three one-cell arms, ~10 minutes. The first arm that records kernels is also the missing SGLang-core column of pi05-nsys.

**Caveats.** A diagnostic of the capture, not a measurement: nsys inflates host time. Each arm is one start of one cell.

Campaigns: `PI05_NSYS_SGLANG_DIAG_GRAPH_TRACE_GRAPH_01_B2B` (graph-trace-graph, back-to-back), `PI05_NSYS_SGLANG_DIAG_FLUSH_100MS_01_B2B` (flush-100ms, back-to-back), `PI05_NSYS_SGLANG_DIAG_NO_CUDA_GRAPH_01_B2B` (no-cuda-graph, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs graph-trace-graph |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-core | droid-46row | graph-trace-graph | 1 | 4 | 232 [232–232] | 100% | 54 | 1,785 | - |
| sglang-core | droid-46row | flush-100ms | 1 | 4 | 175 [175–175] | 100% | 55 | 1,785 | 0.75x |
| sglang-core | droid-46row | no-cuda-graph | 1 | 4 | 201 [201–201] | 100% | 56 | 1,785 | 0.87x |
