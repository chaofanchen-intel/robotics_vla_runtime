# Qwen3-Omni TXT 1,072-token prompt on M*, budget 4,096: nsys WITH the 2,048 prefill graph (KV pool halved), against the same config without nsys

**Question.** qwen-prefill-chunk-nsys measured the budget-4,096 arm with NO 2,048 prefill graph (nsys itself takes the memory; `captured 52 of 74`), so its 1,072-token step ran eagerly (E2E 123 ms) instead of in the padded 2,048 graph the production server uses (E2E 163-168 ms). How long is the padded forward on the GPU, and is it ~2x the unpadded one (GEMM / attention / MoE kernels at 2,048 tokens)? Halving the Thinker KV pool (512 -> 256 pages) is the attempt to leave room for that graph under nsys.

**Pre-registered reading.** First the capture line: the nsys arms' Expect needs `Captured accelerator graph for Thinker: prefill_text[bs=32,tokens=2048]` in server.log; without it the cell is rejected and the answer is `nsys under this config still drops the graph`. Then room-plain (no nsys) E2E ~160-168 = this config still pads to 2,048; room-nsys E2E ~ room-plain = nsys no longer changes the path. Kernel ms of room-nsys's one prompt forward against the eager 1,072 forward of qwen-prefill-chunk-nsys (span 105-112 / busy 85-93 / MoE 34 ms): dense GEMM + attention ~2x and MoE up = padding to 2,048 tokens is real GPU work. room-*-again are the A/B/A/B order check.

**Caveats.** One prompt length (1,072 tokens). The smaller KV pool is a second change next to nsys; room-plain is the control for it (it must still show ~160 ms, not 123). nsys inflates host time; kernel ms are the evidence.

Campaigns: `QWEN_PREFILL_NSYS_ROOM_ROOM_NSYS_01_B2B` (room-nsys, back-to-back), `QWEN_PREFILL_NSYS_ROOM_ROOM_PLAIN_01_B2B` (room-plain, back-to-back), `QWEN_PREFILL_NSYS_ROOM_ROOM_NSYS_AGAIN_01_B2B` (room-nsys-again, back-to-back), `QWEN_PREFILL_NSYS_ROOM_ROOM_PLAIN_AGAIN_01_B2B` (room-plain-again, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs room-nsys |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | txt-1k-1tok | room-nsys | 1 | 10 | 165 [165–165] | 0% | 45 | 2,685 | - |
| mstar | txt-1k-1tok | room-plain | 1 | 10 | 161 [161–161] | 0% | 44 | 2,664 | 0.97x |
| mstar | txt-1k-1tok | room-nsys-again | 1 | 10 | 164 [164–164] | 0% | 47 | 2,684 | 0.99x |
| mstar | txt-1k-1tok | room-plain-again | 1 | 10 | 165 [165–165] | 0% | 46 | 2,694 | 1.00x |
