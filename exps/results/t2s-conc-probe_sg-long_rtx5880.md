# Qwen3-Omni T2S: responses outside the speech-rate band (record mode)

Band = characters of the target sentence per second of audio. `too long` = audio much longer than the sentence needs (runaway Talker), `too short` = truncated or implausibly fast.

| framework | case | B | responses | outliers | rate | too long | too short | duration range s |
|---|---|---:|---:|---:|---:|---:|---:|---|
| sglang-omni | verbatim-en-b16 | 16 | 720 | 10 | 1.4% | 10 | 0 | 1.0-20.5 |

## Outliers

| framework | case | phase | wave | position | fixture row | chars | audio s | chars/s | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| sglang-omni | verbatim-en-b16 | measured | 7 | 11 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 19 | 7 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 20 | 8 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 22 | 10 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 23 | 11 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 27 | 3 | seedtts_en_0542 | 51 | 20.48 | 2.5 | too long |
| sglang-omni | verbatim-en-b16 | measured | 30 | 2 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 32 | 4 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 38 | 10 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 39 | 11 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |

## Outlier rows across batch sizes (failed / served)

| framework | fixture row | B=16 |
|---|---|---|
| sglang-omni | seedtts_en_0139 | 9 / 45 |
| sglang-omni | seedtts_en_0542 | 1 / 7 |

Fixture rows that failed more than once (same framework): sglang-omni row seedtts_en_0139 x9
