# Qwen3-Omni T2S: responses outside the speech-rate band (record mode)

Band = characters of the target sentence per second of audio. `too long` = audio much longer than the sentence needs (runaway Talker), `too short` = truncated or implausibly fast.

| framework | case | B | responses | outliers | rate | too long | too short | duration range s |
|---|---|---:|---:|---:|---:|---:|---:|---|
| sglang-omni | verbatim-en-b16 | 16 | 720 | 18 | 2.5% | 18 | 0 | 3.4-20.5 |

## Outliers

| framework | case | phase | wave | position | fixture row | chars | audio s | chars/s | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| sglang-omni | verbatim-en-b16 | measured | 17 | 9 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 17 | 10 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 17 | 11 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 17 | 12 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 17 | 13 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 17 | 14 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 17 | 15 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 34 | 5 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 34 | 6 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 34 | 7 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 34 | 8 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 34 | 9 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 34 | 10 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 34 | 11 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 34 | 12 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 34 | 13 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 34 | 14 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 34 | 15 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |

## Outlier rows across batch sizes (failed / served)

| framework | fixture row | B=16 |
|---|---|---|
| sglang-omni | seedtts_en_0139 | 18 / 720 |

Fixture rows that failed more than once (same framework): sglang-omni row seedtts_en_0139 x18
