# Qwen3-Omni T2S: responses outside the speech-rate band (record mode)

Band = characters of the target sentence per second of audio. `too long` = audio much longer than the sentence needs (runaway Talker), `too short` = truncated or implausibly fast.

| framework | case | B | responses | outliers | rate | too long | too short | duration range s |
|---|---|---:|---:|---:|---:|---:|---:|---|
| sglang-omni | verbatim-en-b4 | 4 | 180 | 0 | 0.0% | 0 | 0 | 1.0-6.8 |
| sglang-omni | verbatim-en-b8 | 8 | 360 | 0 | 0.0% | 0 | 0 | 1.2-7.3 |
| sglang-omni | verbatim-en-b12 | 12 | 540 | 1 | 0.2% | 1 | 0 | 1.2-20.5 |
| sglang-omni | verbatim-en-b16 | 16 | 720 | 7 | 1.0% | 7 | 0 | 1.0-20.5 |

## Outliers

| framework | case | phase | wave | position | fixture row | chars | audio s | chars/s | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| sglang-omni | verbatim-en-b12 | measured | 12 | 5 | seedtts_en_0649 | 87 | 20.48 | 4.2 | too long |
| sglang-omni | verbatim-en-b16 | precondition | 2 | 1 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 5 | 9 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 11 | 15 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 15 | 3 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 19 | 7 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 21 | 9 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 39 | 11 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |

## Outlier rows across batch sizes (failed / served)

| framework | fixture row | B=4 | B=8 | B=12 | B=16 |
|---|---|---|---|---|---|
| sglang-omni | seedtts_en_0139 | 0 / 45 | 0 / 45 | 0 / 45 | 7 / 45 |
| sglang-omni | seedtts_en_0649 | 0 / 1 | 0 / 3 | 1 / 5 | 0 / 6 |

Fixture rows that failed more than once (same framework): sglang-omni row seedtts_en_0139 x7
