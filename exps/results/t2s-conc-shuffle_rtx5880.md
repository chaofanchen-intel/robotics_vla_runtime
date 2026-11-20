# Qwen3-Omni T2S: responses outside the speech-rate band (record mode)

Band = characters of the target sentence per second of audio. `too long` = audio much longer than the sentence needs (runaway Talker), `too short` = truncated or implausibly fast.

| framework | case | B | responses | outliers | rate | too long | too short | duration range s |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mstar | verbatim-en | 1 | 110 | 0 | 0.0% | 0 | 0 | 1.1-7.5 |
| mstar | verbatim-en-b4 | 4 | 440 | 0 | 0.0% | 0 | 0 | 1.0-8.5 |
| mstar | verbatim-en-b16 | 16 | 1760 | 13 | 0.7% | 13 | 0 | 1.1-20.5 |
| sglang-omni | verbatim-en | 1 | 110 | 0 | 0.0% | 0 | 0 | 1.1-7.3 |
| sglang-omni | verbatim-en-b4 | 4 | 440 | 0 | 0.0% | 0 | 0 | 1.0-7.3 |
| sglang-omni | verbatim-en-b16 | 16 | 1760 | 3 | 0.2% | 3 | 0 | 1.0-20.5 |
| vllm-omni | verbatim-en | 1 | 110 | 0 | 0.0% | 0 | 0 | 1.5-7.1 |
| vllm-omni | verbatim-en-b4 | 4 | 440 | 0 | 0.0% | 0 | 0 | 0.9-7.7 |
| vllm-omni | verbatim-en-b16 | 16 | 1760 | 0 | 0.0% | 0 | 0 | 0.8-8.5 |

## Outliers

| framework | case | phase | wave | position | fixture row | chars | audio s | chars/s | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| mstar | verbatim-en-b16 | precondition | 3 | 12 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 3 | 0 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 8 | 5 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 21 | 15 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 27 | 8 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 29 | 13 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 44 | 3 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 48 | 8 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 58 | 7 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 66 | 12 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 72 | 8 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 88 | 14 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 91 | 5 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| sglang-omni | verbatim-en-b16 | precondition | 6 | 11 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 42 | 11 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |
| sglang-omni | verbatim-en-b16 | measured | 86 | 9 | seedtts_en_0139 | 60 | 20.48 | 2.9 | too long |

## Outlier rows across batch sizes (failed / served)

| framework | fixture row | B=1 | B=4 | B=16 |
|---|---|---|---|---|
| mstar | seedtts_en_0590 | 0 / 1 | 0 / 4 | 13 / 17 |
| sglang-omni | seedtts_en_0139 | 0 / 1 | 0 / 4 | 3 / 18 |

Fixture rows that failed more than once (same framework): mstar row seedtts_en_0590 x13, sglang-omni row seedtts_en_0139 x3
