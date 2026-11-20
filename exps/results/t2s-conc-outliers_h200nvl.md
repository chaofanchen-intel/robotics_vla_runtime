# Qwen3-Omni T2S: responses outside the speech-rate band (record mode)

Band = characters of the target sentence per second of audio. `too long` = audio much longer than the sentence needs (runaway Talker), `too short` = truncated or implausibly fast.

| framework | case | B | responses | outliers | rate | too long | too short | duration range s |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mstar | verbatim-en | 1 | 40 | 0 | 0.0% | 0 | 0 | 2.1-7.9 |
| mstar | verbatim-en-b4 | 4 | 160 | 0 | 0.0% | 0 | 0 | 1.1-8.5 |
| mstar | verbatim-en-b16 | 16 | 640 | 1 | 0.2% | 1 | 0 | 1.0-20.5 |
| sglang-omni | verbatim-en | 1 | 40 | 0 | 0.0% | 0 | 0 | 2.3-8.1 |
| sglang-omni | verbatim-en-b4 | 4 | 160 | 0 | 0.0% | 0 | 0 | 1.1-8.1 |
| sglang-omni | verbatim-en-b16 | 16 | 640 | 0 | 0.0% | 0 | 0 | 1.1-8.1 |
| vllm-omni | verbatim-en | 1 | 40 | 0 | 0.0% | 0 | 0 | 1.9-7.2 |
| vllm-omni | verbatim-en-b4 | 4 | 160 | 0 | 0.0% | 0 | 0 | 1.6-7.3 |
| vllm-omni | verbatim-en-b16 | 16 | 640 | 0 | 0.0% | 0 | 0 | 0.9-9.7 |

## Outliers

| framework | case | phase | wave | position | fixture row | chars | audio s | chars/s | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| mstar | verbatim-en-b16 | precondition | 9 | 1 | seedtts_en_0255 | 50 | 20.46 | 2.4 | too long |

## Outlier rows across batch sizes (failed / served)

| framework | fixture row | B=1 | B=4 | B=16 |
|---|---|---|---|---|
| mstar | seedtts_en_0255 | 0 / 1 | 0 / 2 | 1 / 7 |

Fixture rows that failed more than once (same framework): none
