# Qwen3-Omni T2S: responses outside the speech-rate band (record mode)

Band = characters of the target sentence per second of audio. `too long` = audio much longer than the sentence needs (runaway Talker), `too short` = truncated or implausibly fast.

| framework | case | B | responses | outliers | rate | too long | too short | duration range s |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mstar | verbatim-en | 1 | 40 | 0 | 0.0% | 0 | 0 | 2.1-7.7 |
| mstar | verbatim-en-b4 | 4 | 160 | 0 | 0.0% | 0 | 0 | 1.3-7.2 |
| mstar | verbatim-en-b16 | 16 | 640 | 0 | 0.0% | 0 | 0 | 1.1-7.5 |

No response was outside the band.
