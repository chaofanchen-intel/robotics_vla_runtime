# Qwen3-Omni T2S: responses outside the speech-rate band (record mode)

Band = characters of the target sentence per second of audio. `too long` = audio much longer than the sentence needs (runaway Talker), `too short` = truncated or implausibly fast.

| framework | case | B | responses | outliers | rate | too long | too short | duration range s |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mstar | verbatim-en-b16 | 16 | 720 | 30 | 4.2% | 30 | 0 | 1.1-20.5 |

## Outliers

| framework | case | phase | wave | position | fixture row | chars | audio s | chars/s | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| mstar | verbatim-en-b16 | precondition | 3 | 2 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | precondition | 5 | 4 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 1 | 5 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 2 | 6 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 4 | 8 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 5 | 9 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 6 | 10 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 7 | 11 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 8 | 12 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 9 | 13 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 10 | 14 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 11 | 15 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 14 | 2 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 17 | 5 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 18 | 6 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 19 | 7 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 20 | 8 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 21 | 9 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 22 | 10 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 24 | 12 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 25 | 13 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 26 | 14 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 27 | 15 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 33 | 5 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 34 | 6 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 36 | 8 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 37 | 9 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 38 | 10 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 39 | 11 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 40 | 12 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |

## Outlier rows across batch sizes (failed / served)

| framework | fixture row | B=16 |
|---|---|---|
| mstar | seedtts_en_0590 | 30 / 45 |

Fixture rows that failed more than once (same framework): mstar row seedtts_en_0590 x30
