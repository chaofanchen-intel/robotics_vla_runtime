# Qwen3-Omni T2S: responses outside the speech-rate band (record mode)

Band = characters of the target sentence per second of audio. `too long` = audio much longer than the sentence needs (runaway Talker), `too short` = truncated or implausibly fast.

| framework | case | B | responses | outliers | rate | too long | too short | duration range s |
|---|---|---:|---:|---:|---:|---:|---:|---|
| mstar | verbatim-en-b4 | 4 | 180 | 0 | 0.0% | 0 | 0 | 1.2-6.9 |
| mstar | verbatim-en-b8 | 8 | 360 | 1 | 0.3% | 1 | 0 | 1.1-20.5 |
| mstar | verbatim-en-b12 | 12 | 540 | 7 | 1.3% | 7 | 0 | 1.2-20.5 |
| mstar | verbatim-en-b16 | 16 | 720 | 31 | 4.3% | 31 | 0 | 1.2-20.5 |

## Outliers

| framework | case | phase | wave | position | fixture row | chars | audio s | chars/s | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| mstar | verbatim-en-b8 | measured | 6 | 3 | seedtts_en_0255 | 50 | 20.46 | 2.4 | too long |
| mstar | verbatim-en-b12 | measured | 11 | 3 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b12 | measured | 18 | 10 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b12 | measured | 23 | 3 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b12 | measured | 25 | 5 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b12 | measured | 26 | 6 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b12 | measured | 28 | 8 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b12 | measured | 31 | 2 | seedtts_en_0139 | 60 | 20.46 | 2.9 | too long |
| mstar | verbatim-en-b16 | precondition | 3 | 2 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 1 | 5 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 2 | 6 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 3 | 7 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 4 | 8 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 5 | 9 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 6 | 10 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 7 | 11 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 8 | 12 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 9 | 13 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 10 | 14 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 11 | 15 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 13 | 1 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 16 | 4 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 17 | 5 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 18 | 6 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 19 | 6 | seedtts_en_0318 | 52 | 20.46 | 2.5 | too long |
| mstar | verbatim-en-b16 | measured | 19 | 11 | seedtts_en_0990 | 59 | 20.46 | 2.9 | too long |
| mstar | verbatim-en-b16 | measured | 20 | 8 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 21 | 9 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 22 | 10 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 23 | 11 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 26 | 14 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 27 | 15 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 28 | 0 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 33 | 5 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 34 | 6 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 36 | 8 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 38 | 10 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 39 | 11 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |
| mstar | verbatim-en-b16 | measured | 40 | 12 | seedtts_en_0590 | 64 | 20.46 | 3.1 | too long |

## Outlier rows across batch sizes (failed / served)

| framework | fixture row | B=4 | B=8 | B=12 | B=16 |
|---|---|---|---|---|---|
| mstar | seedtts_en_0139 | 0 / 1 | 0 / 3 | 1 / 5 | 0 / 6 |
| mstar | seedtts_en_0255 | 0 / 1 | 1 / 3 | 0 / 5 | 0 / 7 |
| mstar | seedtts_en_0318 | 0 / 1 | 0 / 3 | 0 / 5 | 1 / 7 |
| mstar | seedtts_en_0590 | 0 / 45 | 0 / 45 | 6 / 45 | 29 / 45 |
| mstar | seedtts_en_0990 | 0 / 1 | 0 / 3 | 0 / 5 | 1 / 7 |

Fixture rows that failed more than once (same framework): mstar row seedtts_en_0590 x35
