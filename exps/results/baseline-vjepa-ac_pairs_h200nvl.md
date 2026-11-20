# V-JEPA BF16 paired cohort: BASELINE_VJEPA_AC_MAIN_02_B2B

| horizon | M* HTTP p50 ms | Meta-native in-process p50 ms | Meta/M* | drift starts M* / Meta |
|---|---:|---:|---:|---:|
| H15 | 393 | 953 | 2.42x | 0/3 / 2/3 |
| H30 | 789 | 2,110 | 2.67x | 1/3 / 3/3 |

Each cell has the same source video bytes, float32 action/state trajectory, horizon and frame count. This does not prove equal decoded tensors, output quality, or equal timing boundaries. M* is measured over HTTP; Meta-native is timed in process. Native BF16 weights and internal compute are checked; the M* BF16 path is expected from engine defaults, not re-proven per cell by this client. No residual is assigned to graph + walk.
