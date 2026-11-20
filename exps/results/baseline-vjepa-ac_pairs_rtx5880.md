# V-JEPA BF16 paired cohort: BASELINE_VJEPA_AC_MAIN_01_B2B

| horizon | M* HTTP p50 ms | Meta-native in-process p50 ms | Meta/M* | drift starts M* / Meta |
|---|---:|---:|---:|---:|
| H15 | 394 | 1,132 | 2.88x | 0/3 / 1/3 |
| H30 | 662 | 2,826 | 4.27x | 1/3 / 0/3 |

Each cell has the same source video bytes, float32 action/state trajectory, horizon and frame count. This does not prove equal decoded tensors, output quality, or equal timing boundaries. M* is measured over HTTP; Meta-native is timed in process. Native BF16 weights and internal compute are checked; the M* BF16 path is expected from engine defaults, not re-proven per cell by this client. No residual is assigned to graph + walk.
