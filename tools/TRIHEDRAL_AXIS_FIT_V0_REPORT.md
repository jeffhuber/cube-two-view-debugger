# Trihedral Axis Fit V0

Diagnostics/data-only artifact. This does not alter recognition behavior.

This scorer evaluates a full visible-trihedral fit: vertex position plus three outgoing cube-edge ray directions.

## Summary

- Rows: 28
- Full trihedral labels: 0
- Axis labels pending: 23
- Missing human vertex: 5
- Missing model trihedral: 0
- Strict-ready: 0
- Plausible: 0
- Blocked: 0
- Mean vertex error: 
- Median vertex error: 
- Mean max-axis angle error: 
- Median max-axis angle error: 

## Rows

| Row | Status | Vertex error | Max axis angle | Mean axis angle |
|---|---|---:|---:|---:|
| `12_B` | `axis_labels_pending` | 37.2 px |  |  |
| `14_B` | `axis_labels_pending` | 33.0 px |  |  |
| `15_A` | `axis_labels_pending` | 63.5 px |  |  |
| `15_B` | `axis_labels_pending` | 72.0 px |  |  |
| `17_B` | `axis_labels_pending` | 110.0 px |  |  |
| `21_A` | `axis_labels_pending` | 38.7 px |  |  |
| `21_B` | `axis_labels_pending` | 75.7 px |  |  |
| `24_A` | `axis_labels_pending` | 87.6 px |  |  |
| `26_A` | `axis_labels_pending` | 72.5 px |  |  |
| `26_B` | `axis_labels_pending` | 80.9 px |  |  |
| `27_B` | `missing_human_vertex` |  |  |  |
| `28_A` | `axis_labels_pending` | 124.3 px |  |  |
| `28_B` | `axis_labels_pending` | 102.6 px |  |  |
| `29_A` | `axis_labels_pending` | 57.4 px |  |  |
| `29_B` | `missing_human_vertex` |  |  |  |
| `30_A` | `axis_labels_pending` | 39.7 px |  |  |
| `30_B` | `missing_human_vertex` |  |  |  |
| `31_A` | `axis_labels_pending` | 47.2 px |  |  |
| `31_B` | `missing_human_vertex` |  |  |  |
| `32_A` | `axis_labels_pending` | 72.3 px |  |  |
| `32_B` | `axis_labels_pending` | 70.6 px |  |  |
| `36_B` | `axis_labels_pending` | 103.0 px |  |  |
| `42_B` | `axis_labels_pending` | 89.2 px |  |  |
| `44_A` | `axis_labels_pending` | 185.5 px |  |  |
| `44_B` | `axis_labels_pending` | 117.6 px |  |  |
| `57_A` | `missing_human_vertex` |  |  |  |
| `61_A` | `axis_labels_pending` | 70.7 px |  |  |
| `61_B` | `axis_labels_pending` | 60.0 px |  |  |

## Interpretation

- No row currently has all three human axis endpoints, so this PR cannot claim axis-fit quality yet.
- This is intentional: previous conclusions were limited by vertex-only labels. The scorer is now ready for the next human labeling pass.
