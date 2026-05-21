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
- Mean vertex error: n/a
- Median vertex error: n/a
- Mean max-axis angle error: n/a
- Median max-axis angle error: n/a

## Rows

| Row | Status | Vertex error | Max axis angle | Mean axis angle |
|---|---|---:|---:|---:|
| `12_B` | `axis_labels_pending` | 37.2 px | n/a | n/a |
| `14_B` | `axis_labels_pending` | 33.0 px | n/a | n/a |
| `15_A` | `axis_labels_pending` | 63.5 px | n/a | n/a |
| `15_B` | `axis_labels_pending` | 72.0 px | n/a | n/a |
| `17_B` | `axis_labels_pending` | 110.0 px | n/a | n/a |
| `21_A` | `axis_labels_pending` | 38.7 px | n/a | n/a |
| `21_B` | `axis_labels_pending` | 75.7 px | n/a | n/a |
| `24_A` | `axis_labels_pending` | 87.6 px | n/a | n/a |
| `26_A` | `axis_labels_pending` | 72.5 px | n/a | n/a |
| `26_B` | `axis_labels_pending` | 80.9 px | n/a | n/a |
| `27_B` | `missing_human_vertex` | n/a | n/a | n/a |
| `28_A` | `axis_labels_pending` | 124.3 px | n/a | n/a |
| `28_B` | `axis_labels_pending` | 102.6 px | n/a | n/a |
| `29_A` | `axis_labels_pending` | 57.4 px | n/a | n/a |
| `29_B` | `missing_human_vertex` | n/a | n/a | n/a |
| `30_A` | `axis_labels_pending` | 39.7 px | n/a | n/a |
| `30_B` | `missing_human_vertex` | n/a | n/a | n/a |
| `31_A` | `axis_labels_pending` | 47.2 px | n/a | n/a |
| `31_B` | `missing_human_vertex` | n/a | n/a | n/a |
| `32_A` | `axis_labels_pending` | 72.3 px | n/a | n/a |
| `32_B` | `axis_labels_pending` | 70.6 px | n/a | n/a |
| `36_B` | `axis_labels_pending` | 103.0 px | n/a | n/a |
| `42_B` | `axis_labels_pending` | 89.2 px | n/a | n/a |
| `44_A` | `axis_labels_pending` | 185.5 px | n/a | n/a |
| `44_B` | `axis_labels_pending` | 117.6 px | n/a | n/a |
| `57_A` | `missing_human_vertex` | n/a | n/a | n/a |
| `61_A` | `axis_labels_pending` | 70.7 px | n/a | n/a |
| `61_B` | `axis_labels_pending` | 60.0 px | n/a | n/a |

## Interpretation

- No row currently has all three human axis endpoints, so this PR cannot claim axis-fit quality yet.
- This is intentional: previous conclusions were limited by vertex-only labels. The scorer is now ready for the next human labeling pass.
