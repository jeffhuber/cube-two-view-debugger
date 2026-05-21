# Trihedral Axis Fit V0

Diagnostics/data-only artifact. This does not alter recognition behavior.

This scorer evaluates a full visible-trihedral fit: vertex position plus three outgoing cube-edge ray directions.

## Summary

- Rows: 28
- Full trihedral labels: 28
- Axis labels pending: 0
- Missing human vertex: 0
- Missing model trihedral: 0
- Strict-ready: 4
- Plausible: 8
- Blocked: 20
- Mean vertex error: 67.0 px
- Median vertex error: 70.7 px
- Mean max-axis angle error: 27.0 deg
- Median max-axis angle error: 5.8 deg
- Failure categories: `axis_correspondence_blocked` 1, `both_blocked` 10, `plausible` 4, `strict_ready` 4, `vertex_localization_blocked` 9

## Rows

| Row | Status | Failure category | Vertex error | Max axis angle | Mean axis angle |
|---|---|---|---:|---:|---:|
| `12_B` | `blocked` | `axis_correspondence_blocked` | 26.9 px | 60.8 deg | 60.0 deg |
| `14_B` | `blocked` | `both_blocked` | 53.9 px | 62.5 deg | 61.0 deg |
| `15_A` | `blocked` | `vertex_localization_blocked` | 63.5 px | 6.5 deg | 2.8 deg |
| `15_B` | `blocked` | `vertex_localization_blocked` | 72.0 px | 4.1 deg | 3.1 deg |
| `17_B` | `blocked` | `vertex_localization_blocked` | 107.2 px | 5.2 deg | 1.9 deg |
| `21_A` | `plausible` | `plausible` | 38.7 px | 5.1 deg | 2.5 deg |
| `21_B` | `blocked` | `vertex_localization_blocked` | 75.7 px | 5.1 deg | 4.5 deg |
| `24_A` | `blocked` | `both_blocked` | 87.6 px | 61.7 deg | 59.4 deg |
| `26_A` | `blocked` | `vertex_localization_blocked` | 72.5 px | 3.0 deg | 1.8 deg |
| `26_B` | `blocked` | `both_blocked` | 80.9 px | 64.0 deg | 59.6 deg |
| `27_B` | `strict_ready` | `strict_ready` | 7.0 px | 3.4 deg | 1.4 deg |
| `28_A` | `blocked` | `vertex_localization_blocked` | 124.3 px | 4.9 deg | 2.8 deg |
| `28_B` | `blocked` | `vertex_localization_blocked` | 102.6 px | 2.9 deg | 2.1 deg |
| `29_A` | `blocked` | `both_blocked` | 57.4 px | 60.6 deg | 59.0 deg |
| `29_B` | `strict_ready` | `strict_ready` | 25.4 px | 1.5 deg | 0.7 deg |
| `30_A` | `plausible` | `plausible` | 39.7 px | 7.5 deg | 3.3 deg |
| `30_B` | `strict_ready` | `strict_ready` | 10.0 px | 3.9 deg | 3.3 deg |
| `31_A` | `plausible` | `plausible` | 47.2 px | 2.3 deg | 1.0 deg |
| `31_B` | `strict_ready` | `strict_ready` | 7.7 px | 2.7 deg | 1.6 deg |
| `32_A` | `blocked` | `both_blocked` | 72.3 px | 59.9 deg | 58.7 deg |
| `32_B` | `blocked` | `both_blocked` | 70.6 px | 63.0 deg | 60.4 deg |
| `36_B` | `blocked` | `vertex_localization_blocked` | 103.0 px | 1.8 deg | 1.0 deg |
| `42_B` | `blocked` | `both_blocked` | 92.0 px | 64.1 deg | 59.1 deg |
| `44_A` | `blocked` | `both_blocked` | 185.5 px | 62.5 deg | 59.4 deg |
| `44_B` | `blocked` | `both_blocked` | 117.6 px | 62.2 deg | 59.3 deg |
| `57_A` | `plausible` | `plausible` | 5.0 px | 8.6 deg | 4.7 deg |
| `61_A` | `blocked` | `both_blocked` | 70.7 px | 61.3 deg | 59.3 deg |
| `61_B` | `blocked` | `vertex_localization_blocked` | 60.0 px | 4.1 deg | 2.2 deg |

## Interpretation

- Rows are `strict_ready` only when both the vertex and all three axis directions are within the configured thresholds.
- Axis assignment is order-invariant, so high errors point at geometry, not label ordering.
- `vertex_localization_blocked` means axis directions are plausible but the visible trihedral vertex is too far from the human label.
- `axis_correspondence_blocked` means the vertex is plausible but the outgoing axis family is wrong.
- `both_blocked` means neither the vertex nor the axis family is currently usable.
