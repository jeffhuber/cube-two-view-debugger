# Vertex + Axis Human Feedback V0

Diagnostics/data-only artifact. This does not alter recognizer behavior.

This fixture upgrades the vertex-only labels into a scaffold for complete visible-trihedral labels: vertex point plus three outgoing cube-edge rays.

## Summary

- Rows: 28
- Rows with human vertex labels: 28
- Rows with all three axis endpoints: 28
- Full trihedral labels: 28
- Rows with current model axes attached: 28
- Mean current-model vertex error: 67.0 px
- Median current-model vertex error: 70.7 px

## Labeling Target

For each row, mark the visible trihedral vertex and one endpoint on each of the three outgoing cube-edge rays. Axis endpoint order does not matter; the scorer will use best assignment.

Run the labeler with:

```bash
.venv/bin/python tools/vertex_axis_label_server.py --port 8778
```

## Rows

| Row | Status | Vertex label | Axis endpoints | Current model vertex error | Notes |
|---|---|---|---:|---:|---|
| `12_B` | `trihedral_labeled` | yes | 3 | 26.9 px |  |
| `14_B` | `trihedral_labeled` | yes | 3 | 53.9 px |  |
| `15_A` | `trihedral_labeled` | yes | 3 | 63.5 px | but very close! |
| `15_B` | `trihedral_labeled` | yes | 3 | 72.0 px | but very close! |
| `17_B` | `trihedral_labeled` | yes | 3 | 107.2 px | but very close! |
| `21_A` | `trihedral_labeled` | yes | 3 | 38.7 px |  |
| `21_B` | `trihedral_labeled` | yes | 3 | 75.7 px | but very close! |
| `24_A` | `trihedral_labeled` | yes | 3 | 87.6 px | but close! |
| `26_A` | `trihedral_labeled` | yes | 3 | 72.5 px | but very close! |
| `26_B` | `trihedral_labeled` | yes | 3 | 80.9 px | but close! |
| `27_B` | `trihedral_labeled` | yes | 3 | 7.0 px |  |
| `28_A` | `trihedral_labeled` | yes | 3 | 124.3 px | but close! |
| `28_B` | `trihedral_labeled` | yes | 3 | 102.6 px | but close! |
| `29_A` | `trihedral_labeled` | yes | 3 | 57.4 px | but very close! |
| `29_B` | `trihedral_labeled` | yes | 3 | 25.4 px |  |
| `30_A` | `trihedral_labeled` | yes | 3 | 39.7 px |  |
| `30_B` | `trihedral_labeled` | yes | 3 | 10.0 px |  |
| `31_A` | `trihedral_labeled` | yes | 3 | 47.2 px | but very close! |
| `31_B` | `trihedral_labeled` | yes | 3 | 7.7 px |  |
| `32_A` | `trihedral_labeled` | yes | 3 | 72.3 px | but close! |
| `32_B` | `trihedral_labeled` | yes | 3 | 70.6 px | but close! |
| `36_B` | `trihedral_labeled` | yes | 3 | 103.0 px | but close! |
| `42_B` | `trihedral_labeled` | yes | 3 | 92.0 px | but close! |
| `44_A` | `trihedral_labeled` | yes | 3 | 185.5 px |  |
| `44_B` | `trihedral_labeled` | yes | 3 | 117.6 px |  |
| `57_A` | `trihedral_labeled` | yes | 3 | 5.0 px |  |
| `61_A` | `trihedral_labeled` | yes | 3 | 70.7 px | but close! |
| `61_B` | `trihedral_labeled` | yes | 3 | 60.0 px | but very close! |

## Interpretation

- The fixture now contains full visible-trihedral labels: human vertex plus all three outgoing cube-edge rays.
- These labels are input data for trihedral model scoring; recognition behavior remains unchanged.
