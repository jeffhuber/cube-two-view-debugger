# Vertex + Axis Active-Learning Feedback V0

Diagnostics/data-only artifact. This does not alter recognizer behavior.

This fixture is the active-learning queue for additional visible-trihedral labels: vertex point plus three outgoing cube-edge rays.

## Summary

- Rows: 30
- Rows with human vertex labels: 30
- Rows with all three axis endpoints: 30
- Full trihedral labels: 30
- Rows with current model axes attached: 30
- Mean current-model vertex error: 308.0 px
- Median current-model vertex error: 214.5 px

## Labeling Target

For each row, mark the visible trihedral vertex and one endpoint on each of the three outgoing cube-edge rays. Axis endpoint order does not matter; the scorer will use best assignment.

Run the labeler with:

```bash
.venv/bin/python tools/vertex_axis_label_server.py --feedback tests/fixtures/vertex_axis_active_learning_feedback_v0.json --report tools/VERTEX_AXIS_ACTIVE_LEARNING_FEEDBACK_V0_REPORT.md --port 8778
```

## Rows

| Row | Status | Vertex label | Axis endpoints | Current model vertex error | Notes |
|---|---|---|---:|---:|---|
| `42_A` | `trihedral_labeled` | yes | 3 | 301.9 px |  |
| `36_A` | `trihedral_labeled` | yes | 3 | 1018.0 px |  |
| `23_B` | `trihedral_labeled` | yes | 3 | 120.4 px |  |
| `37_A` | `trihedral_labeled` | yes | 3 | 266.5 px |  |
| `23_A` | `trihedral_labeled` | yes | 3 | 99.6 px |  |
| `37_B` | `trihedral_labeled` | yes | 3 | 372.2 px |  |
| `12_A` | `trihedral_labeled` | yes | 3 | 115.9 px |  |
| `27_A` | `trihedral_labeled` | yes | 3 | 190.8 px |  |
| `24_B` | `trihedral_labeled` | yes | 3 | 142.7 px |  |
| `14_A` | `trihedral_labeled` | yes | 3 | 435.3 px |  |
| `17_A` | `trihedral_labeled` | yes | 3 | 118.2 px |  |
| `57_B` | `trihedral_labeled` | yes | 3 | 829.8 px |  |
| `62_A` | `trihedral_labeled` | yes | 3 | 120.0 px |  |
| `62_B` | `trihedral_labeled` | yes | 3 | 181.6 px |  |
| `22_B` | `trihedral_labeled` | yes | 3 | 73.3 px |  |
| `58_A` | `trihedral_labeled` | yes | 3 | 86.1 px |  |
| `58_B` | `trihedral_labeled` | yes | 3 | 33.8 px |  |
| `39_A` | `trihedral_labeled` | yes | 3 | 184.5 px |  |
| `47_A` | `trihedral_labeled` | yes | 3 | 263.4 px |  |
| `47_B` | `trihedral_labeled` | yes | 3 | 323.2 px |  |
| `49_A` | `trihedral_labeled` | yes | 3 | 390.1 px |  |
| `22_A` | `trihedral_labeled` | yes | 3 | 167.4 px |  |
| `49_B` | `trihedral_labeled` | yes | 3 | 183.6 px |  |
| `48_B` | `trihedral_labeled` | yes | 3 | 156.2 px |  |
| `39_B` | `trihedral_labeled` | yes | 3 | 238.3 px |  |
| `46_B` | `trihedral_labeled` | yes | 3 | 573.8 px |  |
| `25_B` | `trihedral_labeled` | yes | 3 | 362.4 px |  |
| `48_A` | `trihedral_labeled` | yes | 3 | 369.4 px |  |
| `46_A` | `trihedral_labeled` | yes | 3 | 818.1 px |  |
| `25_A` | `trihedral_labeled` | yes | 3 | 702.4 px |  |

## Interpretation

- The fixture now contains full visible-trihedral labels: human vertex plus all three outgoing cube-edge rays.
- These labels are input data for trihedral model scoring; recognition behavior remains unchanged.
