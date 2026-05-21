# Vertex + Axis Active-Learning Feedback V0

Diagnostics/data-only artifact. This does not alter recognizer behavior.

This fixture is the active-learning queue for additional visible-trihedral labels: vertex point plus three outgoing cube-edge rays.

## Summary

- Rows: 30
- Rows with human vertex labels: 0
- Rows with all three axis endpoints: 0
- Full trihedral labels: 0
- Rows with current model axes attached: 30
- Mean current-model vertex error: n/a
- Median current-model vertex error: n/a

## Labeling Target

For each row, mark the visible trihedral vertex and one endpoint on each of the three outgoing cube-edge rays. Axis endpoint order does not matter; the scorer will use best assignment.

Run the labeler with:

```bash
.venv/bin/python tools/vertex_axis_label_server.py --feedback tests/fixtures/vertex_axis_active_learning_feedback_v0.json --report tools/VERTEX_AXIS_ACTIVE_LEARNING_FEEDBACK_V0_REPORT.md --port 8778
```

## Rows

| Row | Status | Vertex label | Axis endpoints | Current model vertex error | Notes |
|---|---|---|---:|---:|---|
| `42_A` | `unlabeled` | no | 0 | n/a |  |
| `36_A` | `unlabeled` | no | 0 | n/a |  |
| `23_B` | `unlabeled` | no | 0 | n/a |  |
| `37_A` | `unlabeled` | no | 0 | n/a |  |
| `23_A` | `unlabeled` | no | 0 | n/a |  |
| `37_B` | `unlabeled` | no | 0 | n/a |  |
| `12_A` | `unlabeled` | no | 0 | n/a |  |
| `27_A` | `unlabeled` | no | 0 | n/a |  |
| `24_B` | `unlabeled` | no | 0 | n/a |  |
| `14_A` | `unlabeled` | no | 0 | n/a |  |
| `17_A` | `unlabeled` | no | 0 | n/a |  |
| `57_B` | `unlabeled` | no | 0 | n/a |  |
| `62_A` | `unlabeled` | no | 0 | n/a |  |
| `62_B` | `unlabeled` | no | 0 | n/a |  |
| `22_B` | `unlabeled` | no | 0 | n/a |  |
| `58_A` | `unlabeled` | no | 0 | n/a |  |
| `58_B` | `unlabeled` | no | 0 | n/a |  |
| `39_A` | `unlabeled` | no | 0 | n/a |  |
| `47_A` | `unlabeled` | no | 0 | n/a |  |
| `47_B` | `unlabeled` | no | 0 | n/a |  |
| `49_A` | `unlabeled` | no | 0 | n/a |  |
| `22_A` | `unlabeled` | no | 0 | n/a |  |
| `49_B` | `unlabeled` | no | 0 | n/a |  |
| `48_B` | `unlabeled` | no | 0 | n/a |  |
| `39_B` | `unlabeled` | no | 0 | n/a |  |
| `46_B` | `unlabeled` | no | 0 | n/a |  |
| `25_B` | `unlabeled` | no | 0 | n/a |  |
| `48_A` | `unlabeled` | no | 0 | n/a |  |
| `46_A` | `unlabeled` | no | 0 | n/a |  |
| `25_A` | `unlabeled` | no | 0 | n/a |  |

## Interpretation

- The active-learning scaffold is ready for human vertex+axis labels; committed rows remain unlabeled until the labeler writes them.
- The next scorer can evaluate current model axes as soon as full trihedral labels exist; until then, axis-quality conclusions should stay pending.
