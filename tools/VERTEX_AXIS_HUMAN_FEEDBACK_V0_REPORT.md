# Vertex + Axis Human Feedback V0

Diagnostics/data-only artifact. This does not alter recognizer behavior.

This fixture upgrades the vertex-only labels into a scaffold for complete visible-trihedral labels: vertex point plus three outgoing cube-edge rays.

## Summary

- Rows: 28
- Rows with human vertex labels: 23
- Rows with all three axis endpoints: 0
- Full trihedral labels: 0
- Rows with current model axes attached: 28
- Mean current-model vertex error: 78.8 px
- Median current-model vertex error: 72.3 px

## Labeling Target

For each row, mark the visible trihedral vertex and one endpoint on each of the three outgoing cube-edge rays. Axis endpoint order does not matter; the scorer will use best assignment.

Run the labeler with:

```bash
.venv/bin/python tools/vertex_axis_label_server.py --port 8778
```

## Rows

| Row | Status | Vertex label | Axis endpoints | Current model vertex error | Notes |
|---|---|---|---:|---:|---|
| `12_B` | `vertex_labeled_axes_pending` | yes | 0 | 37.2 px |  |
| `14_B` | `vertex_labeled_axes_pending` | yes | 0 | 33.0 px |  |
| `15_A` | `vertex_labeled_axes_pending` | yes | 0 | 63.5 px | but very close! |
| `15_B` | `vertex_labeled_axes_pending` | yes | 0 | 72.0 px | but very close! |
| `17_B` | `vertex_labeled_axes_pending` | yes | 0 | 110.0 px | but very close! |
| `21_A` | `vertex_labeled_axes_pending` | yes | 0 | 38.7 px |  |
| `21_B` | `vertex_labeled_axes_pending` | yes | 0 | 75.7 px | but very close! |
| `24_A` | `vertex_labeled_axes_pending` | yes | 0 | 87.6 px | but close! |
| `26_A` | `vertex_labeled_axes_pending` | yes | 0 | 72.5 px | but very close! |
| `26_B` | `vertex_labeled_axes_pending` | yes | 0 | 80.9 px | but close! |
| `27_B` | `judgment_only` | no | 0 |  |  |
| `28_A` | `vertex_labeled_axes_pending` | yes | 0 | 124.3 px | but close! |
| `28_B` | `vertex_labeled_axes_pending` | yes | 0 | 102.6 px | but close! |
| `29_A` | `vertex_labeled_axes_pending` | yes | 0 | 57.4 px | but very close! |
| `29_B` | `judgment_only` | no | 0 |  |  |
| `30_A` | `vertex_labeled_axes_pending` | yes | 0 | 39.7 px |  |
| `30_B` | `judgment_only` | no | 0 |  |  |
| `31_A` | `vertex_labeled_axes_pending` | yes | 0 | 47.2 px | but very close! |
| `31_B` | `judgment_only` | no | 0 |  |  |
| `32_A` | `vertex_labeled_axes_pending` | yes | 0 | 72.3 px | but close! |
| `32_B` | `vertex_labeled_axes_pending` | yes | 0 | 70.6 px | but close! |
| `36_B` | `vertex_labeled_axes_pending` | yes | 0 | 103.0 px | but close! |
| `42_B` | `vertex_labeled_axes_pending` | yes | 0 | 89.2 px | but close! |
| `44_A` | `vertex_labeled_axes_pending` | yes | 0 | 185.5 px |  |
| `44_B` | `vertex_labeled_axes_pending` | yes | 0 | 117.6 px |  |
| `57_A` | `judgment_only` | no | 0 |  |  |
| `61_A` | `vertex_labeled_axes_pending` | yes | 0 | 70.7 px | but close! |
| `61_B` | `vertex_labeled_axes_pending` | yes | 0 | 60.0 px | but very close! |

## Interpretation

- The scaffold is ready for human axis labels, but the committed durable labels are still vertex-only.
- The next scorer can evaluate current model axes as soon as full trihedral labels exist; until then, axis-quality conclusions should stay pending.
