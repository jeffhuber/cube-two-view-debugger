# Geometry-First Face Split V0

Diagnostics/data-only artifact. This does not alter recognition behavior.

This report evaluates the face-splitting step that should replace semantic `top_face`/`left_face`/`right_face` masks: use one fitted cube model to generate the three visible face quads and their 3x3 cell grids.

## Summary

- Compared rows: 23
- Skipped rows: 5
- Strict vertex threshold: 30 px
- Plausible vertex threshold: 50 px

| Source | Rows | Nondegenerate splits | Strict-ready | Plausible | Vertex-blocked | Mean vertex error |
|---|---:|---:|---:|---:|---:|---:|
| `rembg` | 23 | 23 | 0 | 5 | 18 | 78.8 px |
| `sam3` | 23 | 23 | 6 | 9 | 14 | 72.2 px |
| `oracle_best_source` | 23 | 23 | 6 | 11 | 12 | 58.7 px |

## Rows

| Row | Best source | rembg status/error | SAM3 status/error |
|---|---|---|---|
| `12_B` | `sam3` | `split_plausible` / 37 px | `split_ready_strict` / 26 px |
| `14_B` | `sam3` | `split_plausible` / 33 px | `split_ready_strict` / 16 px |
| `15_A` | `sam3` | `vertex_error_blocked` / 64 px | `split_plausible` / 47 px |
| `15_B` | `sam3` | `vertex_error_blocked` / 72 px | `vertex_error_blocked` / 54 px |
| `17_B` | `rembg` | `vertex_error_blocked` / 110 px | `vertex_error_blocked` / 137 px |
| `21_A` | `rembg` | `split_plausible` / 39 px | `vertex_error_blocked` / 56 px |
| `21_B` | `rembg` | `vertex_error_blocked` / 76 px | `vertex_error_blocked` / 118 px |
| `24_A` | `sam3` | `vertex_error_blocked` / 88 px | `vertex_error_blocked` / 86 px |
| `26_A` | `rembg` | `vertex_error_blocked` / 73 px | `vertex_error_blocked` / 121 px |
| `26_B` | `sam3` | `vertex_error_blocked` / 81 px | `split_ready_strict` / 24 px |
| `28_A` | `sam3` | `vertex_error_blocked` / 124 px | `vertex_error_blocked` / 78 px |
| `28_B` | `rembg` | `vertex_error_blocked` / 103 px | `vertex_error_blocked` / 106 px |
| `29_A` | `sam3` | `vertex_error_blocked` / 57 px | `split_plausible` / 45 px |
| `30_A` | `sam3` | `split_plausible` / 40 px | `split_ready_strict` / 20 px |
| `31_A` | `rembg` | `split_plausible` / 47 px | `vertex_error_blocked` / 92 px |
| `32_A` | `sam3` | `vertex_error_blocked` / 72 px | `vertex_error_blocked` / 56 px |
| `32_B` | `sam3` | `vertex_error_blocked` / 71 px | `split_plausible` / 48 px |
| `36_B` | `sam3` | `vertex_error_blocked` / 103 px | `vertex_error_blocked` / 70 px |
| `42_B` | `rembg` | `vertex_error_blocked` / 89 px | `vertex_error_blocked` / 186 px |
| `44_A` | `sam3` | `vertex_error_blocked` / 186 px | `vertex_error_blocked` / 148 px |
| `44_B` | `sam3` | `vertex_error_blocked` / 118 px | `split_ready_strict` / 15 px |
| `61_A` | `sam3` | `vertex_error_blocked` / 71 px | `split_ready_strict` / 19 px |
| `61_B` | `rembg` | `vertex_error_blocked` / 60 px | `vertex_error_blocked` / 89 px |

## Interpretation

- Geometry-first splitting is mechanically viable: the fitted models usually produce nondegenerate face quads and cells.
- The blocker is still upstream vertex/axis selection. When the vertex is off by more than roughly 50 px, the face split is not trustworthy even if the quads are well-formed.
- SAM3 whole-cube masks improve the strict-ready count versus rembg in this local bakeoff, but the oracle across the two sources is still far below a production threshold.
- Next work should improve source selection / vertex confidence before any rectified color read path uses these generated face grids.
