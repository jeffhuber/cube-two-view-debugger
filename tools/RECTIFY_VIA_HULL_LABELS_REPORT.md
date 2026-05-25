# Hull-labels rectification — empirical comparison

End-to-end first-principles rectification: 6 silhouette extrema labeled by image position → vertex via parallelogram completion → 3 face_quads → 3 flat rectified faces. Diagnostic-only candidate replacement for the production `tools/global_cube_model.py` pipeline.

## Origin

PR #271's draft hypothesis-tiebreaker investigation closed (informational dead end after 3 attempts). During the post-mortem walkthrough of 20_A's broken pipeline output, the user asked from first principles: given that rembg + convex hull produces 6 silhouette extrema with stable physical meaning, and given that the per-side corner-labeling convention is fixed in `FACE_DEFS_BY_SIDE`, why does production run a 720-perm Procrustes search + PnP + chirality detector + vertex ensemble + image-based vertex refinement, when the entire mapping is deterministic?

This tool answers that question. The empirical result on the 12 approved full-corner ground-truth rows:

| | Production (PR #268 baseline) | Hull-labels (this tool) |
|---|---:|---:|
| Rows with usable axis fit (≤30° misfit) | 3 / 12 (25%) | 12 / 12 (100%) |
| Rows with broken axis fit (≥150° misfit) | 9 / 12 (75%) | 0 / 12 |
| Implementation size | ~800 LOC across `global_cube_model.py` + helpers | ~50 LOC of geometry |
| Pipeline components | rembg + bezel detection + 720-perm Procrustes + PnP + mean-of-3 ensemble + phase-check + image-based vertex refinement | rembg + label-by-position + parallelogram-completion |

## Per-row results

Score = sum of CIELAB distance from each of 27 sampled stickers (9 per face × 3 faces) to its nearest canonical cube color. Lower = better cluster (each face's stickers cleanly match canonical cube colors).

| Row | Labeling mean corner err | Derived vertex err | Hull-labels score | Oracle score | Δ vs oracle |
|---|---:|---:|---:|---:|---:|
| 20_A | 17.2 | 49.5 | 656.7 | 637.9 | +18.8 |
| 20_B | 19.3 | 60.1 | 631.5 | 587.5 | +44.0 |
| 38_A | 14.1 | 43.4 | 583.9 | 582.0 | +1.9 |
| 38_B | 14.4 | 27.6 | 496.6 | 497.5 | **−0.9** |
| 40_A | 15.4 | 29.9 | 469.6 | 448.9 | +20.7 |
| 40_B | 16.4 | 36.0 | 312.0 | 316.9 | **−4.9** |
| 41_A | 17.5 | 34.1 | 461.8 | 459.3 | +2.5 |
| 41_B | 13.9 | 21.9 | 380.8 | 378.2 | +2.6 |
| 43_A | 14.8 | 20.3 | 441.3 | 441.2 | +0.1 |
| 43_B | 14.0 | 41.1 | 439.2 | 436.1 | +3.1 |
| 45_A | 14.2 | 52.7 | 529.1 | 521.7 | +7.4 |
| 45_B | 21.7 | 67.4 | 596.1 | 590.9 | +5.2 |

Summary stats:

- **Score delta vs oracle:** min −4.9, max +43.9, median +2.8. 2/12 rows beat oracle slightly (derived vertex landed closer to the visual trihedral junction than the human-labeled GT vertex). 9/12 rows within 25 points of oracle; 12/12 within 50 points.
- **Derived vertex error:** min 20.3, max 67.4, median 38.5 px. Better than production's bezel-detected vertex on the same rows (which had 43-241 px errors per PR #274's stage-transition trace).
- **Labeling mean corner error:** min 13.9, max 21.7, median 15.2 px. Per-corner accuracy is dominated by rembg silhouette + convex-hull precision, not by the labeling logic itself.

## Sample panels

### 20_A (canonical broken-on-production row, axis misfit was 177.4° before)

![20_A](rectify_via_hull_labels_sample_20_A.png)

Top row: source image with hull-labels-derived face_quads overlaid (red). Middle row: 3 rectified faces produced by the hull-labels pipeline. Bottom row: 3 rectified faces from oracle ground-truth corners. The hull-labels rectifications are visually indistinguishable from oracle on this row, with a score delta of +18.8 (3% above oracle).

### 45_B

![45_B](rectify_via_hull_labels_sample_45_B.png)

Score delta +5.2 (<1% above oracle). Mean per-corner labeling error 21.7 px (worst row in the corpus); derived vertex error 67.4 px (worst row); still produces faces that are visually clean 3×3 sticker grids.

## What this eliminates from production

If this approach replaces the global_cube_model pipeline, the following production code becomes unused:

- `tools/global_cube_model.py::fit_cube_template_to_anchors` — the 720-perm Procrustes search, lex-first tie-breaking, the entire "which permutation of detected → template" enumeration
- `tools/global_cube_model.py::_solve_pnp_calibrated` + `_project_perspective` — PnP refinement and perspective projection
- `tools/global_cube_model.py::_resolve_near_far_phase` — chirality detector, line-darkness sampling, 60° body-diagonal flip correction
- `tools/global_cube_model.py::_refine_vertex_via_image_junction` + `_trihedral_junction_score` — image-based vertex refinement
- The mean-of-3 vertex ensemble (PnP + bezel + hex_centroid average)
- `tools/interior_bezel_detection.py` dependency — bezel detection is no longer needed for geometry; rembg silhouette alone suffices

Total: ~800 LOC of pipeline + ~300 LOC of bezel detection + the supporting chirality/phase-check tests and reports.

## When this approach may fail

1. **Strong cube tilt (>~30° from vertical).** The silhouette-position labeling assumes the cube is held roughly upright. Heavy tilt could shuffle which hull extremum lands at "TOP" vs "upper-right." Acceptable for cube-snap's app flow (instructs the user to hold the cube white-up); a concern for unconstrained capture.
2. **Heavy perspective.** The parallelogram-completion vertex derivation is exact under iso projection; under strong perspective (camera very close to cube), parallelograms become non-parallelograms and the vertex estimate drifts. On the 12-row corpus, vertex error stays within 70 px, but for tight close-ups the error may grow.
3. **Sides other than A and B.** `SILHOUETTE_TO_CORNER` only covers A and B today. New views (e.g., a side-on view, or the proposed Tier-3 automatic-pose flow) would need additional per-view mappings.
4. **Sticker-color scoring tied across degraded cases.** When vertex error is very large (e.g. PR #271's V1-V3 hypothesis-test probe on 38_B with 208 px vertex error), the color-distance score may not distinguish good from broken hypotheses. The current tool doesn't use sticker-color scoring at all — it's deterministic from the labeling — so this caveat doesn't apply here, but it's worth flagging if the approach ever needs a fallback "is this rectification any good" gate.

## Test plan

- [x] `tests/test_rectify_via_hull_labels.py` — 10 unit tests covering the silhouette-position labeling (canonical, mild tilt, both sides, error handling) + the parallelogram-completion vertex (iso-exact, perspective-robust, side parity) + the per-side mapping table sanity (covers both sides, consistent with `FACE_DEFS_BY_SIDE`).
- [x] `tests/fixtures/rectify_via_hull_labels_trace.json` — committed canonical 12-row trace.
- [x] Per-row gallery rendered to `/tmp/rectify_via_hull_labels/by_row/{key}.png`; 2 sample panels committed inline above.
- [x] Diagnostic-only — no production behavior change in `tools/global_cube_model.py` or `rubik_recognizer/`. Production pipeline is unaffected unless/until a follow-up wires this in.

## What's next (NOT in this PR)

1. **Expand corpus from 12 to 58 rows** (the axis-labeled gallery from PR #221). Verify the 100% rectification rate holds at larger N. Specifically watch for tilt-induced labeling failures.
2. **Vertex source ablation.** Compare parallelogram-completion vertex (this tool) vs bezel-detected vs hex-centroid vs production's mean-of-3 ensemble. Quantify which gives the cleanest rectification — possibly an ensemble of these could be even better than any single source.
3. **Production wiring.** If 1+2 pass, draft a PR replacing `fit_global_cube_model` with this approach. Substantial production change; would need a measured before/after across the full corpus and end-to-end recognizer accuracy comparison before merge.
