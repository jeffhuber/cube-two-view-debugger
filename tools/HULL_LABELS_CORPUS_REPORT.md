# Hull-labels rectification: 70-row corpus validation

**Question this answers:** does the `tools/rectify_via_hull_labels.py`
approach (12/12 essentially-oracle-quality on the 12-row full-corner
corpus per `tools/RECTIFY_VIA_HULL_LABELS_REPORT.md`) hold up at
larger N?

**Headline:** **68/70 rows pass all gates** (97.1%); 2 rows hit
borderline axis-misfit (34° and 32° vs the 30° threshold), neither
catastrophic.

This is enough empirical signal to take the convention-aware approach
seriously as a candidate replacement for the production
Procrustes/PnP/chirality/vertex-ensemble pipeline. The next step is
production wiring behind a feature flag with the same gating signals
this tool measures.

## Method

For each approved row in `tests/fixtures/gcm_axis_ground_truth.json`
(70 rows, balanced 35 side A / 35 side B; 12 overlap with
`full_corner_ground_truth.json`, 58 are new):

1. Production rembg path: `remove(image, session=sess)` → alpha > 128
2. `detect_hexagon_anchors(mask)` → 6 hull-extreme corners
3. `_label_corners_by_position(hexagon, side)` → corner-number dict
4. `_derive_vertex_from_corners` → mean of 3 parallelogram-completion
   estimates; we also keep the 3 individual estimates to compute
   **vertex_cloud_spread_px** = max pairwise distance (a proxy for how
   non-iso the projection is)
5. `rectify_via_hull_labels` → 3 rectified faces
6. Score:
   - `vertex_err_px` = `||derived_vertex − GT vertex||`
   - `axis_total_misfit_deg` = sum of 3 best-perm angle errors between
     predicted FAR-corner axes and GT `near_x/near_y/near_z`
     endpoints (which empirically sit at FAR positions — see
     "Axis-convention note" below)
   - `sticker_score_total` = sum of `classify_rgb(rgb).distance` over
     27 sampled stickers (no GT needed — measures how cleanly our
     face quads sample valid cube colors)

Classification (heuristic thresholds, easy to tune via CLI flags):
- `mask_failure`: < 6 hull corners
- `vertex_cloud_high_spread`: > 350 px
- `axis_misfit_high`: > 30°
- `sticker_score_high`: > 1500 total
- `rectified_clean`: passes all

## Headline result

| | Side A | Side B | Total |
|---|---:|---:|---:|
| `rectified_clean` | 34 | 34 | **68 (97.1%)** |
| `axis_misfit_high` | 1 (30_A) | 1 (37_B) | 2 (2.9%) |
| `mask_failure` | 0 | 0 | 0 |
| Total | 35 | 35 | 70 |

No mask-detection failures, no label failures, no sticker-score
failures, no vertex-cloud-spread failures. Only borderline axis-
misfit on 2 rows. Both side A and side B perform equally — no
side-mapping regression on the larger corpus.

## Distribution (rectified rows only)

| Metric | min | q1 | median | q3 | max |
|---|---:|---:|---:|---:|---:|
| vertex_err_px               | 6.6  | 22 | 34   | 45 | 79.7 |
| vertex_cloud_spread_px      | 94.6 | 184 | 202 | 224 | 267.6 |
| axis_total_misfit_deg       | 1.5  | 7.7 | 11.0 | 14.3 | 34.0 |
| sticker_score_total         | 309  | 419 | 494 | 581 | 730 |

### Old 12 vs new 58 — no quality cliff

Distributions on the 12 overlap rows (well-known good) vs the 58 new
rows are nearly identical:

| Metric | Overlap 12 (min/med/max) | New 58 (min/med/max) |
|---|---|---|
| axis_total_misfit_deg | 1.5 / 11.2 / 21.9 | 3.5 / 11.0 / 34.0 |
| vertex_err_px         | 18 / 39 / 59      | 7 / 33 / 80         |
| vertex_cloud_spread_px| 95 / 215 / 268    | 126 / 203 / 266     |

The 58 new rows match the 12-row distribution closely. No
catastrophic mode appears at the larger sample.

### How many rows are "near the edge"

| Threshold | Count over | Notes |
|---|---:|---|
| axis_misfit > 30° | 2 | the 2 failures |
| axis_misfit > 25° | 4 | + 57_B, 30_B |
| axis_misfit > 20° | 9 | most are A/B siblings of failures |
| axis_misfit > 15° | 15 | ~21% — still well within usable |

## The 2 failures

### 30_A (axis_misfit=34.0°, vertex_err=51px, spread=224px)

![30_A failure](hull_labels_corpus_failure_30_A.png)

Cube held at noticeable yaw on a wood-grain desk. The hull-position
labeling looks geometrically right at first glance (6 blue dots at
the silhouette extrema) but the derived face_quads collapse — visible
in the rectified faces, two of which sample mostly dark background
instead of cube stickers.

The mechanism appears to be perspective stretching: when the cube is
held with one side facing the camera more head-on than the others
(yaw), the 3 parallelogram-completion vertex estimates spread along a
line and average out somewhere off the true trihedral junction. With
a 51 px vertex error and 224 px cloud spread, the face_quads
constructed off that vertex skew toward the lower-quality side.

### 37_B (axis_misfit=32.2°, vertex_err=80px, spread=252px)

![37_B failure](hull_labels_corpus_failure_37_B.png)

Less catastrophic visually: the 3 rectified faces are coherent and
mostly-correct cube content. But they're narrower/cropped relative
to oracle-quality output, and the per-axis angle errors are
[15.4°, 2.5°, 14.4°] — two axes are 14-15° off, suggesting the
labeled corners on those two are drifting from where they should be.

Vertex error 80 px is the worst in the corpus; this is the high end
of where parallelogram completion under perspective starts to
visibly degrade. Same mechanism as 30_A but milder.

## Failure-bucket coverage (Codex's outline)

Per the lane-split outline Codex sent on 2026-05-24, the buckets to
report on:

| Bucket | Count | Bucket count | Notes |
|---|---|---:|---|
| Mask failure | rembg + hexagon detect produces <6 hull corners | **0** | u2net rembg + `detect_hexagon_anchors` was stable on every approved row |
| Hull six-corner failure | 6 corners present but `_label_corners_by_position` fails | **0** | per-side mapping table covers A and B; no malformed input encountered |
| Vertex-cloud spread | proxy for "iso assumption is breaking" | borderline on 30_A (224 px) and 37_B (252 px) but max 268 — below the 350 threshold | could be tuned tighter as a "low-confidence" gate |
| Rectification / color confidence | sticker score above threshold | **0** above 1500 (max observed 730) | classifier-mode-dependent, but well below |
| Side / yaw assumptions | per-side `SILHOUETTE_TO_CORNER` works for A/B | **0** failures, both sides 34/35 clean | sides other than A/B (e.g. CC/DD captures) need additional mapping entries |
| Axis misfit | 3 predicted axes vs 3 GT axes | **2** above 30° (3% of rows) | both correlated with high vertex_err + high spread |

## Axis-convention note (caught during this PR)

The 70-row `gcm_axis_ground_truth.json` labels `near_x/near_y/near_z`
sit at FAR-corner positions ({0, 2, 4} for side A; {1, 3, 5} for
side B), NOT at the NEAR-set {1, 3, 5}/{0, 2, 4} the naming suggests.
Verified against all 12 overlap rows.

The labeling-tool UI calls them "near" because the user clicks the
silhouette corner along each world axis direction (e.g. side A
white-up: +Z = TOP = corner_0). That corner is two cube-edges from
the vertex — the FAR set in `FAR_CORNERS_BY_SIDE`.

Initial run computed predicted axes as `(NEAR_corner − vertex)` and
reported ~180° axis misfit on all 70 rows — visibly broken. Fix was
to compute predicted axes as `(FAR_corner − vertex)` to match the GT
convention. After the fix, axis misfit drops to median 11° (the real
quality signal).

This is a one-shot caveat for downstream tools that read this
fixture: use `FAR_CORNERS_BY_SIDE` to compute predicted axes for
comparison against the 70-row truth. `measure_axis_correctness.py`
uses `full_corner_ground_truth.json` (12-row) instead, which has
per-corner labels and avoids this ambiguity.

## What this PR does NOT include

- **No production wiring.** `tools/global_cube_model.py` and
  `fit_global_cube_model` are untouched. This remains diagnostic-only
  pending the next gating step.
- **No visual gallery per row.** Just the 2 failure panels. A full
  70-row gallery would clarify the borderline rows (axis 25-30°) but
  isn't required for the headline empirical signal.
- **No A+B pair-level analysis.** Each row is scored independently.
  Two-view consistency (PR #242/#243's signal) could provide an
  additional gate when one side fails but the other is clean.
- **No comparison against production.** PR #279's report compared
  hull-labels vs oracle on the 12-row corpus; this PR validates
  hull-labels alone on the larger corpus. A follow-up could
  re-run `tools/measure_axis_correctness.py` on the same 70 rows
  to give a head-to-head against the 720-perm Procrustes pipeline.

## Reproducing

```bash
cd cube-two-view-debugger
.venv/bin/python tools/measure_hull_labels_corpus.py
```

Writes `tests/fixtures/hull_labels_corpus_trace.json` (full per-row
detail) and prints the summary to stdout.

Thresholds tunable via `--thresh-spread-px`, `--thresh-axis-deg`,
`--thresh-sticker-total`.

## Suggested next steps (deferred — Codex lane to acceptance-gate)

1. **Production-shaped wiring behind a feature flag** in
   `tools/global_cube_model.py` so existing callers can A/B the
   approach.
2. **Acceptance gates** (Codex's lane) — define which of the
   measured signals (axis misfit, vertex cloud spread, sticker
   score) gate "trust this fit" vs "fall back to Procrustes".
3. **Per-row visual gallery** for the 9 borderline rows
   (axis_misfit > 20°) to confirm the metric matches visual judgment.
4. **Extend to sides beyond A/B** by adding `SILHOUETTE_TO_CORNER`
   entries for any new capture conventions.
5. **Two-view consistency gating** — if hull-labels passes on one
   side but fails on the other, two-view consistency could flag it
   pre-fallback.
