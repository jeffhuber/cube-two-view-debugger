# Auto-geometry evaluation

End-to-end commands for evaluating automatic cube-hull + face-quad
proposers against the hand-labeled ground truth from the Geometry
Labeler (Rubik Two-View Recognizer, port 8080).

## 1. Run the full evaluation

```bash
.venv/bin/python tools/evaluate_auto_geometry.py
```

- Discovers every (setId, side) that has both a hull label in
  `runs/labels/` AND a discoverable image (corpus_manifest or
  `/Users/jhuber/Downloads/Set N - ...JPG`)
- Runs **all four baseline proposers** on each target
- Writes:
  - `runs/auto_geometry_report.json` — full per-pair metrics
  - `runs/auto_geometry_summary.txt` — pretty per-proposer summary
  - `runs/auto_geometry_overlays/<proposer>/<set>-<side>.png` — visual
    overlays (proposed = solid red/cyan, ground truth = dashed yellow)

Filter by set: `--set 15 --set 17`.
Disable overlays for speed: `--no-overlays`.
Subset of proposers: `--proposers recognizer_grids saturation_hexagon`.

## 2. Baselines included

| Proposer | Approach | Strength |
|---|---|---|
| `recognizer_grids` | `analyze_image()` → best `FaceGrid` per `center_face` (by matched_count, fit_error) → 4-point homography from unit-square interior `(1/6, 1/6)..(5/6, 5/6)` to grid outer-sticker centers → evaluate at `(0,0)..(1,1)` for face corners. Cube hull = convex hull of all sticker centers, expanded outward by 13% to cover bezels. | Best baseline by hull IoU (~0.85 mean). |
| `saturation_hexagon` | Saturation-thresholded connected component → largest blob mask → convex hull → fit a 6-vertex hexagon (farthest hull vertex per angular sector). Face quads derived via the Geometry Labeler's template formula (`top = [h0, h1, center, h5]`, etc.). | Fully automatic with no recognizer dependency. Robust to recognizer failures but weaker on textured backgrounds. |
| `saturation_hull` | Same as `saturation_hexagon` but returns the full convex hull and no face quads. | Sanity baseline for the hexagon fit step. |
| `roi_bbox` | `_find_cube_roi()` only — returns the bounding box as a 4-vertex hull. No face quads. | Lower bound; mostly to verify the metric pipeline. |

## 3. Metrics

For each (target, proposer):

- **`cubeHullIoU`** — mask-based polygon IoU of proposed hull vs hand-labeled `cubeHull`
- **`perFace.{U,R,F,D,L,B}.iou`** — face-quad IoU, matched by face *label*. A proposer that puts the right geometry under the wrong label scores 0 here.
- **`perFace.{...}.containment_of_gt`** — fraction of GT face area covered by the proposal (catches over-proposal that still covers the truth)
- **`perFace.{...}.mean_corner_error_px`** — mean nearest-corner distance after canonical CW-from-north ordering
- **`meanFaceIoU_byLabel`** — mean of `perFace.*.iou`. Penalises label mistakes.
- **`meanFaceIoU_bestMatch`** — Hungarian best-assignment of proposed quads to GT quads, normalised by `max(|proposed|, |gt|)`. Reports geometric correctness independent of face-label correctness.

Pass thresholds (used in the summary):
- `cubeHullIoU >= 0.85` for hull
- `meanFaceIoU_bestMatch >= 0.75` for face quads

## 4. Current state (60 (setId, side) targets across all 30 hull-labeled sets)

```
proposer                           n    hullIoU   face(byLabel)   face(bestMatch)
--------------------------------------------------------------------------------
recognizer_grids                  60      0.851           0.429             0.364
saturation_hexagon                60      0.540           0.366             0.372
saturation_hull                   60      0.513           0.000             0.000
roi_bbox                          60      0.334           0.000             0.000

pass rates (per-pair):
  recognizer_grids   hull≥0.85: 36/60 (60%)  face≥0.75: 0/60 (0%)  both: 0/60 (0%)
  saturation_hexagon hull≥0.85:  0/60 (0%)   face≥0.75: 0/60 (0%)  both: 0/60 (0%)
```

**Takeaway.** Classical CV gives us decent cube hull localization (60% of pairs pass `hullIoU >= 0.85` with `recognizer_grids`) but **no proposer crosses the face-quad pass bar**. Face-quad accuracy is the bottleneck — best is `saturation_hexagon` at 0.37 mean best-match IoU.

## 5. Failure modes worth investigating

From the per-pair report:

- **`recognizer_grids` worst cases** (hull IoU 0.58-0.72): Sets 25/30/36 image A — the recognizer's grid fitting struggles on these specific photos.
- **`saturation_hexagon` worst cases** (hull IoU 0.07-0.09): Sets 25, 30, 31 — the saturation mask is contaminated by background (wood grain, shadow) so the convex hull is way larger than the cube and the hexagon-vertex extraction degenerates.
- **Face-quad label mismatches** propagate the U-logo and L/B-flip issues we documented in PR #126 to this evaluation — `meanFaceIoU_byLabel < meanFaceIoU_bestMatch` for `saturation_hexagon` confirms that geometry is often right when label is wrong.

## 6. What to try next

Likely highest-leverage:

1. **SAM2 zero-shot proposer** — feed each photo to SAM2, take its primary mask, hexagon-fit, derive face quads. Should fix the wood-grain failures and probably push hull IoU above 0.95 on most pairs.
2. **Better face-quad derivation from sticker grids** — the homography extrapolation from interior sticker positions to face corners is sensitive to bezel size; a learned offset or per-cube calibration would help.
3. **Use the recognizer's `_select_grid_combo` instead of my "best per `center_face`" heuristic** — that's the internal logic the recognizer already uses to pick the right 3 grids.
4. **Train a small U-Net or keypoint regressor on the existing 70 hull labels** — eliminates classical brittleness, ships as ONNX weights.

This is investigation tooling only — no production-recognizer changes were made. Wiring any of (1)-(4) into the live recognizer would be a separate PR.
