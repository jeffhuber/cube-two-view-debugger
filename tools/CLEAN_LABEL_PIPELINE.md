# Clean-label color-classifier pipeline

End-to-end commands for reproducing the bake-off documented in PR #126.
All commands assume `cd /Users/jhuber/cube-two-view-debugger` (primary
checkout — `runs/labels/` is gitignored and lives here, not in worktrees).

## 1. Extract clean per-sticker dataset

```bash
.venv/bin/python tools/extract_clean_dataset.py
```

- Reads hull labels from `runs/labels/` and ground-truth states from
  `/Users/jhuber/Downloads/Set*-ground-truth-*.json`
- Runs geometric pipeline: 4-point homography per face quad → 9 sticker
  centers → 15×15 RGB patch median
- Joint multiset matching across A+B with adjacency + 6-distinct-faces
  invariant; ambiguous sets are skipped (not silently emitted)
- Outputs `runs/color_samples_geom.jsonl` (one line per sticker, ~1500
  samples across 28 sets)

Expected: perfectly balanced 252/color, 2 sets skipped (`ambiguous_face_id`
on currently-flagged sets 27, 28).

## 2. Visual verification of one set

```bash
.venv/bin/python tools/sample_stickers_from_hull.py 15 --inset 0.167 --suffix=_check
open /tmp/cal-geom-set-15-A_check.png /tmp/cal-geom-set-15-B_check.png
```

Every numbered dot should land inside its sticker, painted with that
sticker's true color. Eyeball this on the sets you care about before
trusting the dataset.

## 3. Re-run classifier bake-off

```bash
.venv/bin/python tools/train_color_classifier.py \
  --input runs/color_samples_geom.jsonl \
  --report runs/color_classifier_geom_report.json
```

- Leave-one-set-out cross-validation across 11 candidates
  (numpy: centroid / kNN / softmax; sklearn: LR / RF-200 / GBT-200)
- Stdout shows per-candidate accuracy + per-color precision/recall/f1
- JSON report written to `runs/color_classifier_geom_report.json`
  (includes confusion matrices + per-set deltas)

Expected: baseline ~95.6%, best learned candidate (RF-200) ~97.95%.

## 4. Unit tests

```bash
.venv/bin/python -m pytest tests/test_clean_label_pipeline.py -v
```

14 cases covering homography correctness, sticker-center geometry,
canonical corner ordering, orientation discovery, pair-level invariant
enforcement.

## Pipeline rationale

See PR #126 description and the `tools/sample_stickers_from_hull.py`
module docstring for the full design. Short version:

1. Hull labels (drawn via the **Geometry Labeler**, served by the Rubik
   Two-View Recognizer on port 8080) give 4-corner quads for each
   visible face.
2. A 4-point projective homography is **exact** for planar faces under
   pinhole projection — no per-set tuning, no per-cube calibration.
3. The 9 sticker centers fall at (1/6, 3/6, 5/6) of the unit square
   (inset value defends against bezel inclusion in hull labels —
   parameterized).
4. **Joint multiset face-ID** across A+B avoids the per-side bug where
   one face name could be assigned twice if classifier ambiguity
   pushed both labels to the same true face (Set 28's original failure
   mode).
5. **Adaptive palette** is built from the 5 non-U face centers (skipping
   U because of the Rubik's brand logo) and used for orientation
   discovery only — NOT for the per-sticker labels themselves, which
   come straight from the ground-truth state at the discovered (face,
   row, col) position.

## What this is NOT

- Not a production-recognizer change. `colors.py`, `recognizer.py`,
  `image_pipeline.py` are untouched.
- Not a Codex-style heuristic PR. Adds no thresholds.
- Not a final-classifier ship. The bake-off identifies a winning
  candidate (RF-200); a separate follow-up PR should hand-code the
  winner's inference in numpy and integrate it into `colors.py` behind
  an A/B path.
