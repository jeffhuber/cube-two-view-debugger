# `rectify_faces.py` — perspective-correct cube faces from quads

Given an isometric cube image and 3 face-quad polygons (4 corners each),
produce flat 300×300 face images via projective rectification. Per-sticker
color extraction then becomes trivial pixel slicing — no homography per
sample, no inset tuning, no canonical-corner ambiguity.

## Why this exists

The recognizer pipeline currently does per-sticker homography sampling on
perspective-warped face quads. That's correct but:

- **Sample positions are fiddly**: bezel inset, sticker inset, corner
  ordering, orientation discovery — all problems we documented in #126.
- **Visual debugging is hard**: a perspective rhombus doesn't tell you
  "is this red or orange" the way a flat 100×100 patch does.
- **Synthetic-vs-real comparison is muddled**: the synthetic renderer
  produces clean cubes; comparing to perspective real cubes mixes
  geometry and color signal.

This tool makes the perspective rectification the *first* step. After
rectification:

- 9 sticker centers are at known pixel positions (`(50,50), (150,50), ...`
  for a 300×300 face)
- Color extraction is `arr[r*100+25:r*100+75, c*100+25:c*100+75].median()`
- Visual inspection is trivial: each face looks like a synthetic flat net
- The flat output is the same shape as the synthetic renderer's per-face
  output (PR #132), enabling apples-to-apples sim-to-real testing
- Same concept as the Fixer's "diamond" view — just made into the
  recognizer's internal representation

## Render one set

```bash
.venv/bin/python tools/rectify_faces.py --set 15 --net
```

Writes:

- `/tmp/rect-set15_A_U.png`, `/tmp/rect-set15_A_R.png`, `/tmp/rect-set15_A_F.png`
- `/tmp/rect-set15_B_D.png`, `/tmp/rect-set15_B_L.png`, `/tmp/rect-set15_B_B.png`
- `/tmp/rect-set15.json` (per-sticker RGB, classified color, GT comparison)
- `/tmp/rect-set15_net.png` (combined view A + view B in a flat-net grid)

## Render an explicit image + hull label

```bash
.venv/bin/python tools/rectify_faces.py \
  --image "/Users/jhuber/Downloads/Set 15 - A - white up IMG_6707.JPG" \
  --hull-label runs/labels/<id>-set-15-a-geometry-label.json \
  --output /tmp/my-render
```

## Per-sticker accuracy on the 30 hand-labeled sets

Per-sticker color classification accuracy when sampling from
hand-labeled face quads (no auto-geometry; just the projective transform
+ default `classify_rgb`):

| Set | Accuracy | Notes |
|---|---|---|
| 15 | 54/54 (100%) | clean case |
| 30 | 53/54 (98%) | one of the harder corpus sets |
| 46 | 45/54 (83%) | wood-grain background; R center mis-IDs |
| 57 | 52/54 (96%) | new OOD challenging-lighting set |
| 58 | 51/54 (94%) | new OOD |
| 61 | 49/54 (91%) | new OOD |
| 62 | 48/54 (89%) | new OOD |

Mean across 4 new OOD challenge sets: **92.5%**. The residual errors are
known limitations of the default `classify_rgb` (red/orange confusion on
shadowed faces), not rectification errors — the rectified images visually
show the correct stickers in the correct positions.

## What this enables next

- **Drop-in input format for the future `recognizer_mask.py`**: rembg →
  hexagon → face quads → rectified faces → trivial color sampling.
- **CNN-friendly format for a learned classifier** (one of the
  recognizer-improvement paths in #126's strategy).
- **Sim-to-real testing for the synthetic corpus**: render synthetic
  cube to 300×300 face images, render real cube to same format, compare.
- **Fixer-style correction UI**: rectified faces are MUCH easier to
  hand-correct than perspective views.

## Conventions

- Output face image: square (default 300×300), in 8-bit RGB.
- Sticker centers in rectified coords: `((col + 0.5) * size/3, (row + 0.5) * size/3)`.
- Sampling patch: 40% of cell width by default (well inside the sticker).
- Corner ordering: canonical CW-from-N from `sample_stickers_from_hull`
  (consistent across the toolchain).
- Face identity for GT comparison: trust labeler's name for U/D anchors;
  for non-anchor faces, classify the rectified center sticker and reverse-
  lookup color → face (handles L/B swap from #126).
- Orientation: same `discover_orientation` brute-force from #126 so the
  rectified row-major output aligns with the GT URFDLB row-major chunk.

## Drive-by fix included

`tools/extract_color_samples.py:133` regex updated to accept both `white up`
(legacy) and `white-up` (newer iPhone-export naming, Sets 57/58/61/62
onwards). Without this, the new OOD sets weren't discoverable.

## Not in this PR

- Auto-geometry integration (rembg → mask → hexagon → rectify): waiting on
  the RANSAC PR's hexagon-fit results to validate.
- Learned classifier on rectified inputs: separate PR.
- Production recognizer integration: only after RANSAC PR validates face
  IoU is good enough.
