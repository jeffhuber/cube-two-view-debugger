# Synthetic Rubik's-cube corpus — v1

Day-1 deliverable: a PIL-based renderer that produces paired (image A,
image B) synthetic cube photos from a 54-char URFDLB state string, with
ground-truth metadata (cube hull, face quads, per-sticker positions) per
image. Tooling-only, no production-recognizer changes.

## Why

Real-photo corpus (~30 sets, ~70 hull-labeled photos) is small, all
captured in a similar setting. The recognizer's measured ceiling (~97.95%
clean-label accuracy from PR #126) is on **that** distribution. Out-of-
distribution lighting/backgrounds (wood grain, different rooms, different
cameras) haven't been tested.

Synthetic data unlocks unlimited variety with perfect ground truth at
zero labeling cost. v1 is the geometric scaffolding; v2 adds the variety.

## Render one pair

```bash
.venv/bin/python tools/render_synthetic_cube.py \
  --state UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB \
  --output /tmp/syn-solved
```

Writes:
- `/tmp/syn-solved_A.png` (1150×862 portrait, URFDLB view A — U on top, F front, R right)
- `/tmp/syn-solved_B.png` (post-flip view B — D on top, L on viewer's left, B on viewer's right)
- `/tmp/syn-solved.json` (state + cubeHull + faceQuads + per-sticker pixel positions)

Per-image metadata schema:

```json
{
  "imagePath": "...",
  "visibleFaces": ["U", "R", "F"],
  "cubeHull": [[x1, y1], ...],
  "faceQuads": {"U": [[...]], "R": [[...]], "F": [[...]]},
  "stickerPositions": {"U": [[[x,y], [x,y], [x,y]], ...]}
}
```

## What v1 produces

- 3D unit cube model with axis-aligned faces, parametric bezel and inter-sticker inset
- Pinhole perspective camera (one preset per view A/B)
- Flat per-face Lambertian-ish shading (U brightest, D dimmest, sides mid)
- **Vivid cube-snap palette** matching the CubeVisualizer's hex colors (`#b71234` red, `#009b48` green, `#ff5800` orange, `#0046ad` blue, etc.) — not the muted recognizer canonical palette
- **Dark cube body (#161616) showing between stickers** — the visual signature that makes the cube read as a real Rubik's cube rather than a polygon mesh
- **Rounded sticker corners** via quadratic-Bezier chamfered polygons (`rounded_quad` helper, matches cube-snap's STICKER_RADIUS = 0.10 look)
- Solid-color background (parameterizable RGB)
- Pixel-perfect ground truth (face quads computed from projection; not approximated)

## What v1 DOESN'T do (deferred to v2)

- **Random background variation** — only solid color today. v2 adds procedural (wood grain, marble) + composite onto real backgrounds.
- **Lighting jitter** — flat shading per face today. v2 adds per-render light direction + intensity randomization.
- **Camera jitter** — fixed pose per view. v2 jitters pitch/yaw/distance to simulate handheld photography.
- **Material realism** — flat colors. v2 adds sticker glints, JPEG noise, mild blur.
- **Configuration generator** — single-state CLI today. v2 adds random-valid-state generation (~5000 pairs).
- **Cube physical variation** — fixed bezel/sticker proportions. v2 varies these per render to simulate different physical cubes.

## Tests

```bash
.venv/bin/python -m pytest tests/test_render_synthetic_cube.py -v
```

11 cases covering state parsing, 3D geometry sanity, sticker-on-face-plane invariant, camera projection bounds for both views, render output schema, color-correctness at center stickers.

## Quality bounds (per the scoping)

The full plan (from the scoping conversation) has 3 acceptance tiers:

1. **Visual sanity** — render 100 random samples, eyeball them.
2. **Recognizer transfer** — run `analyze_image` on synthetic; measure sticker count, color classification accuracy vs known GT, hull IoU vs known. Pass: ≥85% per-sticker color accuracy.
3. **Training transfer** — train classifier on synthetic only, evaluate on real corpus (the 1512 samples from PR #126). Pass: synthetic-trained classifier within 3pp of real-trained.

v1 is at Tier 1 readiness. Tier 2 + 3 evaluation comes after v2 adds the background/lighting variety needed for a fair test (current uniform-gray-background renders are too simple to evaluate transferability honestly).

## Conventions

- State string: 54 chars URFDLB order (cubejs / cube-snap canonical).
- Centers: position 4 of each chunk = face's own color (Rubik's invariant).
- View A: camera at +x, +y, +z octant; up = +y. Sees U/R/F.
- View B: camera at -x, -y, -z octant; up = -y (so D appears at image top). Sees D/L/B.
- Image coords: y grows downward (PIL/numpy convention).

## Not for merge yet

This is the v1 starting point. Open as a PR for review of the rendering
approach, geometry, and metadata schema. Iteration on v2 (variety) will
land as follow-up PRs in this branch or successors.
