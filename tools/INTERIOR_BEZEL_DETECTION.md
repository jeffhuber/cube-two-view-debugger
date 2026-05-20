# Interior bezel-line detection — diagnostics

Diagnostics-only probe that targets the structural ceiling PR #176 identified:
on yawed cubes (67% of worst-pair corpus), 3 of the 6 iso-projection hexagon
vertices (h1, h3, h5) project INSIDE the silhouette where convex-hull-based
fitters provably cannot find them. The information IS in the image, but on
the dark bezel lines between visible cube faces — not on the silhouette.

This probe surfaces a per-image (cube_center, 3 boundary lines) signal. It
is NOT wired into `_derive_face_quad_topology_aware` or any decision path,
per the project's diagnostics-first discipline.

## Algorithm

1. Silhouette centroid → initial cube-center seed.
2. Erode the silhouette by 30 px to exclude the outer hull boundary.
3. Sobel gradient magnitude on grayscale, masked to the eroded silhouette.
4. 1-D angular Hough sweep through the centroid: for each θ in [0, π) at
   1° steps, sum the gradient magnitude along a line through (cx, cy)
   at angle θ.
5. Top-3 angles with non-max suppression (min 15° separation).
6. Local search around the centroid for the (cx, cy) that maximizes the
   SUM of line-masses for the 3 chosen angles (coarse 20-px steps in a
   ±200-px window, then fine 4-px steps in a ±30-px window).
7. Build boundary line segments from the refined cube-center outward to
   where each line exits the silhouette.
8. Heuristic signal-quality score from 3 components (geometric mean):
   - line_mass_ratio: top angular peak / median, normalized into [0,1]
   - angle_regularity: how close the 3 angles are to evenly spaced 60°
   - completeness: fraction of lines that produced valid segments

## Why a Hough-style approach beat sequential RANSAC

First-pass implementation used sequential RANSAC on top-10% gradient
pixels. On Set 47 A this produced 3 lines but ZERO valid intersections —
the lines were dominated by sticker-grid edges, which run parallel to
the cube's 3 face boundaries (because stickers tile aligned with the
face). RANSAC's random pair sampling found the densest collinear
clusters, which were sticker rows / columns, not bezels.

The constrained Hough-through-centroid approach exploits TWO additional
constraints the data satisfies:

- The 3 face boundaries all pass through ONE point (the cube-center
  vertex) — random-pair RANSAC ignores this.
- The cube-center vertex is approximately at the silhouette centroid
  for roughly-symmetric iso projection — gives a 0-cost initial seed.

Anchoring the line search at the centroid filters out parallel
sticker-grid lines: they may have high line-mass somewhere in the
image, but NOT along a line passing through the silhouette centroid.

## Per-pair results (18 worst-pair walkthroughs from PR #176)

| pair  | signal_quality | lines | mass_max/median | refined_shift_px | angles (°)        |
|-------|---------------:|------:|----------------:|-----------------:|-------------------|
| 17 A  | **0.80**       | 3     | 1.70            | 2.8              | 145, 85, 37       |
| 17 B  | **0.76**       | 3     | 1.67            | 17.2             | 144, 87, 40       |
| 21 A  | **0.95**       | 3     | 3.29            | 26.1             | 32, 89, 155       |
| 21 B  | 0.60           | 3     | 1.71            | 15.2             | 31, 92, 5         |
| 30 A  | **0.99**       | 3     | 2.24            | 2.8              | 92, 153, 33       |
| 30 B  | **0.97**       | 3     | 2.67            | 2.8              | 31, 152, 88       |
| 31 A  | **0.83**       | 3     | 2.77            | 40.0             | 88, 30, 167       |
| 31 B  | **0.94**       | 3     | 1.88            | 117.7            | 146, 27, 88       |
| 44 A  | **0.96**       | 3     | 2.66            | 11.7             | 149, 89, 24       |
| 44 B  | **0.99**       | 3     | 2.66            | 198.0            | 153, 32, 93       |
| 47 A  | **0.82**       | 3     | 1.67            | 6.3              | 92, 25, 153       |
| 47 B  | **0.86**       | 3     | 1.78            | 0.0              | 153, 26, 95       |
| 57 A  | **0.98**       | 3     | 2.11            | 28.4             | 27, 148, 86       |
| 57 B  | 0.56           | 3     | 1.70            | 0.0              | 61, 78, 22        |
| 58 A  | 0.69           | 3     | 2.72            | 6.3              | 28, 90, 54        |
| 58 B  | **0.87**       | 3     | 3.59            | 0.0              | 31, 159, 84       |
| 61 A  | 0.65           | 3     | 1.94            | 130.0            | 91, 126, 149      |
| 61 B  | 0.64           | 3     | 1.77            | 178.1            | 91, 73, 144       |

**Headline**: 13 of 18 pairs (72%) detected with signal_quality ≥ 0.70.
9 of 18 (50%) ≥ 0.85. 3 of 18 (Sets 21 A, 30 A, 44 A/B, 57 A, 30 B —
generally the LESS extreme yaws) ≥ 0.95.

## Visual validation (Set 47 A — the canonical extreme case)

PR #176 identified Set 47 A as one of two "extreme degeneracy" pairs:
the hexagon fitter's min_edge is 6.0 px (all 3 hexagon edges between
h0-h1, h2-h3, h4-h5 are collapsed), because only 3 of the 6 cube
corners project onto the hull.

The interior bezel detector finds:
- cube_center at (1366, 1874) — 6.3 px shift from the silhouette
  centroid
- 3 boundary angles at 92°, 25°, 153° — visually MATCH the 3 face
  boundaries in the photo (vertical between front-left & front-right
  faces, upper-right between top & front-right faces, upper-left
  between top & front-left faces)
- signal_quality 0.82

The walkthrough overlay in `set_47_A_overlay.png` shows the 3 detected
lines tracing along the actual cube-face boundaries through the
cube-center vertex.

## Failure-mode pattern on the 5 weak pairs (sq < 0.70)

All 5 weak cases (21 B, 57 B, 58 A, 61 A, 61 B) share the same shape:
the cube_center IS approximately correct (max line-mass anchor is
robust), but 2 of the 3 chosen angles ended up CLOSE TOGETHER (sticker
grid contaminants), not 60° apart.

Example — 61 A picked [91°, 126°, 149°]: a vertical line plus two
sticker-grid lines clustered in the 120-150° range. The angle_regularity
component of signal_quality is what penalizes these picks.

Two non-conflicting tuning directions, both LEFT FOR LATER (per
diagnostics-first discipline):

- **Wider NMS separation** (try 30-45° vs 15°): forces the top-3 picks
  to be ≥60° apart, matching the expected face-boundary geometry.
- **Joint (center, angles) optimization**: currently angles are chosen
  at the centroid then center is refined holding angles fixed. A
  second pass (re-sweep angles at the refined center) might pick
  cleaner peaks now that the geometry is better-anchored.

Neither is wired here — this is a probe shipping signal, not a tuned
production component.

## What this probe is signal for

The (cube_center, 3 angles) pair is the prerequisite for finding h1, h3,
h5 — the 3 hexagon vertices interior to the silhouette. Once we have
those, the cardinal-position cube-face derivation that powers
`_derive_face_quad_topology_aware` becomes well-defined on yawed cubes.

Future signals to mine, gated on this prototype's signal_quality holding
up across the broader corpus:

- **h1, h3, h5 locations**: trace each boundary line from cube-center
  outward to where the bezel terminates at a cube corner (a visible
  change in bezel direction).
- **Refined hexagon fit**: combine the 3 hull-detectable hexagon
  vertices (h0, h2, h4 from `_fit_hexagon_to_hull`) with h1/h3/h5
  from this probe → a complete 6-vertex hexagon that respects the
  cube's interior geometry.
- **Cell-level disambiguation**: when slot/src mismatches are
  ambiguous (the May 18 overlay-feedback pattern), the detected
  boundary lines tell us which sticker cells span TWO faces.

## Files

| File                                    | Purpose                                              |
|-----------------------------------------|------------------------------------------------------|
| `tools/interior_bezel_detection.py`     | Standalone detection module (numpy + scipy, no cv2) |
| `tools/test_interior_bezel.py`          | Test driver + visualization (uses rembg for masks)  |
| `tools/INTERIOR_BEZEL_DETECTION.md`     | This doc (results writeup)                          |

## Reproducing this artifact

```bash
# Requires rembg (.venv/bin/pip install rembg) for the silhouette mask
.venv/bin/python tools/test_interior_bezel.py \
    --sets 17 21 30 31 44 47 57 58 61 \
    --out /tmp/interior_bezel_results
```

Outputs:
- `set_<N>_<side>_overlay.png` — side-by-side photo + mask + detection
  overlay
- `set_<N>_<side>_data.json` — structured detection result (cube_center,
  boundary lines, angles, signal_quality, debug)
- `summary.json` — index over all per-pair results

The PNGs are not committed (large binaries); the generator + this doc
are the durable artifacts.
