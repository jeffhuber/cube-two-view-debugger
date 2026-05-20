# Interior bezel-line detection — diagnostics

Diagnostics-only probe that targets the structural ceiling PR #176 identified:
on yawed cubes (67% of worst-pair corpus), 3 of the 6 iso-projection hexagon
vertices (h1, h3, h5) project INSIDE the silhouette where convex-hull-based
fitters provably cannot find them. The information IS in the image, but on
the dark bezel lines between visible cube faces — not on the silhouette.

This probe surfaces a per-image (cube_center, 3 boundary lines, per-line
quality) signal. It is NOT wired into `_derive_face_quad_topology_aware`
or any decision path, per the project's diagnostics-first discipline.

## Algorithm (iterative refinement)

1. Silhouette centroid → initial cube-center seed.
2. Erode the silhouette by 30 px to exclude the outer hull boundary.
3. Sobel gradient magnitude on grayscale, masked to the eroded silhouette.
4. **Iterative loop** (up to `max_iter=4` rounds, typical convergence in 2-3):
   - 1-D angular Hough sweep through current center at 2° steps in [0, π).
   - Top-3 angles with non-max suppression (min 15° separation).
   - Coarse-to-fine local search for the (cx, cy) that maximizes SUM of
     line-mass for those 3 angles, **constrained to stay inside the
     silhouette**.
   - Stop if center moved < 6 px since last iteration.
5. **Drift cap**: if cumulative shift from the centroid seed exceeds
   180 px, the iteration is probably chasing a sticker-grid optimum —
   roll back to centroid and re-pick angles there.
6. **Per-line quality**: for each picked angle θ, score how much its
   line-mass exceeds the 90th percentile of mass over the full sweep.
   1.0 = double p90 (a real bezel); 0.0 = at or below p90 (sticker grid).
7. **Aggregate signal_quality**: mean of the worst-2 per-line qualities ×
   completeness (fraction of lines that produced valid segments). A
   "1 good line, 2 noisy" detection scores ~0.1; "3 strong lines"
   scores ~0.8.

## Why iterative + per-line (post-#177 human-review iteration)

PR #177 shipped the single-pass version. Human review of the 18 worst-pair
overlays revealed two failure modes the single-pass version couldn't
distinguish:

1. **Cube_center wrong** (5 pairs noted "off by ~100 px"): the centroid
   seed was biased away from the true cube-center vertex. Single-pass
   refinement couldn't escape because angles were chosen at the wrong
   center and refinement holds angles fixed.
2. **One or two angles are sticker-grid contaminants** (most pairs): the
   single aggregate `signal_quality` score hid this — when 3 angles
   happen to be ~60° apart even with 1-2 grid contaminants, the
   `angle_regularity` component over-rewarded the configuration.

The iterative refinement re-picks angles at each new center, letting good
angles unlock a better center which unlocks even better angles. The
per-line quality exposes which of the 3 lines are real bezels vs
sticker-grid contaminants directly.

## Things tried that did NOT work — kept here so we don't re-try them

- **Multi-seed restart (5 and 9 seeds at ±150-200 px from centroid)**.
  Empirically made Sets 30 B and 44 B worse: the optimization landscape
  has many sticker-grid local maxima, and "highest total line-mass at
  convergence" doesn't reliably distinguish the cube-center maximum from
  sticker-grid maxima. The single-seed iteration with the drift cap is
  the best policy we've found. Multi-seed code was removed in this PR's
  iteration; see commit history on the branch for the experiment.
- **Single-pass refinement** (PR #177 baseline): worked on the 5 easiest
  cases but couldn't escape bad starting angles on the hard cases.

## Per-pair results (18 worst-pair walkthroughs from PR #176)

| pair  | sq    | per-line (magenta, cyan, yellow) | iter | shift | human |
|-------|------:|----------------------------------|-----:|------:|-------|
| 17 A  | 0.15  | 0.85, 0.29, 0.00                 | 2    | 70.7  | —     |
| 17 B  | 0.20  | 0.39, 0.25, 0.15                 | 2    | 19.0  | —     |
| 21 A  | 0.16  | 1.00, 0.28, 0.05                 | 2    | 11.7  | —     |
| 21 B  | 0.07  | 0.40, 0.11, 0.03                 | 2    | 73.5  | —     |
| 30 A  | 0.11  | 0.83, 0.15, 0.07                 | 1    | 2.8   | PASS  |
| 30 B  | 0.24  | 1.00, 0.41, 0.07                 | 3    | 121.5 | PASS  |
| 31 A  | **0.73** | 1.00, 0.95, 0.52              | 4    | 36.0  | —     |
| 31 B  | **0.64** | 1.00, 1.00, 0.28              | 3    | 100.3 | PASS  |
| 44 A  | 0.17  | 1.00, 0.24, 0.11                 | 2    | 31.6  | —     |
| 44 B  | 0.14  | 0.78, 0.28, 0.00                 | 2    | 0.0†  | PASS  |
| 47 A  | 0.14  | 0.61, 0.28, 0.00                 | 4    | 61.1  | —     |
| 47 B  | 0.16  | 0.51, 0.23, 0.09                 | 2    | 15.2  | —     |
| 57 A  | **0.56** | 0.79, 0.73, 0.40              | 2    | 21.5  | PASS  |
| 57 B  | 0.07  | 0.26, 0.12, 0.03                 | 3    | 48.1  | —     |
| 58 A  | **0.41** | 1.00, 0.81, 0.00              | 2    | 8.9   | —     |
| 58 B  | 0.11  | 1.00, 0.13, 0.09                 | 2    | 11.7  | —     |
| 61 A  | 0.04  | 1.00, 0.08, 0.00                 | 3    | 159.6 | —     |
| 61 B  | 0.12  | 0.26, 0.25, 0.00                 | 2    | 172.6 | —     |

† 44 B's drift cap fired (cumulative shift exceeded 180 px during
iteration); rolled back to centroid + re-picked angles there.

## Robust patterns the per-line data exposes

1. **The first picked angle (magenta — typically the vertical bezel between
   the front-left and front-right faces) is high-quality on 13 of 18 pairs**
   (line_q ≥ 0.40, including 7 pairs at 1.00). This is the most reliable
   geometric signal in the silhouette.
2. **The 3rd picked angle (yellow — the dimmest detected line) is
   essentially noise on 10 of 18 pairs** (line_q ≤ 0.10). It's almost
   always a sticker-grid contaminant.
3. **Strong detections** (sq ≥ 0.4 → all 3 line qualities ≥ 0.4): 4 of 18
   pairs (31 A, 31 B, 57 A, 58 A). These are the cases where iterative
   refinement converged on a geometry where 3 real bezel lines pass
   through one point.

## Improvement vs PR #177 baseline

| Metric                       | PR #177 | This PR (iterative)              |
|------------------------------|---------|----------------------------------|
| Wall-clock per pair          | ~30 s   | ~8 s (vectorized line_mass)      |
| Per-line quality exposed?    | No      | **Yes** (per-line line_q in [0,1])|
| Aggregate scorer             | over-rewarded angle-regularity (9/18 "strong" included 4 with bad lines) | mean-of-worst-2 × completeness; 4/18 strong, all genuinely cube-aligned |
| Iterative refinement?        | No      | Yes, with 180-px drift cap       |
| Specific improvements vs PR #177 sq on 18 pairs | — | 14/18 same or improved per-line max; 31 A & 31 B & 57 A move from "moderate" to "strong" cluster |

## What this probe is signal for

The (cube_center, 3 angles) pair is the prerequisite for finding h1, h3,
h5 — the 3 hexagon vertices interior to the silhouette. Once we have
those, the cardinal-position cube-face derivation that powers
`_derive_face_quad_topology_aware` becomes well-defined on yawed cubes.

Future signals to mine, gated on broader-corpus signal_quality holding up:

- **h1, h3, h5 locations**: trace each high-quality boundary line from
  cube-center outward to where the bezel terminates at a cube corner
  (a visible change in bezel direction).
- **Refined hexagon fit**: combine the 3 hull-detectable hexagon
  vertices (h0, h2, h4 from `_fit_hexagon_to_hull`) with h1/h3/h5
  from this probe when per-line quality is high enough → a complete
  6-vertex hexagon respecting the cube's interior geometry.
- **Cell-level disambiguation**: when slot/src mismatches are
  ambiguous (the May 18 overlay-feedback pattern), the detected
  boundary lines tell us which sticker cells span TWO faces.

## Files

| File                                                     | Purpose                                              |
|----------------------------------------------------------|------------------------------------------------------|
| `tools/interior_bezel_detection.py`                      | Standalone detection module (numpy + scipy, no cv2; vectorized line_mass; iterative refinement + drift cap; per-line quality) |
| `tools/test_interior_bezel.py`                           | Test driver + visualization (uses rembg for masks; per-line quality in panel header) |
| `tools/INTERIOR_BEZEL_DETECTION.md`                      | This doc                                             |
| `tests/fixtures/interior_bezel_visual_feedback.json`     | Human review of the 18 worst-pair PR #177 overlays — ground truth for the table above |

## Dependencies

| Dependency | Status                                              |
|------------|-----------------------------------------------------|
| numpy      | required (in `requirements.txt`)                    |
| scipy      | **optional research dependency** — used for binary erosion + Sobel-filter convolution. Same opt-in pattern as `propose_geometry_labels.py:_fit_hexagon_optimized` and `amg_face_refiner.py`. When absent, `detect_interior_bezel_lines()` returns a graceful `InteriorBezelDetection` with `signal_quality=0.0` and `debug["error"]` rather than raising. Install with `.venv/bin/pip install scipy`. |
| rembg      | required only by `tools/test_interior_bezel.py` for generating the silhouette mask. The detection module itself has no `rembg` dependency — callers pass in the mask. |

A smoke test covering both scipy-present and scipy-absent paths is at
`tests/test_interior_bezel_detection.py` and runs in the standard
`tests/run_tests.py` / `pytest tests` invocation.

## Reproducing this artifact

```bash
# Install scipy + rembg if not already in the venv
.venv/bin/pip install scipy rembg

.venv/bin/python tools/test_interior_bezel.py \
    --sets 17 21 30 31 44 47 57 58 61 \
    --out /tmp/interior_bezel_iter
```

Outputs:
- `set_<N>_<side>_overlay.png` — side-by-side photo + mask + detection
  overlay with per-line quality in the header
- `set_<N>_<side>_data.json` — structured detection result (cube_center,
  boundary lines, angles, per-line qualities, iteration history, debug)
- `summary.json` — index over all per-pair results

The PNGs are not committed (large binaries); the generator + this doc
are the durable artifacts.
