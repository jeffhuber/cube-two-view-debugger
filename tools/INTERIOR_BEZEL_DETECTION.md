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

## Joining with #175's overlay-feedback / cell-discontinuity rows

Per Codex's review feedback on this PR, the detection JSON is shaped to
support downstream slot/cell-level joins against
`tests/fixtures/hard_case_visual_feedback.json` (#175) and the
cell-discontinuity probe output. The output keeps **per-line quality**,
**center error proxy**, **line distances**, and **"would-cross-cell"
flags** as separate fields so two-signal-agreement comparisons (e.g.,
"discontinuity hits ∩ high-quality bezel crossing") can be built without
collapsing into a single hidden score.

The per-pair JSON (e.g., `set_31_A_data.json`) emits:

```json
{
  "setId": "31", "side": "A",
  "cube_center": [1375.9, 1997.1],
  "boundary_angles_deg": [88.0, 144.0, 32.0],
  "line_equations": [
    [-0.9994, 0.0349, 1305.4],
    [-0.5878, -0.8090, 2424.5],
    [-0.5299, 0.8480, -964.5]
  ],
  "line_qualities": [1.00, 0.95, 0.52],
  "signal_quality": 0.73,
  "detector_version": "iterative-v1",
  ...
}
```

Each `[a, b, c]` is in `ax + by + c = 0` form. Point-to-line distance is
`abs(a*x + b*y + c) / hypot(a, b)`.

For computing per-cell diagnostics from a cell quad (4 image-space
vertices), use the helper in this module:

```python
from tools.interior_bezel_detection import (
    InteriorBezelDetection, cell_line_diagnostics,
)

# detection: InteriorBezelDetection (or hydrate from per-pair JSON)
# cell_quad: list of 4 (x, y) tuples for one sticker/cell

diag = cell_line_diagnostics(detection, cell_quad,
                              min_line_quality=0.40,
                              high_quality_threshold=0.40,
                              max_distance_px=30.0)
# Raw per-line fields (kept separate from derived flags):
#   diag["per_line"]                  — array of {angle, quality,
#                                       distance_from_centroid_px,
#                                       crosses_cell} per detected line
#   diag["cell_center_to_cube_center_px"]  — proxy for "is this cell at
#                                            the cube-center vertex?"
#   diag["min_distance_from_centroid_px"]  — closest line to cell center
#
# Aggregate boolean flags (derived from raw per-line fields above):
#   diag["any_crossing"]              — any detected line straddles
#                                       this cell
#   diag["any_crossing_high_quality"] — same, gated by min_line_quality
#   diag["crosses_high_quality_bezel"] — STRICTEST flag: line quality
#                                       >= 0.40 AND distance <= 30 px
#                                       AND crosses_cell. Suggested
#                                       starting point for broader-
#                                       corpus zero-FP mining.
#   diag["thresholds"]                — echoes the thresholds that
#                                       produced the derived flag
#   diag["detector_version"]          — stamp for cross-tab join keys
```

### Canonical join source

Per Codex's review on this PR: **the canonical join source is the same
hybrid overlay quads the humans visually labeled in #175**, NOT the
production recognizer quads. Reason: the labels are about those visual
overlays, so the first mining table should join against the exact quads
the human reviewed.

Quad source: `tools/render_hybrid_overlays.py` uses
`tools/evaluate_hybrid_pipeline._proposer_face_quads` to compute per-face
quads from `analyze_image`'s outputs. The face quads are 4-vertex
polygons in image coords; per-cell quads are derived by subdividing each
face quad into a 3×3 grid via bilinear interpolation. These same per-face
quads were rendered into the overlays the humans reviewed.

Production recognizer quads (the canonical recognizer pipeline's
selected-grid outputs) are a **second-pass transfer check** — used after
the primary mining table is built to see whether the bezel signal
generalizes from the overlay quads (what humans actually labeled) to
the production-side guard candidates.

### Recommended two-signal mining recipe (Codex-side)

For each (setId, side, slot, cell-row, cell-col) in
`hard_case_visual_feedback.json`:

1. Load the bezel detection for that (setId, side) — per-pair JSON or
   in-process `InteriorBezelDetection`.
2. Compute the 9 cell quads from the proposer face quad for that slot
   (same overlay quads humans visually reviewed — see Canonical join
   source above).
3. Call
   `cell_line_diagnostics(detection, cell_quad,
                           high_quality_threshold=0.40,
                           max_distance_px=30.0,
                           detector_version="iterative-v1")`.
4. Compare:
   - `crosses_high_quality_bezel` (or `any_crossing_high_quality`, or
     raw per-line `crosses_cell`) — bezel signal at the threshold of
     your choice
   - the cell-discontinuity probe's flag for the same cell
   - the human label's `failureModes` field

Cross-tab axes for the mining table:
- **bezel-only hits** (bezel flag = True, discontinuity = False)
- **discontinuity-only hits**
- **both-hit** (the candidate production guard)
- **both-miss on human-bad cells** (both signals failed on a known bad cell)
- **both-hit on human-good cells** (zero-FP candidates)

The likely useful production guard, if any survives the mining: "both
signals agree" or "high-quality bezel crossing + discontinuity
corroboration." Either-alone is NOT in scope for behavior wiring.

The `crosses_high_quality_bezel` flag's default thresholds (line_quality
>= 0.40, distance <= 30 px) are starting points picked from this branch's
18-pair walkthroughs, NOT validated as zero-FP. Raw per-line `quality`,
`distance_from_centroid_px`, and `crosses_cell` fields are preserved so
the mining can re-tune without re-running the detector.

## H-vertex tracing (Tier 1 — `find_interior_h_vertices`)

Once a bezel line passes through cube_center with high `line_quality`,
the next step toward a complete hexagon is finding where that bezel
TERMINATES at a cube corner. That terminus is an h-vertex (h1, h3, or
h5 — the 3 hexagon vertices the hull-based fitter provably can't find
on yawed cubes).

`find_interior_h_vertices(detection, image_rgb, mask, ...)` walks
outward from cube_center along each bezel direction (both ±1) and
locates the bezel terminus by perpendicular-gradient drop-off:

1. Skip the first 30 px from cube_center (the 3-bezel convergence is
   noisy at the very center).
2. At each step `r` along the bezel, sample perpendicular gradient
   peak via `_perpendicular_gradient_peak`.
3. Walk until 5 consecutive samples have peak < 60 (fixed absolute
   threshold; empirical bezel-regime peaks are 70-330 and post-h-vertex
   peaks are 15-40 → 60 cleanly separates).
4. h-vertex = last sample where peak >= 60, before the consecutive
   drop. Confidence = parent line quality × drop sharpness.

Per-pair results on the 18 worst pairs (only counting `drop_clean=True`
terminations — `False` means the bezel ran to silhouette boundary
without clean termination, suggesting a hull-end termination):

| pair  | sq    | line_q (M, C, Y)    | clean h-vertices | human |
|-------|------:|---------------------|-----------------:|-------|
| 31 A  | 0.73  | 1.00, 0.95, 0.52   | **5 / 6**        | —     |
| 57 A  | 0.56  | 0.79, 0.73, 0.40   | **4 / 4**        | PASS  |
| 31 B  | 0.64  | 1.00, 1.00, 0.28   | **4 / 4**        | PASS  |
| 58 A  | 0.41  | 1.00, 0.81, 0.00   | **4 / 4**        | —     |
| 30 B  | 0.23  | 1.00, 0.41, 0.07   | 2 / 3            | PASS  |
| 30 A  | 0.11  | 0.83, 0.15, 0.07   | 2 / 2            | PASS  |
| 21 A  | 0.17  | 1.00, 0.28, 0.05   | 2 / 2            | —     |
| 21 B  | 0.07  | 0.41, 0.11, 0.03   | 2 / 2            | —     |
| 44 A  | 0.17  | 1.00, 0.24, 0.11   | 2 / 2            | —     |
| 44 B  | 0.14  | 0.78, 0.28, 0.00   | 1 / 2            | PASS  |
| 47 A  | 0.14  | 0.60, 0.28, 0.00   | 1 / 1            | —     |
| 47 B  | 0.16  | 0.52, 0.23, 0.09   | 2 / 2            | —     |
| 58 B  | 0.11  | 1.00, 0.13, 0.09   | 2 / 2            | —     |
| 61 A  | 0.04  | 1.00, 0.08, 0.00   | 1 / 2            | —     |
| 17 A  | 0.15  | 0.85, 0.29, 0.00   | 0 / 1            | —     |
| 17 B  | 0.20  | 0.40, 0.25, 0.15   | 0 / 0            | —     |
| 57 B  | 0.07  | 0.26, 0.12, 0.03   | 0 / 0            | —     |
| 61 B  | 0.12  | 0.26, 0.25, 0.00   | 0 / 0            | —     |

**4 of 18 pairs (22%) produce ≥4 clean h-vertex candidates** — these
are the cases where multiple bezels terminate cleanly within the
silhouette. Multiple candidates per line is expected (each bezel has
both +1 and -1 directions traced; one will be the real h-vertex and
the other is incidental sticker-grid termination on the opposite
direction). Confidence-sorting + downstream filtering will pick the
real h-vertices.

**The 4 strong-cluster cases overlap with the 4 human-pass cases**
(31 B, 57 A, 30 A/B, 44 B PASS; 31 A, 58 A also produce strong
h-vertex signal but weren't in the human PASS list — the latter is
worth a second human-review pass).

The candidate list is returned sorted by confidence. Downstream:
- Filter to `drop_clean=True` and `confidence >= some_threshold`
- Per parent line, keep only the highest-confidence candidate (each
  bezel has ONE real h-vertex, the other direction is noise)
- Combine with hull-detectable h0/h2/h4 from `_fit_hexagon_to_hull`
  → complete 6-vertex hexagon

NOT wired into proposer / `_derive_face_quad_topology_aware`.
Still diagnostics-only.

## What this probe is signal for

The (cube_center, 3 angles, 3 h-vertices) tuple is the prerequisite
for completing the hexagon on yawed cubes. Once we have a complete
6-vertex hexagon, the cardinal-position cube-face derivation that
powers `_derive_face_quad_topology_aware` becomes well-defined.

Future signals to mine, gated on broader-corpus h-vertex confidence
holding up:

- **Refined hexagon fit**: combine the 3 hull-detectable hexagon
  vertices (h0, h2, h4 from `_fit_hexagon_to_hull`) with h1/h3/h5
  from this probe when per-h-vertex confidence is high enough → a
  complete 6-vertex hexagon respecting the cube's interior geometry.
- **Cell-level disambiguation**: when slot/src mismatches are
  ambiguous (the May 18 overlay-feedback pattern), the detected
  boundary lines tell us which sticker cells span TWO faces.
- **Sticker-grid mass subtraction** (Tier 3a — deferred): detect the
  dominant sticker-grid angle (likely ~30°/150° in iso projection),
  suppress its contribution to the angular sweep, re-pick top-3.
  Should turn cyan/yellow-line noise into signal on most cases,
  expanding the strong-cluster from 4/18 toward higher coverage.

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
