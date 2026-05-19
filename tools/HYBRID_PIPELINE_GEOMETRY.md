# Hybrid pipeline geometry — 2026-05-19 session learnings

Single-session arc from 0.6532 → 0.8455 per-sticker assembled accuracy
(+19.2pp) via five merged PRs. This document captures what worked, what
didn't, and what's left, so future agents/sessions can extend without
re-deriving.

## Headline numbers (33-pair corpus, knn5_lab_full classifier)

| Variant | rect | assembled | exact-54 | failed-to-assemble |
|---|---|---|---|---|
| Baseline (hull-guard only) | 0.6532 | 0.6532 | 0 | ? |
| PR #160 (clip-to-hull) | 0.7970 | 0.7970 | 2 | 7 |
| PR #163 (hull-shared + cache) | 0.8381 | 0.8374 | 6 | 1 |
| PR #166 (post-derivation guard) | 0.8438 | 0.8438 | 6 | 1 |
| PR #168 (min-edge calibration) | 0.8455 | 0.8455 | 6 | 1 |
| PR #171 (AMG research tool) | 0.8455 | 0.8455 | 6 | 1 (no change) |
| Non-OOD on current main | **0.8538** | — | — | — |
| Existing recognizer ref | — | 0.827 | — | — |
| Human-quad ceiling | — | 0.9929 | — | — |

**The hybrid pipeline now beats the existing recognizer reference (0.8538 vs 0.827 non-OOD).**

## What worked (shipped)

### 1. Cardinal-position corner extraction (PR #160)

`canonical_corner_order` sorts CW from smallest-CW-from-N angle.
**Which corner ends up at index 0 depends on the quad's specific
geometry — for Set 17 A's U quad, index 0 was the EAST corner (h1);
for a different photo it could be NORTH (h0).** Treating index 0 as
"always h0" gave wrong shared-corner identification across face quads.

Fix: identify corners by image-space cardinal direction (`N=min_y`,
`E=max_x`, `S=max_y`, `W=min_x`). Invariant to canonical_corner_order's
start-index ambiguity.

### 2. Clip-to-hull (PR #160)

Even "trusted" analyze_image grids (fit_error ~ 2.0) had extrapolated
corners 30–60 px past the cube boundary. **Set 17 A's R-slot grid
extrapolated its south corner to (426, 1012) — 100+ px below the
cube.** When propagated through topology derivation, off-cube corners
dragged derived quads off-cube too.

Fix: project out-of-hull corners to the nearest hull edge.

### 3. Full-hull angular lookup (PR #160, extended in PR #163)

The 6-vertex hexagon from `_fit_hexagon_to_hull` is **degenerate on
63% of the corpus** (collapsed adjacent vertices on yawed cubes —
documented in `/tmp/rembg_outlines/summary.txt` survey). The full
51+ vertex hull always has a clean extreme point in each angular
sector.

PR #160 used this only for unique-to-face vertices (h0/h2/h4); PR #163
extended it to shared hexagon vertices (h1/h3/h5) when the trusted
neighbor wasn't available. PR #163 added angles for h1 (-π/3 NE),
h3 (π/2 S), h5 (-2π/3 NW).

### 4. Parallelogram derivation (PR #160)

For 3 visible iso-projected cube faces, the diagonals-bisect identity
`A + C = B + D` gives the 4th corner exactly when the other 3 are
known precisely. Used for unique-to-face vertices when 2 neighbor
quads are trustworthy.

### 5. Content-derived cache key (PR #163)

**Pre-fix:** `_rembg_cube_hull` cache used `id(processing_image)`.
Python reuses ids of garbage-collected objects, so in sweep mode (33
pairs sequentially) freed image objects' ids got reassigned to new
images → false cache hits with stale hulls.

**Manifested as:** Set 17 A single-pair gave 27/27 stickers correct
on side A, but the aggregate sweep gave 14/27 because earlier pairs'
hulls leaked into Set 17's lookup.

**Fix:** content-derived key (image size + 32×32 thumbnail bytes).
Structurally impossible for distinct images to collide.

### 6. Post-derivation hull guard (PR #166)

When the topology fallback has only 1 trusted neighbor AND the rembg
hexagon is degenerate, the derived quad can collapse (corners cluster
on top of each other → tiny area, min_edge ≈ 0). Rectifying through
such a degenerate quad produces "beige noise" sampling background.

**Two checks** (both must pass to keep the derivation):
- `min_edge >= MIN_EDGE_PX` — collapsed quads have min_edge < 10 px
- `>= 5 of 9 sampling centroids inside the rembg cube hull` — catches
  off-cube derivations that aren't strictly degenerate

If either fails, KEEP the original analyze_image grid quad. Makes the
fallback a strict non-regression.

### 7. Min-edge calibration (PR #168)

PR #166's `MIN_EDGE_PX = 30` was conservative. 198-quad corpus survey
showed:
- p5 = 135 px (working face floor)
- p25 = 273 px
- median = 312 px (working faces)
- median (1-4/9 correct faces) = 160-286 px

Bumping to 100 caught Set 22 A's U slot (min_edge=53) and similar
borderline-narrow degenerate cases without false-positives on
working faces.

## What didn't work (negative results)

### Aspect-only multi-face-span rejection (NOT shipped)

Tried: reject grids with `aspect > 1.5` as likely multi-face spans.
**Result: -22.7pp catastrophic regression.** Many legitimate yawed-cube
face grids have aspect 1.5-2.0; the aspect-only threshold over-rejected
real grids alongside the multi-face spans.

### Combined aspect + bbox-vs-hull soft penalty (NOT shipped)

Combined signal with soft penalty multipliers. **Result: -15.5pp
regression.** The penalty over-fires on legitimate borderline grids,
forcing them into topology-fallback derivations that are LESS accurate
than even an "OK" original grid.

**Takeaway:** heuristic thresholds on bbox shape aren't discriminating
enough. The signal-to-noise on aspect/bbox alone is too low. A better
discriminating signal (color-discontinuity within cells, cell-spacing
variance) might work but wasn't pursued.

### Patch-fraction 0.40 → 0.55 (NOT shipped)

Tried: bigger sample patch in `extract_stickers_from_rectified` to be
robust to small sample drift. **Result: 0.8381 → 0.8376, within noise.**
Some pairs gain, others lose; net flat. The "drift-to-bezel"
hypothesis isn't broadly applicable.

### SAM2 per-face point prompting (NOT shipped, invalidated)

Tried: prompt SAM2 with 9 per-face sticker centroids to get per-face
masks. **Result: FAILED — SAM2 treats the cube as one object**, not 3
faces. Negative prompts on adjacent faces collapsed masks to ~0.

### SAM2 AMG-as-replacement (NOT shipped, replaced by refinement)

Tried: AMG → 27 sticker masks → k-means → 4-corner extraction +
extrapolation → new face quads from scratch. **Result: regressed
both hard and easy pairs.** The convex-hull-simplification picked
wrong 4 corners on noisy masks; PCA-bucketing failed on non-orthogonal
3×3 grids.

### SAM2 AMG-as-refinement (shipped as research tool only, PR #171)

Final AMG variant: keep PR #168 quads, snap corners to nearby AMG
sticker centroids (within 25 px). Gate by `topologyFallbackRejectedByHullGuard=True`.

**Result: +0.06pp aggregate.** Concentrated per-pair wins (Set 44 +7.4,
Set 17 +3.7, Set 31 +1.9) cancel with smaller losses (Set 12, 61 each
-1.9). Real wins, but not enough to justify the 150 MB SAM2 dependency
+ scipy monkey-patch + 25-30 min sweep cost.

**Shipped as `tools/amg_face_refiner.py` research tool only.** Not
wired into evaluator. Available for future revisit when SAM2 ships
pre-built C-extension wheels or when a real-world Set-44-style
failure surfaces.

## Things explored + verified-not-bias

### Side B vs side A bias (HYPOTHESIS WRONG)

Visual review of overlays suggested side B was systematically harder.
**Aggregate data showed side B is actually 0.8777 vs side A 0.8095
(+6.8pp BETTER on B).** No anti-B bias in the code; the perception
came from non-representative failing pairs (Set 17, Set 61 where B
happened to be the bad side).

### Color-discontinuity / cell-variance signals (NOT YET TRIED)

Multi-face-span detection from color-gradient WITHIN 3×3 cells.
A grid cell crossing a face boundary shows a strong color delta
inside the sticker region; single-face cells are visually uniform.
This is the next discriminating signal worth trying when threshold
heuristics fail. Not implemented this session.

## The remaining ceiling (per-pair-specific)

After PR #168, the 7-10 worst pairs have architectural failure modes
that threshold-tuning can't fix:

### Set 31 (both sides ~60%)
- Side A: U fallback kept, R clean, F fallback REVERTED → bad grid kept
- Side B: D fallback kept, L clean, B fallback REVERTED → bad grid kept
- Pattern: 1 clean grid + 2 catastrophic grids per side, neither
  topology nor original survives. Cube has unusual yaw making
  analyze_image's grid detection fundamentally struggle.

### Set 47 A (worst non-OOD pair)
- U fallback reverted to (degenerate) original
- R is multi-face-span with fit_error 0.85 (low residual, but spans 2+ faces)
- F fallback reverted to (degenerate) original
- AMG refinement gives +7.4pp here (the canonical case), but kept
  as research tool only.

### Set 17 B's L/B label swap (joint-ID issue)
- Geometry is fine, but joint face-ID assigns L→B and B→L
- Color classifier confusion at corner stickers? Yaw-config issue?
- Not debugged.

## Architecture-level next moves (when ready)

1. **Replace `_fit_hexagon_to_hull`**: 63% degenerate hexagons is the
   systemic bottleneck. Options: Douglas-Peucker on hull, RANSAC line
   fitting, learned vertex detector.

2. **Color-discontinuity multi-face-span detector**: the unexplored
   high-value signal — cell-internal color variance.

3. **Joint face-ID tiebreaker improvements**: corner-sticker-weighted
   multiset matching (currently uniform per-sticker).

4. **AMG re-evaluation**: when SAM2 ships pre-compiled wheels with the
   C++ extension, AMG cost drops 5-10×, may flip the cost/benefit.

5. **Step back to product-readiness**: 0.8455 already beats
   WhiteUpRecognizer reference. Is this good enough to integrate into
   cube-snap, with user-reported failures as the next data signal?

## Reusable session insights

- **`canonical_corner_order` is not start-anchored**: any code that
  treats `quad[0]` as "the top-left" or similar will break on yawed
  cubes. Use cardinal-position lookup (`_cardinal_corners`).

- **`id()` is not a stable cache key**: Python reuses ids of
  garbage-collected objects. Content-derived keys are required for
  any cache that survives multiple image loads in one process.

- **Test in sweep mode AND single-pair mode**: the Set 17 cache bug
  was invisible in single-pair tests (27/27 correct) but produced
  14/27 in the aggregate sweep. Cache state leaks across iterations
  even when the visible state seems contained.

- **5 perfect-pair preservation is a clean signal**: if a change to
  the proposer keeps the 5 perfect baseline pairs (15, 28, 29, 41, 42)
  at 1.0000, it's safe to ship. If it regresses any of them, it's
  introducing geometric noise.

- **Aggregate aggregate small movements (≤0.1pp) are noise**: at the
  level we're at (0.8455), individual sticker classifications can flip
  due to color-classifier uncertainty even when geometry doesn't change.
  Trust per-pair deltas more than aggregate at this scale.

- **The "view both before comparing" rule** (from CLAUDE.md) saved
  multiple false claims this session. Re-rendering overlays vs
  reasoning from memory is essential.

## Files / PRs

| PR | Topic |
|---|---|
| #160 | Cardinal + clip-to-hull + full-hull (unique vertices) |
| #163 | Full-hull for shared vertices + cache key fix |
| #166 | Post-derivation sanity guard |
| #168 | Min-edge threshold calibration |
| #171 | AMG face refiner (research tool only) |

| File | Purpose |
|---|---|
| `tools/evaluate_hybrid_pipeline.py` | Hybrid pipeline evaluator (geometry proposer + topology fallback) |
| `tools/amg_face_refiner.py` | AMG research tool (NOT wired into pipeline) |
| `tools/AMG_FACE_REFINER.md` | AMG research findings doc |
| `tools/HYBRID_PIPELINE_GEOMETRY.md` | This doc |
