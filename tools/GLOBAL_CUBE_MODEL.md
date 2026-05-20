# Global cube projection model — diagnostics

Diagnostics-only probe that fits a single **6-DOF projected cube
model** per photo so that all 27 sticker quads come from one coherent
geometry. This is the post-pivot Tier 2 work (per Decision Log
entry **2026-05-20** in `COORDINATION.md`).

## Motivation

Three independent first-principles design proposals (Claude, Codex,
Devin) converged on the same architecture:

> Replace per-sticker / per-grid local detection + post-hoc
> reconciliation with a SINGLE GLOBAL CUBE PROJECTION MODEL per photo
> whose 27 sticker quads are coherent **by construction**.

Per Codex: *"Every sampled sticker must come from the same valid
projected cube model."* The structural failure mode of "local
detection then reconcile" — exactly what's been biting #175's per-cell
discontinuity probe and #177-#180's bezel approach — disappears when
the parameterization itself guarantees coherence.

## Parameterization (6 DOF)

```
cube_center: (cx, cy)              # 2 DOF — front cube corner
axis_angles_rad: (θ_0, θ_1, θ_2)   # 3 DOF — 3 axes from cube_center
edge_length_px: L                  # 1 DOF — shared cube edge length
```

From these 6 numbers, the geometry is fully determined:

- **7 visible cube corners**: the front corner (cube_center) + the 6
  hexagon silhouette vertices h0-h5. Computed via:
  ```
  h-vertex (axis i alone)  = cube_center + L * (cos θ_i, sin θ_i)
  h-vertex (axes i + j)    = cube_center + L * (cos θ_i, sin θ_i) + L * (cos θ_j, sin θ_j)
  ```
- **3 face quads**: each face is a parallelogram containing 4 corners
  (front + 2 single-axis h-vertices + their sum).
- **27 sticker cells**: bilinear 3×3 subdivision of each face quad.

## Algorithm

```
1. Initialize from bezel detection (#178) outputs:
   - cube_center: direct from detection
   - 3 axis angles: try ALL 8 sign combinations of detection.boundary_angles
     (the bezel detector returns line angles in [0, π); the model needs
     outward axis angles in [0, 2π) — only one sign combination produces
     3 axes that span 360° once); pick the combination with highest
     initial silhouette IoU
   - edge_length: ~0.32 × silhouette max extent (empirical underestimate;
     optimizer grows it)

2. Optimize the 6-parameter vector via Nelder-Mead (scipy.optimize.minimize):
   - Objective: weighted sum of
     · (1 - silhouette IoU) — predicted hexagon vs rembg mask
     · (1 - bezel angle match) — how well model axes match detected bezels
   - Weights default: silhouette=1.0, bezel=0.5
   - Convergence: xatol=1px, fatol=0.001, maxiter=300

3. Derive geometry from fitted parameters. Output:
   - cube_center, axis angles (deg), edge_length
   - 7 visible corners
   - 3 face quads
   - 27 sticker cells
   - fit_quality = 0.7 * IoU + 0.3 * bezel_match
```

## Per-pair results (18 worst pairs from PR #176)

| pair  | bezel_sq | fit_quality | IoU    | bezel_match | edge_px | shift  | iter |
|-------|---------:|------------:|-------:|------------:|--------:|-------:|-----:|
| 17 A  | 0.15     | **0.91**    | 0.873  | 0.987       | 1140    | 122    | 143  |
| 17 B  | 0.20     | **0.89**    | 0.846  | 0.984       | 1137    | 201    | 214  |
| 21 A  | 0.17     | **0.91**    | 0.869  | 0.998       | 1153    | 48     | 198  |
| 21 B  | 0.07     | **0.89**    | 0.859  | 0.950       | 1172    | 53     | 166  |
| 30 A  | 0.11     | **0.92**    | 0.885  | 0.990       | 912     | 22     | 200  |
| 30 B  | 0.23     | **0.93**    | 0.900  | 0.994       | 933     | 148    | 184  |
| 31 A  | 0.73     | **0.92**    | 0.888  | 1.000       | 954     | 14     | 197  |
| 31 B  | 0.64     | **0.92**    | 0.888  | 1.000       | 991     | 98     | 195  |
| 44 A  | 0.17     | **0.90**    | 0.862  | 1.000       | 1242    | 119    | 160  |
| 44 B  | 0.14     | **0.91**    | 0.867  | 1.000       | 1212    | 25     | 191  |
| 47 A  | 0.14     | **0.91**    | 0.874  | 0.996       | 1077    | 96     | 161  |
| 47 B  | 0.16     | **0.91**    | 0.873  | 0.998       | 1069    | 98     | 209  |
| 57 A  | 0.56     | **0.92**    | 0.880  | 1.000       | 1091    | 33     | 171  |
| 57 B  | 0.07     | 0.81        | 0.787  | 0.849       | 1074    | 150    | 178  |
| 58 A  | 0.41     | **0.92**    | 0.884  | 0.992       | 1072    | 30     | 182  |
| 58 B  | 0.11     | **0.91**    | 0.878  | 0.996       | 1070    | 85     | 193  |
| 61 A  | 0.04     | **0.93**    | 0.909  | 0.962       | 1163    | 142    | 214  |
| 61 B  | 0.12     | **0.93**    | 0.902  | 0.992       | 1104    | 119    | 204  |

**Headline:** **17 of 18 pairs (94%) at fit_quality ≥ 0.85.**
**18 of 18 (100%) at fit_quality ≥ 0.70.**

Compare to prior approaches on the same 18-pair corpus:
- `_fit_hexagon_to_hull` (original): 12/18 degenerate (min_edge < 20 px)
- PR #177 (single-pass bezel detection): 9/18 "strong" by overconfident scorer
- PR #178 (iterative bezel + per-line quality): 4/18 strong by recalibrated scorer
- PR #180 (h-vertex tracing): 4/18 strong cluster (subset of #178's)
- **This PR**: **17/18 strong** by silhouette-IoU + bezel-match scorer

## Even where bezel signal is weak, the model converges

Note Set 61 A: bezel_sq=0.04 (the worst bezel signal in the corpus —
only the magenta line was high-quality). The global model fit STILL
achieves fit_quality=0.93 because silhouette IoU drives optimization
when bezel information is sparse. The model's parameterization (3
axes from one center, fixed edge length) is so constraining that
even bad initial angles can be corrected by the silhouette objective.

This is the key structural win the pivot was after: the fit is robust
to weak inputs because the parameter space is small and physically
meaningful.

## The one non-strong case (Set 57 B, 0.81)

Set 57 B has bezel_sq=0.07 (very weak bezel signal) AND the
silhouette is noisier (background interference). The optimizer
converged to a hexagon slightly larger than the cube. The model is
visibly off but in a clearly-diagnosable way: this is the kind of
case where the upstream silhouette has issues that propagate.

For production, this is what the refusal-to-solve / retake-prompt
path is for. fit_quality < 0.85 → flag for manual review or retake.

## Visible cube corners (h-vertex termini, for free)

Because the model is parameterized at the corner level, the 3
interior hexagon vertices (h1, h3, h5 — what PR #180 was trying to
trace) are **direct outputs** of the model: `visible_corners["011"]`,
`["101"]`, `["110"]`. No separate trace step needed.

Similarly h0/h2/h4 (the 3 hull vertices) are `visible_corners["001"]`,
`["010"]`, `["100"]`. The model produces a complete 6-vertex hexagon
on EVERY pair, even ones the hull-based fitter rejected as degenerate.

## What this PR does NOT do

- **NOT wired into recognizer behavior.** Diagnostics-only.
- **NOT integrated with `_derive_face_quad_topology_aware`.** That's
  the next step, gated on validation against ground-truth geometry
  labels (the "human geometry overlay" investment from Devin's
  first-principles design).
- **NOT a face-identity resolver.** The model outputs 3 face quads
  labeled `face_01`, `face_12`, `face_02` (per the axes they
  contain). Mapping to U/F/R/D/L/B requires center-sticker color
  reading + the capture convention (white-up A, yellow-up B).
  That's a separate step.
- **NOT a color classifier.** Per-photo calibration + probabilistic
  color reads are the next module to build, separately.

## Dependencies

| Dependency | Status |
|---|---|
| numpy | required |
| Pillow | required (for silhouette IoU rasterization) |
| scipy | optional — `scipy.optimize.minimize` for the Nelder-Mead fit, AND `scipy.ndimage` via the bezel detection module's helpers. When missing, the fitter returns an initialization-only model with `debug["error"]` rather than raising. |

## Files

| File | Purpose |
|---|---|
| `tools/global_cube_model.py` | Module: 6-DOF parameterization, geometry derivation, silhouette IoU, bezel match score, fitter |
| `tools/test_global_cube_model.py` | Test driver + visualization (uses rembg for mask, fits + renders overlay) |
| `tools/GLOBAL_CUBE_MODEL.md` | This doc |
| `tests/test_global_cube_model.py` | Unit tests (4 passing: import, geometry derivation, scipy-missing fallback, end-to-end synthetic) |

## Reproducing this artifact

```bash
.venv/bin/pip install scipy rembg

.venv/bin/python tools/test_global_cube_model.py \
    --sets 17 21 30 31 44 47 57 58 61 \
    --out /tmp/gcm_results
```

Outputs per pair:
- `set_<N>_<side>_overlay.png` — 3-panel side-by-side (photo / rembg
  mask / photo with fitted model overlay: 3 face quads + 27 sticker
  cells + cube_center + hexagon vertices)
- `set_<N>_<side>_data.json` — fitted model parameters + derived
  geometry + fit quality + optimizer telemetry

PNGs are not committed (large binaries); the generator + this doc
are the durable artifacts.

## What to validate next

This PR is a prototype establishing the global model approach works.
The natural follow-ups, in order:

1. **Ground-truth geometry labels** (per Devin: "human marks true
   face quads on bad cases"). Validate that the 17/18 fits at
   fit_quality ≥ 0.85 actually have face quads that match human-labeled
   geometry. The current scoring is silhouette + bezel; human review
   tests whether the fits are visually correct on a finer scale.
2. **Broader corpus run**. The 18-pair test set is biased toward
   hard cases. Run on the full corpus_manifest.json (15 additional
   pairs) to verify the easy cases stay easy.
3. **Two-view consistency**: if photos A and B were taken of the
   same cube with the standard flip, the two models should produce
   GEOMETRICALLY CONSISTENT outputs (same cube edge length, same yaw
   roughly, etc.). Useful cross-check.
4. **Face-identity resolution**: combine model output with center
   sticker color reads to assign U/F/R/D/L/B labels.
5. **Behavior wiring**: only after #4, the model can replace
   `_fit_hexagon_to_hull` in `_derive_face_quad_topology_aware`. With
   confidence gating: use model when fit_quality ≥ threshold; fall
   back to current path otherwise.
