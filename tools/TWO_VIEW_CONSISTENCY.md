# Two-view orientation consistency — design + integration plan

## Status

**Math primitive shipped; matrix integration is a follow-up that must
solve A/B axis canonicalization first.**

This PR ships `tools/two_view_consistency.py` — a standalone math
primitive for computing how much A and B view fits *disagree* about a
cube's 3D orientation under the documented capture rotation.

It is explicitly **not** ready to be wired into the Phase 2B matrix
"as is" by simply feeding raw per-image `GlobalCubeModel.axis_x_2d /
axis_y_2d / axis_z_2d` vectors. See "Input precondition" below.

## Why this signal

Phase 4 v1.1 (ctvd PR #244) fit 4 classifier classes on the 6 features
of `phase2b_recomputed_signals.json` against the 70-case corpus. Best
result: `random_forest` at 80% catastrophic recall / 36% GOOD FPR —
above the Phase 2 ≤10% FPR bar. The data lever (expanding 58→70
cases) did not move the needle (v1's 18% FPR was overfit to the
58-case sample; on the larger corpus it generalized to 36%).

The remaining lever is a stronger feature, and the natural candidate
is a pair-level geometric signal: do the A and B fits agree about
the cube's 3D pose under the documented capture rotation?

The existing 6 features are all per-view; none catch the case where
A's and B's fits *individually* look plausible but together imply
geometrically incompatible cube orientations.

Codex's PR #200 shipped a *sticker-spacing* two-view consistency
signal in `rubik_recognizer/recognizer.py` (different surface — feeds
the production recognizer, not the research matrix). This module's
*orientation* consistency is complementary: same idea (compare A vs
B), different geometric quantity (rotation matrix vs spacing).

## The capture convention

Between photo A (white-up, URF visible) and photo B (yellow-up, DLB
visible), the user flips the cube end-over-end by gripping the R
and L sides and rotating **180° around the image-horizontal axis
(camera X)**. There is no additional rotation. The 54-char URFDLB
ordering in `rubik_recognizer/recognizer.py` assumes only this
single flip.

## The math (summary)

Each global-model fit gives 3 image-space axis projection vectors
(ax_2d, ay_2d, az_2d). Under weak-perspective projection, the 2×3
matrix M = [ax | ay | az] equals s · R[:2, :3] where R is the 3×3
rotation matrix from cube-body axes to camera frame.

Recovery (`recover_rotation_from_axes`):
- s ≈ ||M||_F / √2
- R[:2, :3] = M / s, orthonormalized via Gram-Schmidt
- R[2, :3] = R[0] × R[1] (cross product, sign-fixed for right-handed)

Consistency (`two_view_consistency_deg`):
- R_FLIP = 180° around camera X = diag(1, −1, −1)
- residual = (R_FLIP · R_A) · R_B⁻¹
- return rotation_angle_deg(residual) in [0, 180]

Left-multiply (R_B = R_FLIP · R_A) because the cube rotates in the
fixed world/camera frame, not its body frame.

See `tools/two_view_consistency.py` module docstring for the full
derivation including v1 history (R_Y(180°) was wrong) and the
single-canonical-axis rationale.

## Input precondition (READ BEFORE INTEGRATING)

`two_view_consistency_deg(axes_A, axes_B)` assumes its two inputs
are expressed in the **same semantic cube-body axis frame** — i.e.,
the body X axis in A means the same physical cube axis as the body
X axis in B. The math fails silently otherwise: a systematic axis
relabel between A and B produces a residual rotation that does NOT
match R_FLIP, even on perfectly fit pairs.

**This is not automatically true** for the raw axes the recognizer
writes onto `GlobalCubeModel`. Empirically (Codex audit on PR #245):
applying `two_view_consistency_deg` directly to 35 human-labeled
A/B axis pairs gave **median 173° residual and 0/35 pairs under
25°**. That is the signature of an A↔B axis-frame mismatch, not a
math bug: every pair, including known-GOOD pairs, scores like a
catastrophic mis-fit.

In photo A (URF visible) the recognizer sees +U, +R, +F faces; in
photo B (DLB visible) it sees −U (i.e., +D), −R (i.e., +L), −F
(i.e., +B) faces. Whether the recognizer's `axis_x_2d / axis_y_2d /
axis_z_2d` are tied to "visible-face vertices" (in which case A and
B point opposite ways in the body frame) or to "canonical cube
body axes" (in which case they are consistent) determines whether
the raw vectors can be fed in directly.

The empirical 173° median is strong evidence that they are **not**
consistent without a canonicalization step.

## Integration plan (follow-up PR)

Step 0 — **establish axis canonicalization before anything else.**
This is the load-bearing step the math primitive cannot bypass.

  - Audit `tools/global_cube_model.py` and `rubik_recognizer/` to
    determine exactly what `axis_x_2d / axis_y_2d / axis_z_2d` mean
    semantically (visible-vertex vs body-axis) and whether the
    meaning differs between A-side and B-side fits.
  - Derive a canonicalization mapping: how to transform raw B
    axes into the same body frame as raw A axes. Likely something
    like (ax_B, ay_B, az_B) → (−ax_B, −ay_B, −az_B) if the
    "visible-vertex" hypothesis is correct, but VERIFY empirically.
  - Validate the mapping: re-run the metric on the 35 human-
    labeled axis pairs in `tests/fixtures/gcm_axis_ground_truth.json`
    and confirm GOOD-pair median collapses from ~173° to ≤25°.
    If it doesn't, the canonicalization hypothesis is wrong and
    needs more investigation before integration.

Step 1 — capture axes per run. Modify `tools/phase2b_recompute.py`:
after `model = fit_global_cube_model(...)`, append the raw axes
to the run record (the canonicalization step lives in the metric-
computation pass, not the capture, so we keep the raw record).

Step 2 — compute the per-pair metric. Add a post-processing pass
to `recompute_all()` that, for each set, applies the Step 0
canonicalization to the B-side axes and then calls
`two_view_consistency_deg(axes_A, canonicalize(axes_B))`. Inject
the value into both A's and B's matrix rows.

Step 3 — re-run `tools/phase2b_recompute.py` to regenerate
`tests/fixtures/phase2b_recomputed_signals.json` with the 7th
feature populated. Cost: rembg per photo × 70 photos × 2 runs ≈
30-60 min wall time.

Step 4 — update `tools/phase4_trust_ranker.py` `FEATURE_COLUMNS`
to include `two_view_orientation_consistency_deg` and re-run.
Compare v2 results to v1 — does the bar clear with 7 features?

Splitting math vs integration into separate PRs keeps the math
reviewable without blocking on the slow re-run, and isolates the
canonicalization investigation from the math.

## Lane

`tools/two_view_consistency.py` is research scaffolding — Claude's
lane. The integration (steps 0–4) is also Claude's lane. The new
column added in step 3 lives in the existing Shared fixture
`tests/fixtures/phase2b_recomputed_signals.json` (append-only
spirit — adding a column, not mutating existing data).

## Limitations

- Assumes weak-perspective projection. Real iPhone photos have mild
  perspective; treat values as approximate.
- Assumes the user kept the camera approximately stationary between
  photos. Real capture varies; expect a baseline noise floor of
  ~5-15° even on correct fits.
- The "expected 180° rotation around camera X" matches the
  documented cube-snap capture flow. If a user rotates around a
  different axis, this metric will report a large residual — but
  that's the desired behavior, since the recognizer's URFDLB
  ordering also assumes the canonical flip.
- The math assumes its inputs are semantically aligned (Input
  precondition section). Without that, the metric is meaningless.

## See also

- `tools/PHASE_4_TRUST_RANKER_V1.md` — the v1 result this signal
  is designed to improve on as v2.
- `tools/global_cube_model.py` — source of the axis projection
  vectors this module consumes.
- `rubik_recognizer/recognizer.py` (Codex PR #200) — the
  sticker-spacing two-view consistency signal, complementary to
  this orientation-based one.
- `tests/fixtures/gcm_axis_ground_truth.json` — 35 human-labeled
  A/B axis pairs. The natural validation set for the
  canonicalization step in the follow-up PR.
- `tools/STATE_OF_THE_WORLD.md` — Phase 4 positioning.
