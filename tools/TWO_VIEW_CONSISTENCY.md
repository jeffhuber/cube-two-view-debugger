# Two-view orientation consistency — design + integration plan

## Status

**Math module shipped; matrix integration is a follow-up.**

This PR ships `tools/two_view_consistency.py` — the standalone math
for computing how much A and B view fits *disagree* about a cube's
3D orientation. Integration into the Phase 2B matrix regeneration
pipeline (which would write `two_view_orientation_consistency_deg`
into `tests/fixtures/phase2b_recomputed_signals.json` as a 7th
feature for the Phase 4 trust ranker) is left as a separate PR
because it triggers a full rembg re-run on all 58 cases (~30-60 min).

The math module + tests can be reviewed and merged independently of
the slow re-run.

## Why this signal

Phase 4 v1 (ctvd PR #242) fit 4 classifier classes on the 6 features
of `phase2b_recomputed_signals.json`. Best result: `random_forest` at
80% catastrophic recall / 18% GOOD FPR — meaningful progress over the
Phase 2B hand-tuned ceiling (80% / 31%) but still above the 10% FPR
bar.

`PHASE_4_TRUST_RANKER_V1.md` identifies two complementary levers:
1. **More data** (Phase 4 corpus expansion — blocked on gallery bug).
2. **More features** — specifically, a signal derived from comparing
   the A and B view fits per pair. The existing 6 features are all
   per-view; none catch the case where A's and B's fits *individually*
   look plausible but together imply geometrically incompatible cube
   orientations.

Codex's PR #200 shipped a *sticker-spacing* two-view consistency
signal in `rubik_recognizer/recognizer.py` (different surface — feeds
the production recognizer, not the research matrix). This PR's
*orientation* consistency is complementary: same idea (compare A vs
B), different geometric quantity (rotation matrix vs spacing).

## The math (summary)

Each global-model fit gives 3 image-space axis projection vectors
(ax_2d, ay_2d, az_2d). Under weak-perspective projection, the 2×3
matrix M = [ax | ay | az] equals s · R[:2, :3] where R is the 3×3
rotation matrix from cube-body axes to camera frame.

Recovery (in `recover_rotation_from_axes`):
- s ≈ ||M||_F / √2
- R[:2, :3] = M / s, orthonormalized
- R[2, :3] = R[0] × R[1] (cross product, sign-fixed for right-handed)

Consistency (in `two_view_consistency_deg`):
- Apply expected world-Y 180° rotation between A and B.
- Compute the residual rotation angle (degrees).
- Return min over (Y180, Y180.T) so a sign-convention disagreement
  doesn't double-penalize.

See `tools/two_view_consistency.py` module docstring for the
detailed derivation.

## Integration plan (follow-up PR)

Once this math module is merged, the integration steps are:

1. **Modify `tools/phase2b_recompute.py`** to capture the 3 axis
   projection vectors per run (currently stored on the
   `GlobalCubeModel` instance but not extracted into the matrix).
   Roughly 10 LOC: after `model = fit_global_cube_model(...)`,
   append `axes = (model.axis_x_2d, model.axis_y_2d, model.axis_z_2d)`
   to the run record.

2. **Add a post-processing step** to `recompute_all()` (also in
   `tools/phase2b_recompute.py`) that, after the per-case loop,
   pairs each set's A and B runs and computes
   `two_view_consistency_deg(axes_A_run0, axes_B_run0)` per pair.
   Inject the value into both A's and B's matrix rows.

3. **Re-run `tools/phase2b_recompute.py`** to regenerate
   `tests/fixtures/phase2b_recomputed_signals.json` with the 7th
   feature populated.

4. **Update `tools/phase4_trust_ranker.py`** `FEATURE_COLUMNS` to
   include `two_view_orientation_consistency_deg` and re-run.
   Compare v2 results to v1 — does the bar clear with 7 features?

The re-run is expensive (rembg per photo × 58 photos × 2 runs ≈
30-60 min wall time). Splitting math vs integration into separate
PRs keeps the math reviewable without blocking on the re-run.

## Lane

`tools/two_view_consistency.py` is research scaffolding — Claude's
lane. The integration (step 1+2) touches `tools/phase2b_recompute.py`
which is also Claude's. The new fixture written in step 3 is the
existing `tests/fixtures/phase2b_recomputed_signals.json` (Shared,
append-only spirit — adding a column, not mutating existing data).

## Limitations

- Assumes weak-perspective projection. Real iPhone photos have mild
  perspective; expect a baseline noise floor of ~5-15° on correct
  fits.
- Assumes the user kept the camera approximately stationary between
  photos. Real capture varies; this is an empirical baseline to
  characterize during the follow-up integration.
- The "expected 180° rotation around world Y" matches the documented
  cube-snap capture flow (white-up → yellow-up means rotating around
  the vertical axis). The function returns the min over (Y180, Y180.T)
  to absorb sign-convention differences. If a user rotates around a
  different axis, this metric will misreport — but that's a capture-
  flow violation, not a measurement bug.

## See also

- `tools/PHASE_4_TRUST_RANKER_V1.md` — the v1 result this signal is
  designed to improve on as v2.
- `tools/global_cube_model.py` — source of the axis projection
  vectors this module consumes.
- `rubik_recognizer/recognizer.py` (Codex PR #200) — the
  sticker-spacing two-view consistency signal, complementary to this
  orientation-based one.
- `tools/STATE_OF_THE_WORLD.md` — Phase 4 positioning.
