# A/B axis canonicalization — investigation + validation

**Status: Step 0 of the Phase 4 v2 integration plan is complete and
validated. Ready for the matrix-recompute integration PR.**

## The problem (recap from PR #245 audit)

`tools/two_view_consistency.py:two_view_consistency_deg` assumes its
two axis-tuple inputs are expressed in the same semantic cube-body
axis frame. Codex's audit of PR #245 measured the actual behavior
when raw `GlobalCubeModel.axis_x_2d / axis_y_2d / axis_z_2d` are
fed directly into the metric on the 35 human-labeled GOOD A/B
pairs in `tests/fixtures/gcm_axis_ground_truth.json`:

  | Approach                              | median   | under 25° | max     |
  |---------------------------------------|----------|-----------|---------|
  | Raw axes (no canonicalization)        | 173.34°  | 0/35      | 179.52° |

Every pair, including known-good ones, scores like a catastrophic
mis-fit. That is the signature of an A↔B body-frame axis-labeling
mismatch, not a math bug.

## The solution: min over the 48 signed axis permutations

The recognizer's per-image fit assigns `axis_x_2d / axis_y_2d /
axis_z_2d` based on a correspondence search whose A↔B labeling
can differ by any signed permutation of the 3 axes — 48
combinations total: 6 permutations × 8 sign-flip patterns.

24 of those 48 are "real" cube rotations (det=+1, elements of the
chiral octahedral group); the other 24 (det=-1) are reflections,
which aren't physical cube symmetries but DO show up empirically as
the best A↔B alignment for ~one-third of human-labeled GOOD pairs.
The recognizer's per-image chirality detection (PR #218/#220)
corrects most chirality errors but residual A↔B left-handed-vs-
right-handed labeling differences still occur.

```
canonicalized_two_view_consistency_deg(axes_A, axes_B_raw) =
    min over the 48 signed axis permutations T of
        two_view_consistency_deg(axes_A, T · axes_B_raw)
```

`tools/two_view_canonicalization.py` implements this with the
48-element `ALL_AXIS_TRANSFORMS` tuple precomputed at import time
(the 24-element `CUBE_ROTATIONS` subset is kept for diagnostic
comparison but not used in the metric).

## Iteration history

The path to 48 was empirical, not theoretical:

1. **First pass** restricted the search to the 24 det=+1 cube
   rotations on theoretical grounds (those are the "real" cube
   symmetries). Result: 23/35 under 25°, with 12 GOOD pairs at
   34-61° as an apparent "noise floor."

2. **Codex P2 on PR #246** observed empirically that those 12
   pairs (sets 12, 20, 24, 27, 30, 36, 37, 38, 39, 40, 43, 44, 46)
   align best under excluded det=-1 mappings — e.g. even-perm
   with all 3 signs flipped, which is a reflection but is the
   physically right answer for these labelings.

3. **Second pass** searches all 48. Result: 35/35 under 25°,
   max 23.41°, median 10.66°.

The lesson: the validation contract ("GOOD-pair median ≤25°")
isn't enough by itself — also need "ALL GOOD pairs under 25°".

## Validation results

Validation set: `tests/fixtures/gcm_axis_ground_truth.json`,
filtered to pairs where both `_A` and `_B` carry `approved=true`.
N=35.

  | Approach                              | median   | under 25° | max     |
  |---------------------------------------|----------|-----------|---------|
  | Raw axes (no canonicalization)        | 173.34°  |  0/35     | 179.52° |
  | All-signs-flipped only (3 perms)      | 121.33°  | 13/35     | 178.02° |
  | det=+1 only (24 cube rotations)       |  15.56°  | 23/35     |  61.08° |
  | All 48 signed axis permutations       |  10.66°  | 35/35     |  23.41° |

The 48-transform canonicalization collapses the GOOD-pair median
from 173° → 10.66° and brings all 35 pairs under the 25° target.

## Discrimination check — synthetic catastrophic perturbations

Take each GOOD pair, perturb A's axes by a 3D rotation
(simulating a recognizer that got A's yaw/pitch/roll wrong), and
re-measure the canonicalized residual (with the 48-transform search):

  | Perturbation | median | min | under 25° |
  |--------------|--------|-----|-----------|
  | yaw 15°      | 20.63° | 11° | 25/35     |
  | yaw 30°      | 35.45° | 20° |  7/35     |
  | yaw 60°      | 27.31° | 18° | 12/35     |
  | yaw 90°      | 10.66° |  4° | 35/35 ⚠️  |
  | pitch 30°    | 29.38° | 20° |  6/35     |
  | pitch 60°    | 32.10° | 17° |  7/35     |
  | roll 30°     | 33.24° | 18° |  5/35     |
  | roll 60°     | 32.44° | 18° |  8/35     |

### Threshold sweep — GOOD vs yaw-30° perturbed

  | threshold | GOOD < thr | yaw-30° BAD < thr |
  |-----------|------------|-------------------|
  |    10°    |   45.7%    |    0.0%           |
  |    15°    |   68.6%    |    0.0%           |
  |    20°    |   88.6%    |    0.0%           |
  |    25°    |  100.0%    |   20.0%           |
  |    30°    |  100.0%    |   34.3%           |
  |    40°    |  100.0%    |   74.3%           |

Strong discrimination in the [15°, 25°] range. At threshold = 20°:
**88.6% GOOD recall, 0% catastrophic acceptance** — well above the
Phase 2 target (≥80% catastrophic recall, ≤10% GOOD FPR). At
threshold = 25°: 100% GOOD recall, 20% catastrophic acceptance.

The catastrophic numbers above are from synthetic perturbations.
Real catastrophic-pair characterization requires the v2 matrix
re-run.

### ⚠️ 90° yaw is invisible

The metric cannot distinguish a 90° yaw error from GOOD because the
cube has a 4-fold rotational symmetry around each body axis — the
canonicalization absorbs the 90° rotation as "the same cube
relabeled." A recognizer that systematically gets yaw wrong by an
exact 90° will not be flagged by this metric. Orthogonal signals
(e.g., sticker spacing in `rubik_recognizer/recognizer.py` from
PR #200) handle the 90°-yaw failure mode.

### ⚠️ Unflipped / same-pose pairs (Codex P1 on PR #246)

If the user fails to flip the cube and takes two photos of the
same URF view, `axes_B = axes_A` (modulo small camera shake). The
canonicalization search includes `((0,1,2), (1,-1,-1))` which IS
R_FLIP itself, so the search finds a transform that emulates R_FLIP
and reports ~0° "consistency" — incorrectly flagging a same-pose
pair as GOOD. Verified empirically: same-pose pairs measure median
10.62°, 35/35 under 25° (looks GOOD by the canonicalized metric
alone).

**Mitigation:** `consistency_features()` returns both the
canonicalized value AND the raw (un-canonicalized) value. The raw
metric DOES detect same-pose pairs — exactly 180° because R_A = R_B
makes the residual equal R_FLIP. v2 integration must include both
features so the trust ranker can detect "low canon + ~180° raw" as
the same-pose mode.

Defense in depth: an unflipped pair would also produce malformed
facelet output (URF colors only, no D/L/B faces) and be rejected
by upstream facelet-validity checks before reaching the trust signal.

## API: canonicalized scalar OR multi-feature dict

Two integration paths offered:

  | Function                                  | Returns                | When to use                                                  |
  |-------------------------------------------|------------------------|--------------------------------------------------------------|
  | `canonicalized_two_view_consistency_deg`  | scalar (degrees)        | Single-feature integration; primary canonical residual only. |
  | `consistency_features`                    | dict with 3 features    | v2 trust ranker integration (preferred); supplies canonicalized, raw, and gap to allow the classifier to learn the same-pose mitigation. |
  | `best_canonicalization`                   | (residual, perm, signs) | Diagnostics; inspect which signed permutation aligned a pair. |

## Integration plan after this PR

Math primitive (PR #245) and canonicalization helper (this PR) cover
"Step 0" of the Phase 4 v2 integration plan in
`tools/TWO_VIEW_CONSISTENCY.md`. After both land, the remaining work
is mechanical:

1. **Modify `tools/phase2b_recompute.py`** to record raw
   `model.axis_x_2d / axis_y_2d / axis_z_2d` per run alongside the
   existing per-run signals.
2. **Add a post-processing pass** in `recompute_all()`: pair each
   set's A and B runs, call `consistency_features(axes_A, axes_B)`
   (NOT the bare scalar `canonicalized_two_view_consistency_deg`),
   and inject all THREE returned fields into both A's and B's matrix
   rows as separate features:
     - `two_view_orientation_consistency_canonicalized_deg`
     - `two_view_orientation_consistency_raw_deg`
     - `two_view_orientation_consistency_canon_gap_deg`

   **Why all three, not just the canonicalized scalar:** the
   canonicalized value alone has the documented same-pose false
   negative (low canon score on `axes_B == axes_A` because the
   48-transform search incidentally finds R_FLIP — Codex P1 on
   the prior canonicalization PR). The raw value detects this
   exactly (180° when R_A = R_B). The trust ranker classifier
   needs both to learn the joint "low canon + ~180° raw"
   same-pose signature.

3. **Re-run `tools/phase2b_recompute.py`** to regenerate
   `tests/fixtures/phase2b_recomputed_signals.json` with all 3
   new features populated per row. Cost: ~30-60 min (rembg on
   70 photos × 2 runs).
4. **Update `tools/phase4_trust_ranker.py`** `FEATURE_COLUMNS` to
   include all 3 new fields (NOT just `..._canonicalized_deg`).
   Re-run the 4-model bake-off. Target: 80% catastrophic recall
   at ≤10% GOOD FPR (Phase 2 acceptance bar — v1.1 measured at
   36% FPR with 6 features).

   **Empirical-gate before claiming ranker lift** (Codex's design
   bias on the prior PR): before promoting any feature into the
   shipped ranker, characterize per-ROW behavior on real
   catastrophic pairs from the matrix — not just aggregate
   AUC/recall. If the canonicalized value alone improves
   aggregate FPR but fails the same-pose row check, the integration
   ships the multi-feature dict and the classifier's feature
   importance reports the same-pose mitigation working.

The validation set for the integration is the existing
phase2b_recomputed_signals fixture (which has the GOOD/catastrophic
ground-truth labels per pair). The canonicalization helper has its
own validation against `gcm_axis_ground_truth.json` (this PR).

## See also

- `tools/two_view_canonicalization.py` — the helper implementation.
- `tests/test_two_view_canonicalization.py` — 20 tests including
  the empirical contract `GOOD-pair median ≤25°` and the same-pose
  Codex P1 documentation/mitigation.
- `tools/two_view_consistency.py` (PR #245) — the math primitive
  this helper wraps.
- `tools/TWO_VIEW_CONSISTENCY.md` (PR #245) — overall integration
  plan; this doc covers "Step 0" of that plan.
- `tests/fixtures/gcm_axis_ground_truth.json` — the 35-pair
  human-labeled validation set.
