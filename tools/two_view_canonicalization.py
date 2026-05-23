#!/usr/bin/env python3
"""A/B axis canonicalization for two-view orientation consistency.

## Why this exists

`tools/two_view_consistency.py` ships the math primitive
`two_view_consistency_deg(axes_A, axes_B)`. It assumes A and B are
already in the same semantic cube-body axis frame. Empirically that
assumption is NOT satisfied for raw per-image `GlobalCubeModel.
axis_x_2d / axis_y_2d / axis_z_2d` vectors: Codex's audit of PR #245
measured a median 173° residual and 0/35 pairs under 25° using raw
axes on 35 human-labeled GOOD pairs.

This module supplies the canonicalization step needed before the
primitive can be used in v2 of the trust ranker pipeline.

## The approach

The recognizer's per-image fit assigns `axis_x_2d / axis_y_2d /
axis_z_2d` based on a correspondence search whose A↔B body-frame
labeling can differ by any signed permutation of the 3 axes — 48
combinations total: 6 permutations × 8 sign-flip patterns.

24 of those 48 are "real" cube rotations (det=+1, elements of the
chiral octahedral group); the other 24 (det=-1) are reflections,
which aren't physical cube symmetries but DO show up empirically as
the best A↔B alignment for ~one-third of human-labeled GOOD pairs.
The recognizer's per-image chirality detection (PR #218/#220)
corrects most chirality errors but residual A↔B left-handed-vs-
right-handed labeling differences still occur.

Codex P2 on PR #246: restricting the canonicalization search to
det=+1 only excluded the very transforms that align 12/35 known-
GOOD pairs (sets 12, 20, 24, 27, 30, 36, 37, 38, 39, 40, 43, 44,
46), forcing those pairs to ~55° residuals. The fix is to search
all 48 signed permutations.

  consistency_deg = min over 48 signed axis permutations T of
                    two_view_consistency_deg(axes_A, T · axes_B)

## Empirical validation (35 human-labeled A/B axis pairs)

Validation set: `tests/fixtures/gcm_axis_ground_truth.json` filtered
to pairs where both `_A` and `_B` carry `approved=true`. N=35.

  | Approach                              | median   | under_25 | max     |
  |---------------------------------------|----------|----------|---------|
  | Raw axes (no canonicalization)        | 173.34°  |  0/35    | 179.52° |
  | det=+1 only (24 cube rotations)       |  15.56°  | 23/35    |  61.08° |
  | All 48 signed axis permutations       |  10.66°  | 35/35    |  23.41° |

The 48-transform canonicalization brings ALL 35 known-good pairs
under the 25° target. The max residual on any GOOD pair is 23.41°,
giving a clean separation from synthetic catastrophic perturbations
(see discrimination section below).

## Discrimination characterization

Threshold sweep against the 35 GOOD pairs and the same pairs with
A's axes perturbed by a 30° yaw rotation (simulating a recognizer
fit that got A's yaw wrong):

  | threshold | GOOD < thr | yaw-30° BAD < thr |
  |-----------|------------|-------------------|
  |    10°    |   45.7%    |    0.0%           |
  |    15°    |   68.6%    |    0.0%           |
  |    20°    |   88.6%    |    0.0%           |
  |    25°    |  100.0%    |   20.0%           |
  |    30°    |  100.0%    |   34.3%           |
  |    40°    |  100.0%    |   74.3%           |

Strong discrimination in the [15°, 25°] range. At threshold = 20°:
88.6% GOOD recall, 0% catastrophic acceptance — well above the
Phase 2 target (≥80% catastrophic recall, ≤10% GOOD FPR). At
threshold = 25°: 100% GOOD recall, 20% catastrophic acceptance.

The catastrophic discrimination shown above is on SYNTHETIC
perturbations (3D rotations applied to known-good A axes). Real
catastrophic-pair characterization requires the v2 matrix re-run.

## Known limitations

### 1. Unflipped / same-pose pair (Codex P1 on PR #246)

If the user fails to flip the cube and takes two photos of the same
URF view, the recognizer produces R_A ≈ R_B. The canonicalization
search includes `((0,1,2), (1,-1,-1))` which IS R_FLIP itself, so
the search finds a transform that emulates R_FLIP and reports
~0° "consistency" — incorrectly flagging a same-pose pair as GOOD.

Verified empirically on the 35 ground-truth A axes substituted as
both A and B: canonicalized median 10.62°, 35/35 under 25° (all
look "consistent" by the metric).

**The RAW metric (no canonicalization) IS the signal for this case:**
exactly 180° for same-pose pairs, ~173° for real GOOD pairs. The
v2 integration must consume `consistency_features(...)` (defined
below) which returns both canonicalized and raw values so the
trust ranker can detect "low canon + exactly-180° raw" as the
same-pose mode.

Production pipeline note: an unflipped pair would also produce
malformed facelet output (URF colors only, no D/L/B faces) and be
rejected by upstream facelet-validity checks. The geometry trust
signal is a defense-in-depth layer, not the only gate.

### 2. 90° yaw rotation indistinguishable from GOOD

A recognizer that systematically gets yaw wrong by exactly 90° is
absorbed by the cube's 4-fold rotational symmetry around each body
axis and reports ~0° residual. Orthogonal signals (e.g., sticker
spacing in `rubik_recognizer/recognizer.py` from PR #200) must
handle the 90°-yaw failure mode.

### 3. Search cost

Min-over-48 adds ~48× the bare primitive cost. On the 70-pair
matrix this is still microseconds per pair — negligible.

### 4. Catastrophic absorption risk

The min-over-48 approach can incidentally find a small residual
for a catastrophic pair if the wrong fit happens to align under
some signed permutation. Empirically rare on synthetic 30° yaw
perturbation (7/35 = 20% under 25°) but characterize on real
catastrophic pairs once axes are captured in the matrix.
"""

from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np

from tools.two_view_consistency import (
    R_FLIP,
    recover_rotation_from_axes,
    rotation_angle_deg,
)

Axes = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]


# --- The 48 signed axis permutations, as (perm, signs) pairs ---
#
# 24 of these are proper cube rotations (det=+1, chiral octahedral
# group elements). The other 24 are reflections (det=-1). Empirically,
# the recognizer's per-image labeling can produce A↔B alignments that
# require either; see module docstring for the empirical evidence.


def _perm_sign(perm: Tuple[int, ...]) -> int:
    """Parity of a permutation: +1 for even, -1 for odd."""
    p = list(perm)
    s = 1
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                s *= -1
    return s


def _transform_det(perm: Tuple[int, ...], signs: Tuple[int, ...]) -> int:
    """Determinant of the (axis permutation + sign-flip) linear transform.
    +1 for proper rotations (the 24 cube rotations); -1 for reflections."""
    d = _perm_sign(perm)
    for v in signs:
        d *= v
    return d


# All 48 signed axis permutations. Precomputed once at import time.
ALL_AXIS_TRANSFORMS: Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...] = tuple(
    itertools.product(
        itertools.permutations((0, 1, 2)),
        itertools.product((1, -1), repeat=3),
    )
)

# Subset retained for diagnostics: just the 24 det=+1 proper rotations.
# Not used in the metric — kept for tests that want to compare det=+1
# vs det=±1 behavior.
CUBE_ROTATIONS: Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...] = tuple(
    t for t in ALL_AXIS_TRANSFORMS if _transform_det(*t) == 1
)


def _apply_transform(
    axes: Axes,
    perm: Tuple[int, ...],
    signs: Tuple[int, ...],
) -> Axes:
    """Apply (perm, signs) to a 3-axis tuple: reorder by perm, flip signs."""
    reordered = [axes[i] for i in perm]
    return tuple(
        (signs[i] * reordered[i][0], signs[i] * reordered[i][1])
        for i in range(3)
    )


# --- Public API ---


def canonicalized_two_view_consistency_deg(
    axes_A: Axes,
    axes_B_raw: Axes,
) -> float:
    """Canonicalized two-view orientation consistency.

    Takes the raw per-image axes (from `GlobalCubeModel.axis_x_2d/
    axis_y_2d/axis_z_2d`) for both A and B, and returns the minimum
    `two_view_consistency_deg` residual over the 48 signed axis
    permutations applied to B's axes — i.e., the residual under the
    most charitable A/B body-frame alignment achievable via any
    signed permutation of B's labeled axes.

    Returns: angle in degrees in [0, 180]. Lower = better agreement.

    Raises `ValueError` if either input is degenerate (zero-norm
    axes) — caller must filter/reject those rows rather than propagate
    a non-finite trust feature. (Codex P2 on PR #246: the loop must
    not silently return `inf` when every transform fails recovery.)

    For a well-fit GOOD pair on the human-labeled validation set,
    expect median ~11° (max ~23°, all 35 pairs under 25°). For a
    catastrophic pair (one or both fits geometrically wrong), expect
    ≥25° in most cases.

    See module docstring for the empirical justification of the
    48-transform search (including why det=-1 reflections are
    necessary, not just the 24 det=+1 cube rotations) and the
    discrimination characterization.
    """
    # Validate axes_A first — fail fast with the same error shape
    # `recover_rotation_from_axes` would raise.
    R_A = recover_rotation_from_axes(*axes_A)

    best = float("inf")
    n_valid = 0
    for perm, signs in ALL_AXIS_TRANSFORMS:
        axes_B = _apply_transform(axes_B_raw, perm, signs)
        try:
            R_B = recover_rotation_from_axes(*axes_B)
        except ValueError:
            continue
        n_valid += 1
        residual = (R_FLIP @ R_A) @ R_B.T
        deg = rotation_angle_deg(residual)
        if deg < best:
            best = deg
    if n_valid == 0:
        # Every signed permutation of axes_B_raw is degenerate (all-
        # zero axes_B_raw, or some pathological input where every
        # transform produces zero-norm vectors). The recover_…
        # primitive would have raised; we maintain that contract.
        raise ValueError(
            "axes_B_raw is degenerate — no signed permutation produces "
            "a valid rotation. (Caller must filter degenerate rows "
            "rather than receive a non-finite consistency value.)"
        )
    return float(best)


def consistency_features(
    axes_A: Axes,
    axes_B_raw: Axes,
) -> dict:
    """Return a dict of two-view consistency features for the trust
    ranker. The dict shape is the integration contract for v2.

    Why a dict (multiple features) instead of a single scalar: per
    Codex P1 on PR #246, the canonicalized value alone cannot detect
    unflipped/same-pose pairs (where the search incidentally finds
    R_FLIP as the "canonicalization"). Pairing the canonicalized value
    with the raw (un-canonicalized) value lets the trust ranker
    classifier detect the "low canon + ~180° raw" same-pose mode.

    Keys returned:
      - `canonicalized_deg`: min over 48 signed axis permutations
        (the primary feature; ~10° for GOOD pairs, ≥25° for
        catastrophic on average).
      - `raw_deg`: residual with no canonicalization (T = identity);
        sensitive to the same-pose mode (exactly 180°) and the
        recognizer's labeling artifacts on legitimate pairs.
      - `canon_gap_deg`: raw_deg − canonicalized_deg. Large gap
        means the canonicalization absorbed a lot of disagreement
        (typical for ambiguous-labeling GOOD pairs and same-pose
        false-positives both — not a clean signal on its own, but
        included for completeness).

    Raises `ValueError` if either input is degenerate — same contract
    as `canonicalized_two_view_consistency_deg`.
    """
    canon = canonicalized_two_view_consistency_deg(axes_A, axes_B_raw)
    # Raw: residual with the identity transform.
    R_A = recover_rotation_from_axes(*axes_A)
    R_B = recover_rotation_from_axes(*axes_B_raw)
    residual = (R_FLIP @ R_A) @ R_B.T
    raw = rotation_angle_deg(residual)
    return {
        "canonicalized_deg": float(canon),
        "raw_deg": float(raw),
        "canon_gap_deg": float(raw - canon),
    }


def best_canonicalization(
    axes_A: Axes,
    axes_B_raw: Axes,
) -> Tuple[float, Tuple[int, ...], Tuple[int, ...]]:
    """Like canonicalized_two_view_consistency_deg but also returns the
    winning (perm, signs) so callers can inspect which signed axis
    permutation aligned the two views. Useful for diagnostics.

    Raises `ValueError` if either input is degenerate — same contract
    as `canonicalized_two_view_consistency_deg`.
    """
    R_A = recover_rotation_from_axes(*axes_A)

    best_deg = float("inf")
    best_pair: Tuple[Tuple[int, ...], Tuple[int, ...]] = (
        (0, 1, 2), (1, 1, 1)
    )
    n_valid = 0
    for perm, signs in ALL_AXIS_TRANSFORMS:
        axes_B = _apply_transform(axes_B_raw, perm, signs)
        try:
            R_B = recover_rotation_from_axes(*axes_B)
        except ValueError:
            continue
        n_valid += 1
        residual = (R_FLIP @ R_A) @ R_B.T
        deg = rotation_angle_deg(residual)
        if deg < best_deg:
            best_deg = deg
            best_pair = (perm, signs)
    if n_valid == 0:
        raise ValueError(
            "axes_B_raw is degenerate — no signed permutation produces "
            "a valid rotation."
        )
    return float(best_deg), best_pair[0], best_pair[1]
