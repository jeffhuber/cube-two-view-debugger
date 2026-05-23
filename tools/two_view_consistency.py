#!/usr/bin/env python3
"""Two-view orientation consistency: a candidate Phase 4 v2 trust signal.

## Why this exists

Phase 4 v1 fit 4 model classes on the 6 features from the existing
Phase 2B matrix. Best result: random_forest at 80% catastrophic recall
/ 18% GOOD false-retake — better than Phase 2B's hand-tuned ceiling
of 31% FPR, but still above the 10% bar.

The Phase 2B fixture has a reserved-but-unpopulated column
(`two_view_consistency_deg`) representing the angle by which A and B
view fits *disagree* about the cube's 3D orientation. When both fits
are physically consistent (cube was rotated 180° around Y between
photos, camera roughly stationary), the recovered orientations should
predict each other. When they disagree significantly, at least one
fit is wrong — a high-signal predictor of catastrophic outcomes that
no single-view feature can capture.

This module provides the pure math for that signal. Integration into
the matrix-regeneration pipeline (`phase2b_recompute.py`) is a
follow-up that triggers the slow rembg re-run; see
`tools/TWO_VIEW_CONSISTENCY.md` for the integration plan.

## The math

The global cube model encodes pose via 3 image-space axis projection
vectors (ax_2d, ay_2d, az_2d) under weak-perspective projection. The
2×3 matrix M = [ax | ay | az] is the top-2 rows of (s · R), where s
is the projection scale and R is the 3×3 rotation matrix from cube-
body axes to camera frame.

Recovery:
- s ≈ ||M||_F / √2  (because the 3 columns of R have unit norm; the
  top-2 rows of R have Frobenius norm √(3 − 1) = √2)
- R[:2, :3] = M / s
- R[2,  :3] = R[0, :] × R[1, :]  (cross product; sign-fix with det)

Once R_A and R_B are recovered, the expected relationship under the
"rotate cube 180° around world Y between photos, keep camera fixed"
convention is **R_B ≈ Y180 · R_A** (left-multiplication: the
rotation acts in the world/camera frame, since the camera is fixed
and world Y is the rotation axis). The body-frame right-
multiplication form `R_A · Y180` would only be correct if Y180 were
the rotation expressed in the cube's body frame — which is NOT the
documented capture convention.

(Codex P2 catch on the v1 draft of this module: with any non-zero
camera pitch, the body-frame form reports a ~70° residual for a
perfectly consistent isometric pair. Always think about whether
rotations are in body or world frame.)

Consistency = the rotation angle of (Y180 · R_A · R_B.T) — 0 means
A and B agree perfectly about the cube's pose. Large values
indicate one (or both) of the fits is geometrically inconsistent.
The function returns the min over (Y180, Y180.T) so a sign-
convention disagreement (cube rotated +180° vs -180° — both visually
identical for a 180° rotation) doesn't double-penalize.

## Limitations

- Assumes weak-perspective projection. Real iPhone photos have mild
  perspective; treat values as approximate.
- Assumes the user kept the camera stationary between A and B photos.
  In practice, the camera moves slightly. The metric will have a
  baseline noise floor of ~5-15° even on correct fits.
- The "expected" 180° rotation is around world Y assuming the cube
  was rotated by the user that way. If they rotated differently
  (e.g., 180° around camera-Z), this metric will report all pairs as
  inconsistent. Photo capture convention is documented in
  cube-snap's CLAUDE.md / capture flow docs.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# Expected world rotation between A and B photos: 180° around world Y.
# Y axis: vertical (cube's white-up axis in A becomes white-down in B).
Y180 = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0],
    [ 0.0,  0.0, -1.0],
], dtype=np.float64)


def recover_rotation_from_axes(
    ax_2d: Tuple[float, float],
    ay_2d: Tuple[float, float],
    az_2d: Tuple[float, float],
) -> np.ndarray:
    """Recover the 3×3 rotation matrix from the 3 2D axis projections.

    Inputs are image-space displacement vectors from cube_center to
    each h-vertex (h_x, h_y, h_z), as stored on GlobalCubeModel.

    Returns a 3×3 numpy array R such that R · [1,0,0]_body ≈ projected
    image direction of the X body axis (etc).

    Algorithm: see module docstring.
    """
    M = np.array([
        [ax_2d[0], ay_2d[0], az_2d[0]],
        [ax_2d[1], ay_2d[1], az_2d[1]],
    ], dtype=np.float64)  # shape (2, 3)
    fro = float(np.linalg.norm(M, ord="fro"))
    if fro < 1e-9:
        raise ValueError("axis projections are all zero — degenerate fit")
    s = fro / np.sqrt(2.0)
    top2 = M / s  # shape (2, 3); approximately top-2 rows of R

    # Orthogonalize the top 2 rows so cross product produces a valid R.
    # (Real fits may have tiny non-orthogonality; project to nearest
    # orthonormal frame via Gram-Schmidt.)
    r0 = top2[0] / max(float(np.linalg.norm(top2[0])), 1e-9)
    r1_raw = top2[1] - r0 * float(np.dot(top2[1], r0))
    r1 = r1_raw / max(float(np.linalg.norm(r1_raw)), 1e-9)
    r2 = np.cross(r0, r1)
    R = np.vstack([r0, r1, r2])

    # Sign-fix: enforce det(R) = +1 (right-handed). If det is negative,
    # the camera is interpreted as flipped — usually a sign of a 180°
    # ambiguity rather than a true measurement, but we flip the third
    # row to make R right-handed.
    if float(np.linalg.det(R)) < 0:
        R[2] = -R[2]
    return R


def rotation_angle_deg(R: np.ndarray) -> float:
    """Return the rotation angle of the 3×3 rotation matrix R in degrees.

    Uses the standard trace formula: angle = arccos((trace − 1) / 2).
    Returns a value in [0, 180].
    """
    # Numerical safety: trace can drift slightly outside [-1, 3] due
    # to floating-point error.
    trace_clipped = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(trace_clipped)))


def two_view_consistency_deg(
    axes_A: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    axes_B: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> float:
    """Compute the orientation-consistency angle (degrees) between two views.

    Inputs: per-view tuples of (ax_2d, ay_2d, az_2d). Each axis is a
    2D vector (in pixel coords) from the cube center to the head of
    that body axis under projection.

    Output: angle in degrees. 0 means A and B agree perfectly about
    the cube's orientation (after applying the expected 180°-around-Y
    world/camera-frame rotation). Higher values indicate disagreement.

    The expected world-frame rotation acts on the LEFT of R_A
    (R_B = Y180 · R_A) because the cube rotates in the fixed
    world/camera frame, not in its own body frame. See module
    docstring for the derivation.

    Returns the min across two sign conventions (Y180 vs Y180.T)
    because a 180° rotation around the same axis is its own inverse:
    cube rotated +180° and cube rotated -180° produce visually
    identical photos, so we shouldn't penalize either direction.
    """
    R_A = recover_rotation_from_axes(*axes_A)
    R_B = recover_rotation_from_axes(*axes_B)
    # Two candidate relationships: R_B = Y180 · R_A   or   R_B = Y180.T · R_A
    # (Right-multiplication would be the body-frame form — wrong here.)
    R_diff_1 = (Y180 @ R_A) @ R_B.T
    R_diff_2 = (Y180.T @ R_A) @ R_B.T
    return min(rotation_angle_deg(R_diff_1), rotation_angle_deg(R_diff_2))
