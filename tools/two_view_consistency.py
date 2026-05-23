#!/usr/bin/env python3
"""Two-view orientation consistency: a candidate Phase 4 v2 trust signal.

## Why this exists

Phase 4 v1.1 (ctvd #244) showed the data lever doesn't move the trust
ranker bar — best learned model still well above the 10% FPR target.
The remaining lever is a stronger feature, and the natural candidate
is a pair-level geometric signal: do the A and B view fits agree
about the cube's 3D pose under the documented capture rotation?

This module provides the pure math for that signal. Integration into
the matrix-regeneration pipeline (`phase2b_recompute.py`) is a
follow-up that triggers the slow rembg re-run; see
`tools/TWO_VIEW_CONSISTENCY.md` for the integration plan.

## The capture convention

Between photo A (white-up, URF visible) and photo B (yellow-up, DLB
visible), the user flips the cube end-over-end. **The flip is a
single 180° rotation around the image-horizontal axis (camera X)** —
the natural physical motion of gripping the cube by its R and L
sides and rotating it through the horizontal axis between those
sides. There is no additional rotation. The effect of this single
flip is to expose the previously-hidden faces (D, L, B) to the
camera, replacing the previously-visible faces (U, R, F).

(The cube-snap CameraCapture.tsx UI text currently says "Turn it
upside down, then rotate 90° clockwise from above" — that "then 90°
clockwise" portion is incorrect with respect to the project's actual
capture convention and should be fixed in cube-snap. The 54-char
URFDLB-ordering logic in the recognizer assumes only the 180° flip.)

## The metric

For a well-fit pair where both fits agree about the cube's pose, the
residual rotation `R_B · R_A⁻¹` should be the 180° rotation around
camera X — call it `R_FLIP`. The metric reports how far the observed
residual lies from that ideal:

  consistency_deg = rotation_angle((R_FLIP · R_A) · R_B⁻¹)

- **Well-fit pair, pure horizontal-axis flip:** ~0°.
- **Well-fit pair, user's grip slightly tilted off horizontal:**
  small residual (~5-15°) reflecting the tilt.
- **Mis-fit pair:** large residual — the recovered transform is far
  from a 180° around camera X.

No sampling/candidate set — there's a single canonical R_FLIP, and
small user-grip variations naturally surface as small values in the
metric rather than being absorbed by a candidate min.

## Recovery of body-to-camera rotation from 2D axis projections

The global cube model encodes pose via 3 image-space axis projection
vectors (ax_2d, ay_2d, az_2d) under weak-perspective projection. The
2×3 matrix M = [ax | ay | az] is the top-2 rows of (s · R), where s
is the projection scale and R is the 3×3 rotation matrix from cube-
body axes to camera frame.

Recovery:
- s ≈ ||M||_F / √2  (because the 3 columns of R have unit norm; the
  top-2 rows of R have Frobenius norm √(3 − 1) = √2)
- R[:2, :3] = M / s, orthonormalized via Gram-Schmidt
- R[2,  :3] = R[0, :] × R[1, :]  (cross product; sign-fix with det)

## Implementation

Given recovered R_A and R_B (each 3×3 body-to-camera rotation):

  R_FLIP = 180° rotation around camera X = diag(1, −1, −1)
  consistency_deg = rotation_angle((R_FLIP · R_A) · R_B⁻¹)

The expected world-frame rotation R_FLIP acts on the LEFT of R_A
(R_B = R_FLIP · R_A) because the cube rotates in the fixed
world/camera frame, not in its own body frame. A perfectly
consistent pair gives a residual close to the identity; the
returned angle is ~0°. A mis-fit pair gives a large angle.

## v1 history (math bug corrected here)

The v1 draft of this module (PR #243) used `R_Y(180°)` (a 180°
rotation around the vertical axis) as the expected world transform.
Codex caught the left-vs-right multiply convention but missed the
more fundamental axis-of-rotation error. Empirical evidence on the
70-case matrix: v1 reported median 124.9° for GOOD pairs (should be
~0°) and 174.8° for catastrophic — discriminative but with a huge
bias floor. Switching to the correct horizontal-axis flip drops the
GOOD-pair bias to near zero while preserving discrimination.

## Input precondition (LOAD-BEARING — read before integrating)

`two_view_consistency_deg(axes_A, axes_B)` assumes its two inputs
are expressed in the **same semantic cube-body axis frame** — i.e.,
the body X axis in A means the same physical cube axis as the body
X axis in B. The math fails silently otherwise: a systematic axis
relabel between A and B produces a residual rotation that does NOT
match R_FLIP, even on perfectly fit pairs.

**This is not automatically true** for the raw axes the recognizer
writes onto `GlobalCubeModel`. Empirically (Codex audit on PR #245):
applying this function directly to 35 human-labeled A/B axis pairs
from `tests/fixtures/gcm_axis_ground_truth.json` gives **median 173°
residual and 0/35 pairs under 25°**. That is the signature of an
A↔B axis-frame mismatch, not a math bug: every pair, including
known-GOOD pairs, scores like a catastrophic mis-fit.

In photo A (URF visible) the recognizer sees +U, +R, +F faces;
in photo B (DLB visible) it sees +D, +L, +B faces. Whether the
recognizer's `axis_x_2d / axis_y_2d / axis_z_2d` are tied to
"visible-face vertices" (in which case A and B point opposite ways
in the body frame) or to "canonical cube body axes" (in which case
they are consistent) determines whether the raw vectors can be fed
in directly. The empirical 173° median is strong evidence they are
NOT consistent without canonicalization.

**Scope of this module:** this is the low-level math primitive. The
caller is responsible for the canonicalization step. See
`tools/TWO_VIEW_CONSISTENCY.md` "Integration plan" → "Step 0" for
the canonicalization investigation that must precede any v2 wiring.
Do not pass raw per-image `model.axis_x_2d / model.axis_y_2d /
model.axis_z_2d` directly into this function until that step is
complete and validated against the ground-truth fixture.

## Limitations

- Assumes weak-perspective projection. Real iPhone photos have mild
  perspective; treat values as approximate.
- Assumes the user kept the camera stationary between A and B photos.
  In practice, the camera moves slightly. The metric will have a
  baseline noise floor of ~5-15° even on correct fits.
- The "expected" 180° rotation is around camera X (image-horizontal,
  the documented capture convention). If the user instead spins the
  cube around the vertical axis between photos, or rotates it some
  other way, the metric will report a large residual — which is the
  intended behavior (the resulting facelet string would also be
  wrong). Capture convention is documented in cube-snap's CLAUDE.md
  / capture flow docs.
- Assumes its inputs are already in the same body-axis frame (see
  Input precondition section above). Without that, the metric is
  meaningless.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# --- The expected world rotation -----------------------------------------

# R_FLIP: the 180° rotation around camera X (image-horizontal axis).
# This is the documented capture convention — see module docstring.
# Closed form via Rodrigues' rotation formula for axis (1, 0, 0):
# R = 2 · n nᵀ − I = diag(1, −1, −1).
R_FLIP = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=np.float64)


# Legacy alias retained for any external caller that may have imported
# `Y180` from this module (pre-fix docstring referenced it). Not used
# in the metric; kept as a no-op import safety net. The v1 math
# (180° around vertical Y) was wrong; the correct flip is R_FLIP above.
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

    **PRECONDITION:** `axes_A` and `axes_B` MUST already be expressed
    in the same semantic cube-body axis frame. Raw per-image axes
    from `GlobalCubeModel.axis_x_2d / axis_y_2d / axis_z_2d` are NOT
    guaranteed to satisfy this — see the "Input precondition" section
    of this module's docstring for the empirical evidence (median
    173° residual on 35 ground-truth labeled pairs without
    canonicalization) and `tools/TWO_VIEW_CONSISTENCY.md` for the
    integration-plan canonicalization step that must run first.

    Output: angle in degrees. 0 means A and B agree perfectly that
    the cube was flipped 180° around camera X between photos (the
    documented capture convention — see module docstring). Higher
    values indicate disagreement.

    The expected world-frame rotation R_FLIP acts on the LEFT of R_A
    (R_B = R_FLIP · R_A) because the cube rotates in the fixed
    world/camera frame, not in its own body frame.
    """
    R_A = recover_rotation_from_axes(*axes_A)
    R_B = recover_rotation_from_axes(*axes_B)
    residual = (R_FLIP @ R_A) @ R_B.T
    return rotation_angle_deg(residual)
