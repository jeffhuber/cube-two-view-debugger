"""Unit tests for tools/two_view_consistency.py.

Note: this is the orientation-based consistency signal (operates on
global-model axis projections). It's distinct from the existing
`tests/test_two_view_consistency.py` which covers Codex's earlier
sticker-spacing-ratio signal in `rubik_recognizer/recognizer.py`
(PR #200). The two signals are complementary 7th-feature candidates
for the Phase 4 trust ranker.

Coverage:
- recover_rotation_from_axes: synthetic round-trip from known R →
  projected 2D axes → recovered R should match within tolerance.
- rotation_angle_deg: identity = 0, Y180 = 180, 90° around Z = 90.
- R_FLIP: documented capture convention is 180° around camera X.
- two_view_consistency_deg: ~0° for a consistent pair flipped by
  R_FLIP, large for other rotations (yaw spin, off-axis flips, etc).
"""

from __future__ import annotations

import numpy as np
import pytest

from tools import two_view_consistency as tvc


def _project_axes(R: np.ndarray, s: float = 100.0):
    """Forward: given a 3D rotation R and scale s, return the 3 projected
    2D axis vectors observed under weak-perspective projection."""
    M = s * R[:2, :3]
    return (
        (float(M[0, 0]), float(M[1, 0])),
        (float(M[0, 1]), float(M[1, 1])),
        (float(M[0, 2]), float(M[1, 2])),
    )


def _euler_to_R(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Intrinsic Z·X·Y Euler rotation (degrees) → 3×3 matrix.

    Matches the convention used in tools/global_cube_model.py:
    R = Rz(roll) · Rx(pitch) · Ry(yaw)
    """
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0],
                   [np.sin(r), np.cos(r), 0],
                   [0, 0, 1]])
    return Rz @ Rx @ Ry


# ----- recover_rotation_from_axes -----


def test_recover_identity_rotation():
    R = np.eye(3)
    axes = _project_axes(R, s=100.0)
    R_rec = tvc.recover_rotation_from_axes(*axes)
    assert np.allclose(R_rec, R, atol=1e-6)


def test_recover_isometric_rotation():
    """Recovering a typical isometric cube view (yaw=45, pitch=-35.26)
    should round-trip within a small tolerance."""
    R = _euler_to_R(yaw_deg=45.0, pitch_deg=-35.26, roll_deg=0.0)
    axes = _project_axes(R, s=200.0)
    R_rec = tvc.recover_rotation_from_axes(*axes)
    angle = tvc.rotation_angle_deg(R @ R_rec.T)
    assert angle < 1.0, f"recovered rotation differs by {angle:.3f}°"


def test_recover_rejects_zero_axes():
    with pytest.raises(ValueError, match="degenerate"):
        tvc.recover_rotation_from_axes((0, 0), (0, 0), (0, 0))


# ----- rotation_angle_deg -----


def test_rotation_angle_identity_is_zero():
    assert tvc.rotation_angle_deg(np.eye(3)) == pytest.approx(0.0, abs=1e-9)


def test_rotation_angle_180_around_y():
    assert tvc.rotation_angle_deg(tvc.Y180) == pytest.approx(180.0, abs=1e-6)


def test_rotation_angle_90_around_z():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert tvc.rotation_angle_deg(R) == pytest.approx(90.0, abs=1e-6)


# ----- R_FLIP (the expected world rotation) -----


def test_R_FLIP_is_180deg_total():
    """The documented capture rotation is a 180° flip; verify R_FLIP
    has rotation angle exactly 180°."""
    assert tvc.rotation_angle_deg(tvc.R_FLIP) == pytest.approx(180.0, abs=1e-6)


def test_R_FLIP_axis_is_camera_X():
    """R_FLIP should fix the camera-X axis (the rotation axis) and
    negate the other two unit axes (since 180° around (1,0,0) sends
    (0,1,0)→(0,-1,0) and (0,0,1)→(0,0,-1))."""
    expected = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    assert np.allclose(tvc.R_FLIP, expected, atol=1e-12)


def test_R_FLIP_is_orthogonal_with_det_plus_one():
    """Sanity: R_FLIP is a proper rotation (R @ R.T = I, det = +1)."""
    assert np.allclose(tvc.R_FLIP @ tvc.R_FLIP.T, np.eye(3), atol=1e-12)
    assert float(np.linalg.det(tvc.R_FLIP)) == pytest.approx(1.0, abs=1e-12)


# ----- two_view_consistency_deg -----


def test_consistency_zero_for_camera_X_flip():
    """Cube flipped 180° around camera X between photos — the
    documented capture convention ("grip by R/L sides, flip end-over-
    end"). Expected: ~0°."""
    R_A = _euler_to_R(yaw_deg=45.0, pitch_deg=-35.26, roll_deg=0.0)
    R_B = tvc.R_FLIP @ R_A
    axes_A = _project_axes(R_A)
    axes_B = _project_axes(R_B)
    deg = tvc.two_view_consistency_deg(axes_A, axes_B)
    assert deg < 1.0, f"expected ~0° for camera-X flip, got {deg:.2f}°"


def test_consistency_zero_for_identity_pair_after_flip():
    """Cube starts at identity in A and identity-then-flip in B —
    most trivial possible consistent pair."""
    R_A = np.eye(3)
    R_B = tvc.R_FLIP
    axes_A = _project_axes(R_A)
    axes_B = _project_axes(R_B)
    deg = tvc.two_view_consistency_deg(axes_A, axes_B)
    assert deg < 1e-3, f"expected ~0° for identity-then-flip, got {deg:.4f}°"


def test_consistency_detects_camera_Z_flip_as_inconsistent():
    """If A and B differ by a 180° rotation around camera Z (the depth
    axis) rather than the documented camera-X flip, the metric should
    report a large residual. Camera Z is NOT the capture convention —
    this is the kind of mis-fit the signal exists to catch."""
    R_A = _euler_to_R(yaw_deg=45.0, pitch_deg=-35.26, roll_deg=0.0)
    R_Z_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
    R_B = R_Z_180 @ R_A
    axes_A = _project_axes(R_A)
    axes_B = _project_axes(R_B)
    deg = tvc.two_view_consistency_deg(axes_A, axes_B)
    assert deg > 60.0, (
        f"camera-Z 180° (wrong axis) should produce large residual, "
        f"got {deg:.2f}°"
    )


def test_consistency_detects_vertical_axis_rotation_as_inconsistent():
    """If A and B differ by a rotation around the VERTICAL axis (e.g.,
    the user spun the cube around its top-to-bottom axis instead of
    flipping it end-over-end), R_FLIP does not match. The metric
    should report a large residual."""
    R_A = _euler_to_R(yaw_deg=45.0, pitch_deg=-35.26, roll_deg=0.0)
    # B = A spun 180° around vertical (camera Y). This is the
    # "spinning, not flipping" failure mode.
    R_Y_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    R_B = R_Y_180 @ R_A
    axes_A = _project_axes(R_A)
    axes_B = _project_axes(R_B)
    deg = tvc.two_view_consistency_deg(axes_A, axes_B)
    assert deg > 60.0, (
        f"vertical-axis 180° (spin, not flip) should produce large "
        f"residual, got {deg:.2f}°"
    )


def test_consistency_small_for_slightly_tilted_grip():
    """If the user's grip tilts the flip axis ~5° off pure camera X,
    the residual should be small (reflecting just the tilt) — not
    zero (it's not pure R_FLIP) and not large (it's close)."""
    R_A = _euler_to_R(yaw_deg=45.0, pitch_deg=-35.26, roll_deg=0.0)
    # 5° tilt: flip around an axis (cos 5°, 0, sin 5°) in the horizontal
    # plane, but with a 180° magnitude. Rodrigues: R = 2 n nᵀ − I.
    t = np.radians(5.0)
    c, s = np.cos(t), np.sin(t)
    R_tilted = np.array([
        [2 * c * c - 1, 0.0,            2 * c * s],
        [0.0,           -1.0,           0.0],
        [2 * c * s,     0.0,            2 * s * s - 1],
    ], dtype=np.float64)
    R_B = R_tilted @ R_A
    axes_A = _project_axes(R_A)
    axes_B = _project_axes(R_B)
    deg = tvc.two_view_consistency_deg(axes_A, axes_B)
    # 5° tilt of the axis of a 180° rotation produces ~10° of residual
    # rotation (the metric responds to axis tilt at ~2× the tilt
    # magnitude for 180° rotations). Bound generously to allow some
    # slop without being so loose the test stops gating.
    assert 0.0 < deg < 25.0, (
        f"expected small-but-nonzero residual for 5° grip tilt, "
        f"got {deg:.2f}°"
    )


def test_consistency_legacy_Y180_alias_still_importable():
    """v1 used `tvc.Y180`. Kept as a no-op safety-net alias; not used
    in the metric. Removing it could break any orphan script that
    grabbed the symbol from the pre-fix module."""
    assert hasattr(tvc, "Y180")
    assert tvc.Y180.shape == (3, 3)
