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
- two_view_consistency_deg: ~0° for ground-truth consistent pair,
  meaningfully larger for explicit yaw disagreement.
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


# ----- two_view_consistency_deg -----


def test_consistency_zero_for_truly_consistent_pair():
    """A and B from the same body-to-camera pose, but the cube was
    rotated 180° around world-Y between photos (camera fixed). The
    pose transforms as R_B = Y180 · R_A (LEFT-multiplication — the
    rotation happens in the world frame, not the cube's body frame).
    Expected: ~0°.

    Codex P2 catch on the v1 draft: using R_A @ Y180 (right-multiply,
    body-frame convention) gave ~70° residual on this exact synthetic
    pose because of the isometric pitch.
    """
    R_A = _euler_to_R(yaw_deg=45.0, pitch_deg=-35.26, roll_deg=0.0)
    R_B = tvc.Y180 @ R_A  # World-frame rotation: left-multiply
    axes_A = _project_axes(R_A)
    axes_B = _project_axes(R_B)
    deg = tvc.two_view_consistency_deg(axes_A, axes_B)
    assert deg < 1.0, f"expected ~0° for consistent pair, got {deg:.2f}°"


def test_consistency_handles_sign_ambiguity():
    """A 180° rotation around the same axis is its own inverse — cube
    rotated +180° and cube rotated -180° produce visually identical
    photos. The function should not penalize either sign convention."""
    R_A = _euler_to_R(yaw_deg=45.0, pitch_deg=-35.26, roll_deg=0.0)
    R_B = tvc.Y180.T @ R_A  # Reversed sign, world-frame
    axes_A = _project_axes(R_A)
    axes_B = _project_axes(R_B)
    deg = tvc.two_view_consistency_deg(axes_A, axes_B)
    assert deg < 1.0, (
        f"sign-flipped pair should still be ~0° consistent, got {deg:.2f}°"
    )


def test_consistency_detects_yaw_disagreement():
    """If A and B disagree about the cube's yaw by a meaningful amount,
    consistency should report nontrivial degrees of disagreement."""
    R_A = _euler_to_R(yaw_deg=45.0, pitch_deg=-35.26, roll_deg=0.0)
    # B is what would be observed if the underlying cube had different
    # yaw (75° instead of 45°), then was rotated 180° around world-Y.
    R_B = tvc.Y180 @ _euler_to_R(yaw_deg=75.0, pitch_deg=-35.26, roll_deg=0.0)
    axes_A = _project_axes(R_A)
    axes_B = _project_axes(R_B)
    deg = tvc.two_view_consistency_deg(axes_A, axes_B)
    assert deg > 10.0, (
        f"30° yaw disagreement should surface as a clearly nonzero "
        f"consistency angle, got {deg:.2f}°"
    )
