"""Tests for the projective vertex solver."""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from tools.projective_vertex import (  # noqa: E402
    derive_axis_edge_pairs,
    intersect_lines,
    line_coeffs,
    projective_vertex,
)
from tools.rectify_via_hull_labels import _derive_vertex_from_corners  # noqa: E402


# ---------------- derive_axis_edge_pairs ----------------
# Codex feedback #1: don't hand-wave side A/B symmetry; pin the
# exact line pairs.


def test_derive_axis_edges_side_a_matches_expected():
    """Side A NEAR={1,3,5}. Expected parallel-edge pairs per cube axis:
       V→c1 (X-axis): edges (5,0) and (3,2) are also along X
       V→c3 (Z-axis): edges (1,2) and (5,4)
       V→c5 (Y-axis): edges (1,0) and (3,4)
    Order within each (a,b) tuple doesn't matter for vanishing-point
    intersection — they name the same line.
    """
    pairs = derive_axis_edge_pairs("A")
    by_near = {near: [tuple(sorted(p)) for p in parallels] for near, parallels in pairs}
    by_near = {near: sorted(plist) for near, plist in by_near.items()}
    expected = {
        1: sorted([tuple(sorted((5, 0))), tuple(sorted((3, 2)))]),
        3: sorted([tuple(sorted((1, 2))), tuple(sorted((5, 4)))]),
        5: sorted([tuple(sorted((1, 0))), tuple(sorted((3, 4)))]),
    }
    assert by_near == expected


def test_derive_axis_edges_side_b_matches_expected():
    """Side B NEAR={0,2,4}. Expected:
       V→c2 (X-axis): (4,3) and (0,1)
       V→c0 (Z-axis): (2,1) and (4,5)
       V→c4 (Y-axis): (2,3) and (0,5)
    Different from side A because side B has different NEAR/FAR sets.
    """
    pairs = derive_axis_edge_pairs("B")
    by_near = {near: [tuple(sorted(p)) for p in parallels] for near, parallels in pairs}
    by_near = {near: sorted(plist) for near, plist in by_near.items()}
    expected = {
        2: sorted([tuple(sorted((4, 3))), tuple(sorted((0, 1)))]),
        0: sorted([tuple(sorted((2, 1))), tuple(sorted((4, 5)))]),
        4: sorted([tuple(sorted((2, 3))), tuple(sorted((0, 5)))]),
    }
    assert by_near == expected


def test_derive_axis_edges_returns_exactly_3_entries_per_side():
    for side in ("A", "B"):
        pairs = derive_axis_edge_pairs(side)
        assert len(pairs) == 3
        for near_cn, parallels in pairs:
            assert len(parallels) == 2
            for a, b in parallels:
                assert a != b
                assert 0 <= a <= 5 and 0 <= b <= 5
                # neither endpoint can be the near corner itself
                assert a != near_cn and b != near_cn


# ---------------- line_coeffs + intersect_lines (2D algebra) ----------------


def test_line_coeffs_passes_through_both_points():
    p, q = (1.0, 2.0), (4.0, 6.0)
    a, b, c = line_coeffs(p, q)
    assert abs(a * p[0] + b * p[1] - c) < 1e-9
    assert abs(a * q[0] + b * q[1] - c) < 1e-9


def test_intersect_lines_simple_cross():
    # x-axis (y=0) and y-axis (x=0) intersect at origin
    L_x = line_coeffs((0.0, 0.0), (1.0, 0.0))  # y = 0
    L_y = line_coeffs((0.0, 0.0), (0.0, 1.0))  # x = 0
    pt = intersect_lines(L_x, L_y)
    assert pt is not None
    assert abs(pt[0]) < 1e-9 and abs(pt[1]) < 1e-9


def test_intersect_lines_parallel_returns_none():
    L1 = line_coeffs((0.0, 0.0), (1.0, 0.0))   # y = 0
    L2 = line_coeffs((0.0, 1.0), (1.0, 1.0))   # y = 1 (parallel)
    assert intersect_lines(L1, L2) is None


# ---------------- projective_vertex on canonical iso hexagon ----------------
# Under iso, vanishing points are at infinity → projective should reduce
# to the affine (parallelogram-completion) vertex.


def _canonical_iso_hexagon(center=(500.0, 500.0), radius=200.0):
    """6 hexagon corners at the canonical iso silhouette positions:
    TOP, upper-right, lower-right, BOTTOM, lower-left, upper-left
    (returned in that order = indices 0..5 in the hexagon list)."""
    pts = []
    for deg in (-90, -30, 30, 90, 150, 210):  # CCW from TOP
        a = math.radians(deg)
        pts.append((center[0] + radius * math.cos(a),
                    center[1] + radius * math.sin(a)))
    return pts


def test_projective_vertex_reduces_to_affine_on_iso_input():
    """On a perfect iso hexagon, all 3 vanishing points are at infinity
    (or numerically very far). The 3-line LSQ should give a vertex
    indistinguishable from the parallelogram-completion vertex.
    Per Codex feedback #4 (degeneracy handling) the result should be
    classified as 'near_affine'.
    """
    from tools.rectify_via_hull_labels import _label_corners_by_position

    for side in ("A", "B"):
        hexagon = _canonical_iso_hexagon()
        corners_by_num = _label_corners_by_position(hexagon, side)
        affine_v, _ = _derive_vertex_from_corners(corners_by_num, side)
        result = projective_vertex(corners_by_num, side)
        # Match within 0.1 px (LSQ + intersection floating-point tolerance)
        assert math.hypot(result.vertex[0] - affine_v[0],
                          result.vertex[1] - affine_v[1]) < 0.1
        # Iso → no finite VP → degeneracy is near_affine
        assert result.degeneracy == "near_affine"


# ---------------- projective_vertex on synthetic perspective cube ----------------


def test_projective_vertex_exact_on_synthetic_perspective():
    """Build a 3D cube + known pinhole camera + project to 2D, then
    solve. The projective vertex should match GT exactly (within
    numerical tolerance) since the perspective model is exact.
    """
    import numpy as np

    # Unit cube vertices: V at origin, 7 others at (-1, ±1, ±1) etc.
    # Match the global_cube_model template orientation.
    cube_3d = {
        "vertex": (0.0, 0.0, 0.0),
        "h_x":    (-1.0, 0.0, 0.0),
        "h_y":    (0.0, -1.0, 0.0),
        "h_z":    (0.0, 0.0, -1.0),
        "h_xy":   (-1.0, -1.0, 0.0),
        "h_xz":   (-1.0, 0.0, -1.0),
        "h_yz":   (0.0, -1.0, -1.0),
    }
    # Side A mapping per the convention derived in this PR:
    # c0=h_xy (FAR), c1=h_x (NEAR), c2=h_xz (FAR), c3=h_z (NEAR),
    # c4=h_yz (FAR), c5=h_y (NEAR)
    side_a_map = {0: "h_xy", 1: "h_x", 2: "h_xz", 3: "h_z", 4: "h_yz", 5: "h_y"}

    # Camera pose: close + off-diagonal viewpoint so perspective is
    # strong AND asymmetric (symmetric diagonal cams give nearly-iso
    # projection where affine accidentally matches). cam at
    # (1.5, 2.0, 2.5) → cube subtends a noticeable angle → finite VPs.
    cam_pos = np.array([1.5, 2.0, 2.5])
    target = np.array([-0.3, -0.4, -0.3])
    world_up = np.array([0.0, 1.0, 0.0])
    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, up, -fwd])  # world → camera
    f = 800.0
    cx, cy = 500.0, 500.0

    def project(p3):
        p_cam = R @ (np.array(p3) - cam_pos)
        # camera looks down -Z (in camera coords), so z_cam should be negative for visible
        x = -f * p_cam[0] / p_cam[2] + cx
        y = -f * p_cam[1] / p_cam[2] + cy
        return (float(x), float(y))

    # Project all 8 cube corners
    proj = {name: project(p3) for name, p3 in cube_3d.items()}
    gt_vertex = proj["vertex"]
    corners_by_num = {cn: proj[side_a_map[cn]] for cn in range(6)}

    result = projective_vertex(corners_by_num, "A")

    # Affine (parallelogram) vertex — should be off due to perspective
    affine_v, _ = _derive_vertex_from_corners(corners_by_num, "A")
    affine_err = math.hypot(affine_v[0] - gt_vertex[0], affine_v[1] - gt_vertex[1])
    projective_err = math.hypot(
        result.vertex[0] - gt_vertex[0], result.vertex[1] - gt_vertex[1],
    )

    # Sanity: perspective is strong enough that affine is noticeably off
    assert affine_err > 1.0, (
        f"perspective too weak — test camera doesn't exercise the path "
        f"(affine_err={affine_err:.3f})"
    )
    # Core assertion: projective is essentially exact (<0.5 px)
    assert projective_err < 0.5, (
        f"projective vertex should be exact under pinhole perspective, "
        f"got err={projective_err:.3f} (affine err {affine_err:.3f})"
    )
    # Degeneracy: finite VPs + tight residual → finite_projective
    assert result.degeneracy == "finite_projective"
    # Residual should be very small (lines meet at one point)
    assert result.residual_norm < 0.01


# ---------------- degeneracy classification gating ----------------


def test_projective_vertex_degeneracy_on_random_jitter_hexagon():
    """If the 6 corners are jittered enough that no consistent cube
    projection fits, the 3 vanishing-point lines won't intersect
    cleanly and the result should classify as 'degenerate'.
    """
    import random
    rng = random.Random(42)
    # Take a canonical hexagon and add HEAVY noise
    base = _canonical_iso_hexagon(center=(500.0, 500.0), radius=200.0)
    noisy = [(p[0] + rng.gauss(0, 60), p[1] + rng.gauss(0, 60)) for p in base]
    from tools.rectify_via_hull_labels import _label_corners_by_position
    corners_by_num = _label_corners_by_position(noisy, "A")
    result = projective_vertex(corners_by_num, "A")
    # With heavy noise, expect non-finite_projective
    assert result.degeneracy in {"degenerate", "near_affine"}


# ---------------- rejects bad input ----------------


def test_projective_vertex_rejects_unknown_side():
    corners = {i: (float(i), float(i)) for i in range(6)}
    with pytest.raises(ValueError, match="unsupported side"):
        projective_vertex(corners, "C")
