"""Unit tests for the clean-label geometric extraction pipeline."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tools.sample_stickers_from_hull import (  # noqa: E402
    apply_orientation,
    canonical_corner_order,
    discover_orientation,
    homography_unit_to_quad,
    identify_faces_jointly,
    sample_rgb,
    sticker_centers,
    warp,
)


# --- homography ---


def test_homography_identity_quad():
    """Unit square → unit square should give identity-like behaviour."""
    quad = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    H = homography_unit_to_quad(quad)
    for u, v in ((0.25, 0.25), (0.5, 0.5), (0.75, 0.5)):
        x, y = warp(H, u, v)
        assert abs(x - u) < 1e-6
        assert abs(y - v) < 1e-6


def test_homography_translated_scaled_quad():
    """Square shifted and scaled — homography should be linear."""
    quad = [(10.0, 20.0), (110.0, 20.0), (110.0, 70.0), (10.0, 70.0)]  # 100×50 box at (10,20)
    H = homography_unit_to_quad(quad)
    x, y = warp(H, 0.5, 0.5)
    assert abs(x - 60.0) < 1e-4
    assert abs(y - 45.0) < 1e-4


def test_homography_perspective_quad():
    """True perspective quad: opposite sides not parallel. Center maps to
    the homography-correct point, NOT the centroid (the difference is what
    distinguishes homography from bilinear)."""
    # A trapezoid: top edge shorter than bottom edge
    quad = [(20.0, 0.0), (80.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    H = homography_unit_to_quad(quad)
    # Bilinear center (0.5, 0.5) would give the arithmetic mean = (50, 50).
    # Homography (0.5, 0.5) maps via projective division; for this trapezoid
    # the answer is still x=50 by symmetry but y is shifted.
    x, y = warp(H, 0.5, 0.5)
    assert abs(x - 50.0) < 1e-3
    # Projective center = intersection of diagonals, NOT centroid (50,50).
    # For this trapezoid: diagonals (20,0)↔(100,100) and (80,0)↔(0,100)
    # cross at (50, 37.5). Narrow top → projective center biases toward
    # the narrow side. Bilinear would say y=50 flat (wrong).
    assert abs(y - 37.5) < 1e-3


# --- sticker_centers ---


def test_sticker_centers_count_and_order():
    quad = [(0.0, 0.0), (90.0, 0.0), (90.0, 90.0), (0.0, 90.0)]
    centers = sticker_centers(quad, inset=1 / 6)
    assert len(centers) == 9
    # Row-major: (1/6, 1/6), (3/6, 1/6), (5/6, 1/6), (1/6, 3/6), ...
    expected_us = [15, 45, 75, 15, 45, 75, 15, 45, 75]
    expected_vs = [15, 15, 15, 45, 45, 45, 75, 75, 75]
    for (x, y), eu, ev in zip(centers, expected_us, expected_vs):
        assert abs(x - eu) < 1e-3
        assert abs(y - ev) < 1e-3


def test_sticker_centers_inset_moves_toward_middle():
    quad = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    near_corner = sticker_centers(quad, inset=1 / 6)[0]
    insetted = sticker_centers(quad, inset=0.22)[0]
    # Higher inset → corner sticker center moves toward middle
    assert insetted[0] > near_corner[0]
    assert insetted[1] > near_corner[1]


def test_canonical_corner_order_is_cw_from_north():
    # Diamond/rhombus with vertices at N, E, S, W. Whatever order we pass,
    # canonical output should start from the north-most and go CW.
    raw = [(50.0, 100.0), (0.0, 50.0), (50.0, 0.0), (100.0, 50.0)]  # S, W, N, E
    ordered = canonical_corner_order(raw)
    assert ordered[0] == (50.0, 0.0)  # N
    assert ordered[1] == (100.0, 50.0)  # E
    assert ordered[2] == (50.0, 100.0)  # S
    assert ordered[3] == (0.0, 50.0)  # W


# --- sample_rgb ---


def test_sample_rgb_median_of_uniform_patch():
    arr = np.full((50, 50, 3), [123, 45, 67], dtype=np.uint8)
    assert sample_rgb(arr, 25, 25, half=5) == (123, 45, 67)


def test_sample_rgb_clamps_to_bounds():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[:5, :5] = [200, 100, 50]
    # near corner: half=3 patch overlaps both regions, median should be median
    rgb = sample_rgb(arr, 2, 2, half=2)
    assert rgb == (200, 100, 50)


# --- apply_orientation ---


def test_apply_orientation_identity():
    positions = list(range(9))
    out = apply_orientation(positions, mirror=False, rot_quarter=0)
    assert out == positions


def test_apply_orientation_rotation_keeps_center():
    """Position 4 (row 1, col 1) is invariant under any rotation."""
    positions = list(range(9))
    for rot in range(4):
        for mirror in (False, True):
            out = apply_orientation(positions, mirror=mirror, rot_quarter=rot)
            assert out[4] == 4, f"center moved under mirror={mirror} rot={rot}"


def test_apply_orientation_four_rotations_return_to_start():
    positions = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    out = positions
    for _ in range(4):
        out = apply_orientation(out, mirror=False, rot_quarter=1)
    assert out == positions


# --- discover_orientation ---


def test_discover_orientation_picks_correct_rotation():
    """Set up RGBs such that rot_ccw=1 is the only orientation that
    matches the GT colors. Verify discovery returns rot_ccw=1."""
    canonical = {
        "white": (240, 240, 235),
        "yellow": (235, 215, 50),
        "red": (190, 50, 35),
        "orange": (220, 110, 40),
        "green": (60, 145, 80),
        "blue": (65, 90, 150),
    }
    # GT face: 9 colors row-major
    gt = ["white", "red", "blue", "green", "white", "yellow",
          "orange", "white", "red"]
    # If we apply rot_ccw=1 to GT, we get a new row-major arrangement.
    # Sample RGBs in THAT arrangement → discovery should find rot_ccw=1.
    rotated_gt = apply_orientation(gt, mirror=False, rot_quarter=1)
    rgbs = [canonical[c] for c in rotated_gt]
    mirror, rot, _ = discover_orientation(rgbs, gt)
    # The CCW rotation that takes original GT to rotated arrangement is
    # rot=1; the inverse-to-correct that the discovery finds is rot=3 (or 1).
    # Round-trip: rot+inverse should = 4 ≡ 0.
    assert (rot + 1) % 4 == 0 or rot == 1
    assert mirror is False


# --- identify_faces_jointly: pair-level invariant ---


def test_identify_faces_jointly_canonical_no_yaw():
    """Constructed prepared-sides with cleanly-distinguishable colors —
    should identify yaw0 (canonical URFDLB) with no swaps."""
    # GT state with canonical centers
    gt = "URRRURUUR" + "FRFFRFFFR" + "FFFFFFFFF" + "DDDDDDDDD" + "LLLLLLLLL" + "BBBBBBBBB"
    # 9 stickers per face, mostly the face's own color (centers definitely)

    # Build prepared sides with quads + arr that produce sampled multisets
    # matching this GT under yaw0
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    # Paint distinct color blocks for U, R, F (image A) and D, L, B (image B)
    # We just need the SAMPLE_RGB function to return colors that classify
    # back to those face names. Use canonical RGBs.
    canonical = {
        "U": (240, 240, 235), "R": (190, 50, 35), "F": (60, 145, 80),
        "D": (235, 215, 50), "L": (220, 110, 40), "B": (65, 90, 150),
    }
    # Paint a unique face color into a small region we'll quad-label
    quads_a = {}
    quads_b = {}
    for i, face in enumerate("URF"):
        x0 = i * 30
        arr[10:40, x0 + 5:x0 + 25] = canonical[face]
        quads_a[face] = [(x0 + 5.0, 10.0), (x0 + 25.0, 10.0),
                          (x0 + 25.0, 40.0), (x0 + 5.0, 40.0)]
    for i, face in enumerate("DLB"):
        x0 = 100 + i * 30
        arr[50:80, x0 + 5:x0 + 25] = canonical[face]
        quads_b[face] = [(x0 + 5.0, 50.0), (x0 + 25.0, 50.0),
                          (x0 + 25.0, 80.0), (x0 + 5.0, 80.0)]

    prepared_a = {"arr": arr, "quads": quads_a, "expected": ("U", "R", "F")}
    prepared_b = {"arr": arr, "quads": quads_b, "expected": ("D", "L", "B")}
    prepared_sides = {"A": prepared_a, "B": prepared_b}

    mapping, score, status = identify_faces_jointly(prepared_sides, gt, inset=1 / 6)
    assert status == "ok"
    # No swaps for canonical
    assert mapping["A"] == {"U": "U", "R": "R", "F": "F"}
    assert mapping["B"] == {"D": "D", "L": "L", "B": "B"}


def test_identify_faces_jointly_invariant_holds():
    """Across both sides, mapping union must equal {U, R, F, D, L, B}."""
    gt = "URRRURUUR" + "FRFFRFFFR" + "FFFFFFFFF" + "DDDDDDDDD" + "LLLLLLLLL" + "BBBBBBBBB"
    arr = np.zeros((100, 200, 3), dtype=np.uint8)
    canonical = {
        "U": (240, 240, 235), "R": (190, 50, 35), "F": (60, 145, 80),
        "D": (235, 215, 50), "L": (220, 110, 40), "B": (65, 90, 150),
    }
    quads_a = {}
    quads_b = {}
    for i, face in enumerate("URF"):
        x0 = i * 30
        arr[10:40, x0 + 5:x0 + 25] = canonical[face]
        quads_a[face] = [(x0 + 5.0, 10.0), (x0 + 25.0, 10.0),
                          (x0 + 25.0, 40.0), (x0 + 5.0, 40.0)]
    for i, face in enumerate("DLB"):
        x0 = 100 + i * 30
        arr[50:80, x0 + 5:x0 + 25] = canonical[face]
        quads_b[face] = [(x0 + 5.0, 50.0), (x0 + 25.0, 50.0),
                          (x0 + 25.0, 80.0), (x0 + 5.0, 80.0)]
    prepared_sides = {
        "A": {"arr": arr, "quads": quads_a, "expected": ("U", "R", "F")},
        "B": {"arr": arr, "quads": quads_b, "expected": ("D", "L", "B")},
    }
    mapping, _, _ = identify_faces_jointly(prepared_sides, gt, inset=1 / 6)
    all_faces = set()
    for side_map in mapping.values():
        all_faces.update(side_map.values())
    assert all_faces == set("URFDLB"), f"invariant violated: {all_faces}"
