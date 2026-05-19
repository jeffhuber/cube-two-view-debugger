"""Unit tests for the pure-math helpers in tools.amg_face_refiner.

These tests DON'T require SAM2 to be installed — they exercise only:
  * predict_sticker_positions_from_quad: bilinear @ (1/6, 3/6, 5/6)
  * extrapolate_face_corners_from_4_outer: 1.5·outer − 0.5·center
  * constrained_kmeans: capacity-bounded k-means

The SAM2-dependent paths (`get_amg_predictor`, `amg_sticker_centroids`,
`amg_refine_quads_per_slot`) require the SAM2 weights at
/tmp/sam2_checkpoints/sam2_hiera_tiny.pt and scipy; these are NOT
exercised here so the test suite stays fast and CI-reproducible.

See tools/AMG_FACE_REFINER.md for the research findings and how to
manually reproduce the SAM2-dependent paths.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.amg_face_refiner import (  # noqa: E402
    constrained_kmeans,
    extrapolate_face_corners_from_4_outer,
    predict_sticker_positions_from_quad,
)


# ---------------------------------------------------------------------------
# predict_sticker_positions_from_quad — bilinear @ face fractions
# ---------------------------------------------------------------------------

def test_predict_sticker_positions_unit_square_as_set():
    """A unit square produces 9 centroids at the 9 (face-fraction)
    cells (1/6, 3/6, 5/6) × (1/6, 3/6, 5/6). The internal canonical
    CW-from-N re-ordering may rotate which centroid is 'first', so
    we test the SET of 9 positions, not their order."""
    quad = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    stickers = predict_sticker_positions_from_quad(quad)
    assert len(stickers) == 9
    expected_set = {
        (round(x, 6), round(y, 6))
        for x in (1/6, 3/6, 5/6)
        for y in (1/6, 3/6, 5/6)
    }
    actual_set = {(round(x, 6), round(y, 6)) for x, y in stickers}
    assert actual_set == expected_set


def test_predict_sticker_positions_scaled_axis_aligned_as_set():
    """A 300×300 axis-aligned square produces 9 centroids at
    (50, 50), (150, 50), ..., (250, 250) — the canonical 3×3 sampling
    grid that `rectify_face` and `extract_stickers_from_rectified` use."""
    quad = [(0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)]
    stickers = predict_sticker_positions_from_quad(quad)
    expected_set = {(x, y) for x in (50, 150, 250) for y in (50, 150, 250)}
    actual_set = {(round(x), round(y)) for x, y in stickers}
    assert actual_set == expected_set


def test_predict_sticker_positions_canonical_reorder_idempotent():
    """Different input orderings of the same 4 corners produce the SAME
    9 sticker positions — canonical_corner_order normalizes inputs."""
    quad_cw = [(0, 0), (10, 0), (10, 10), (0, 10)]
    quad_shuffled = [(10, 10), (0, 0), (10, 0), (0, 10)]
    s_cw = predict_sticker_positions_from_quad(quad_cw)
    s_shuffled = predict_sticker_positions_from_quad(quad_shuffled)
    # Both should produce 9 stickers at the same set of positions
    assert set((round(x, 6), round(y, 6)) for x, y in s_cw) == \
           set((round(x, 6), round(y, 6)) for x, y in s_shuffled)


# ---------------------------------------------------------------------------
# extrapolate_face_corners_from_4_outer
# ---------------------------------------------------------------------------

def test_extrapolate_axis_aligned_square():
    """Outer stickers of a unit square at (1/6, 1/6), (5/6, 1/6),
    (5/6, 5/6), (1/6, 5/6) — extrapolation should recover the unit
    square's 4 corners (0,0), (1,0), (1,1), (0,1)."""
    outer = [(1/6, 1/6), (5/6, 1/6), (5/6, 5/6), (1/6, 5/6)]
    corners = extrapolate_face_corners_from_4_outer(outer)
    expected = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for actual, exp in zip(corners, expected):
        assert abs(actual[0] - exp[0]) < 1e-9
        assert abs(actual[1] - exp[1]) < 1e-9


def test_extrapolate_scaled_square():
    """Extrapolate from 4 outer stickers of a 300×300 square — should
    recover the (0,0), (300,0), (300,300), (0,300) corners."""
    outer = [(50, 50), (250, 50), (250, 250), (50, 250)]
    corners = extrapolate_face_corners_from_4_outer(outer)
    expected = [(0, 0), (300, 0), (300, 300), (0, 300)]
    for actual, exp in zip(corners, expected):
        assert abs(actual[0] - exp[0]) < 1e-6
        assert abs(actual[1] - exp[1]) < 1e-6


def test_extrapolate_rotated_parallelogram():
    """A parallelogram face — extrapolation recovers it exactly (the
    formula is exact for parallelograms by symmetry of the 3×3 grid)."""
    # Parallelogram corners: (0,0), (10, 1), (12, 8), (2, 7)
    Q = [np.array([0.0, 0.0]), np.array([10.0, 1.0]),
         np.array([12.0, 8.0]), np.array([2.0, 7.0])]
    # Compute outer stickers via bilinear @ (1/6, 1/6), (5/6, 1/6),
    # (5/6, 5/6), (1/6, 5/6)
    fracs = [(1/6, 1/6), (5/6, 1/6), (5/6, 5/6), (1/6, 5/6)]
    outer = []
    for u, v in fracs:
        p = ((1 - u) * (1 - v) * Q[0] + u * (1 - v) * Q[1]
             + u * v * Q[2] + (1 - u) * v * Q[3])
        outer.append(tuple(p.tolist()))
    corners = extrapolate_face_corners_from_4_outer(outer)
    expected_corners = [tuple(q.tolist()) for q in Q]
    for actual, exp in zip(corners, expected_corners):
        assert abs(actual[0] - exp[0]) < 1e-6, f"x mismatch: {actual} vs {exp}"
        assert abs(actual[1] - exp[1]) < 1e-6, f"y mismatch: {actual} vs {exp}"


def test_extrapolate_rejects_wrong_count():
    """Defensive: passing 3 or 5 outer stickers raises ValueError —
    the formula requires exactly 4."""
    with pytest.raises(ValueError, match="expected 4 outer stickers"):
        extrapolate_face_corners_from_4_outer([(0, 0), (1, 0), (1, 1)])
    with pytest.raises(ValueError, match="expected 4 outer stickers"):
        extrapolate_face_corners_from_4_outer(
            [(0, 0), (1, 0), (1, 1), (0, 1), (2, 2)]
        )


# ---------------------------------------------------------------------------
# constrained_kmeans
# ---------------------------------------------------------------------------

def test_constrained_kmeans_balanced_3x9():
    """Three well-separated clusters of 9 points each — k-means with
    max_per_cluster=9 produces a balanced 9/9/9 partition with cluster
    centers near the input cluster centroids."""
    np.random.seed(0)
    # 3 clusters at (0, 0), (100, 0), (50, 100)
    cluster_centers = np.array([[0, 0], [100, 0], [50, 100]], dtype=float)
    points = np.vstack([
        cluster_centers[c] + np.random.randn(9, 2) * 5.0
        for c in range(3)
    ])
    init = np.array([[5, 5], [95, 5], [55, 95]], dtype=float)
    assigned, final_centers = constrained_kmeans(
        points, init, max_per_cluster=9,
    )
    # Each cluster should get exactly 9 points
    counts = [sum(1 for a in assigned if a == c) for c in range(3)]
    assert counts == [9, 9, 9]
    # Final centers should be near the true centers — std=5, so the
    # sample mean is within ~5/sqrt(9) ≈ 1.7 px standard error of the
    # true center. Allow 8 px tolerance to absorb the constraint's
    # greedy assignment slightly biasing centers.
    for c in range(3):
        d = np.linalg.norm(final_centers[c] - cluster_centers[c])
        assert d < 8.0, f"cluster {c} center off by {d:.1f} px"


def test_constrained_kmeans_caps_at_max_per_cluster():
    """If one initial center is near 12 points, the capacity constraint
    forces 3 points to the next-nearest cluster."""
    # 12 points around (0, 0), 6 around (100, 100)
    points = np.vstack([
        np.zeros((12, 2)) + np.random.RandomState(1).randn(12, 2) * 1.0,
        np.array([[100, 100]] * 6) + np.random.RandomState(2).randn(6, 2) * 1.0,
    ])
    init = np.array([[0, 0], [100, 100]], dtype=float)
    assigned, _ = constrained_kmeans(points, init, max_per_cluster=9)
    counts = [sum(1 for a in assigned if a == c) for c in range(2)]
    # Cluster 0 capped at 9, the 3 overflow points must go to cluster 1
    assert counts == [9, 9]


def test_constrained_kmeans_converges_in_few_iter():
    """Well-separated clusters converge in few iterations (centers stop
    moving). The early-stop check `np.allclose(..., atol=1.0)` short-circuits."""
    np.random.seed(42)
    points = np.vstack([
        np.array([[0, 0]] * 9) + np.random.randn(9, 2) * 0.1,
        np.array([[1000, 0]] * 9) + np.random.randn(9, 2) * 0.1,
        np.array([[500, 1000]] * 9) + np.random.randn(9, 2) * 0.1,
    ])
    init = np.array([[10, 10], [990, 10], [510, 990]], dtype=float)
    assigned, final_centers = constrained_kmeans(
        points, init, max_per_cluster=9, n_iter=50,
    )
    # All 27 points cleanly partitioned 9/9/9
    counts = [sum(1 for a in assigned if a == c) for c in range(3)]
    assert counts == [9, 9, 9]
