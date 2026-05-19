"""Unit tests for the hull-guard logic in evaluate_hybrid_pipeline.

These pin both the positive case (grid with all centers outside the
hull is rejected) AND the negative case (grid with all centers inside
the hull is ACCEPTED even when those centers span multiple physical
faces of the cube). The negative-case test is the documentation of
why the hull guard alone doesn't solve the catastrophic-grid problem:
multi-face grids still pass because the cube hull encloses all 3
visible faces of an iso-projection photo.

See the hull-guard PR body for the full investigation and the
strategic implications (pivot to Path B(2) learned face-quad
regressor).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from tools import evaluate_hybrid_pipeline  # noqa: E402


def _grid_with_centers(centers):
    """Build a minimal FaceGrid-shaped object with a 3x3 of centers
    (passed as a flat list of 9 (x, y) tuples in row-major order)."""
    assert len(centers) == 9
    points = [list(centers[r * 3:(r + 1) * 3]) for r in range(3)]
    return types.SimpleNamespace(
        center_face="U",
        matched_count=9,
        fit_error=2.0,
        points=points,
        cube_hull_inside_count=None,
    )


def _square_hull(x0, y0, x1, y1):
    """Axis-aligned rectangular hull, as a list of corner points."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def test_guard_rejects_grid_with_centers_outside_hull():
    """Positive case: all 9 sticker centers fall OUTSIDE the cube hull.
    Inside count = 0 < HULL_GUARD_INSIDE_MIN. Guard rejects."""
    hull = _square_hull(100, 100, 200, 200)
    grid = _grid_with_centers([
        (300, 300), (310, 300), (320, 300),
        (300, 310), (310, 310), (320, 310),
        (300, 320), (310, 320), (320, 320),
    ])
    inside = evaluate_hybrid_pipeline._grid_inside_hull_count(grid, hull)
    assert inside == 0
    assert inside < evaluate_hybrid_pipeline.HULL_GUARD_INSIDE_MIN


def test_guard_accepts_well_packed_single_face_grid():
    """Positive case: all 9 centers comfortably inside the hull, tightly
    packed in one region — the canonical "valid single-face grid" shape.
    Guard accepts."""
    hull = _square_hull(0, 0, 600, 600)
    # 3x3 grid clustered in the top-left quadrant — clearly a single face
    grid = _grid_with_centers([
        (100, 100), (150, 100), (200, 100),
        (100, 150), (150, 150), (200, 150),
        (100, 200), (150, 200), (200, 200),
    ])
    inside = evaluate_hybrid_pipeline._grid_inside_hull_count(grid, hull)
    assert inside == 9
    assert inside >= evaluate_hybrid_pipeline.HULL_GUARD_INSIDE_MIN


def test_guard_DOES_NOT_reject_multi_face_grid_inside_hull():
    """**The documentation-of-failure-mode test.** A grid whose 9 sticker
    centers are SPREAD ACROSS multiple physical cube faces (i.e., the
    grid is geometrically catastrophic — its "9 stickers" are actually
    cherry-picked from across the cube image) STILL passes the hull
    guard because all 9 points are inside the cube silhouette. The
    cube hull encloses the WHOLE cube in iso-projection (top + left +
    right faces all together), so any 9 points distributed across
    those 3 faces are "inside" by the hull-inside metric.

    This test pins the negative result: hull-inside-count alone cannot
    discriminate single-face vs multi-face grids. The real fix
    requires either bbox-aspect/centroid heuristics (fragile) or the
    production recognizer's `_select_grid_combo` bounded-overlap
    selection (~1-2 days plumbing) or — preferred — a learned
    face-quad regressor (Path B(2) in the strategic priority list).
    """
    # Cube hull: 600x600 hexagonal-ish shape (here simplified to a square,
    # but the point holds: it encloses the entire cube's silhouette).
    hull = _square_hull(0, 0, 600, 600)
    # Multi-face grid: 3x3 of "stickers" spread across the WHOLE cube
    # silhouette — covers the top row of L face + top row of U face +
    # top row of R face, for example. The bbox is ~500x100 (very wide,
    # very flat) — clearly NOT a single cube face. But all 9 points are
    # well inside the 600x600 hull.
    grid = _grid_with_centers([
        (100, 100), (300, 100), (500, 100),  # row 1: spans full cube width
        (100, 150), (300, 150), (500, 150),  # row 2: same
        (100, 200), (300, 200), (500, 200),  # row 3: same
    ])
    inside = evaluate_hybrid_pipeline._grid_inside_hull_count(grid, hull)
    # All 9 inside the hull → guard would ACCEPT this catastrophic grid
    assert inside == 9
    assert inside >= evaluate_hybrid_pipeline.HULL_GUARD_INSIDE_MIN
    # If we extracted the bbox we'd see it's 400x100 (4:1 aspect ratio,
    # 67% of cube hull width) — a heuristic discriminator that the hull
    # guard alone misses.


def test_guard_returns_9_when_no_hull_provided():
    """When the rembg hull can't be computed (empty/None), the guard
    must degrade gracefully — return 9 ("can't validate, accept") rather
    than rejecting all grids."""
    grid = _grid_with_centers([(i, i) for i in range(100, 1000, 100)])
    assert evaluate_hybrid_pipeline._grid_inside_hull_count(grid, []) == 9
    assert evaluate_hybrid_pipeline._grid_inside_hull_count(grid, [(1, 1)]) == 9
    assert evaluate_hybrid_pipeline._grid_inside_hull_count(grid, [(1, 1), (2, 2)]) == 9


def test_partial_inside_count_returns_correct_value():
    """4 of 9 centers inside, 5 outside → inside_count = 4. Below the
    HULL_GUARD_INSIDE_MIN = 7 threshold, so the guard would reject."""
    hull = _square_hull(0, 0, 200, 200)
    grid = _grid_with_centers([
        (50, 50), (100, 50), (500, 50),    # 2 inside, 1 outside
        (50, 100), (100, 100), (500, 100),  # 2 inside, 1 outside
        (500, 500), (500, 600), (500, 700),  # 0 inside, 3 outside
    ])
    inside = evaluate_hybrid_pipeline._grid_inside_hull_count(grid, hull)
    assert inside == 4
    assert inside < evaluate_hybrid_pipeline.HULL_GUARD_INSIDE_MIN
