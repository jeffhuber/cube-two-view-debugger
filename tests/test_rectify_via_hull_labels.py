"""Unit tests for `tools/rectify_via_hull_labels.py`.

Pure-function tests of the two deterministic core pieces:
1. `_label_corners_by_position` — hull-position to corner-number mapping
   per side
2. `_derive_vertex_from_corners` — parallelogram-completion vertex

End-to-end (rembg + rectify + score) is exercised by running the CLI
against the canonical corpus; this file pins the math.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.rectify_via_hull_labels import (  # noqa: E402
    SILHOUETTE_TO_CORNER,
    _derive_vertex_from_corners,
    _label_corners_by_position,
)


# ---------------- _label_corners_by_position ----------------


def _canonical_hexagon(center=(500.0, 500.0), radius=200.0):
    """Build a regular hexagon at canonical iso positions:
    TOP, upper-right, lower-right, BOTTOM, lower-left, upper-left."""
    import math as _m
    pts = []
    # Math angles (CCW from +x): -90° = TOP (image y down), -30°, 30°,
    # 90° = BOTTOM, 150°, 210° = upper-left.
    for deg in (-90, -30, 30, 90, 150, 210):
        a = _m.radians(deg)
        pts.append((center[0] + radius * _m.cos(a),
                    center[1] + radius * _m.sin(a)))
    return pts


def test_label_corners_canonical_hexagon_side_a():
    """On a perfect regular hexagon at canonical iso positions, side A
    labeling must yield: 0=TOP, 1=upper-right, 2=lower-right, 3=BOTTOM,
    4=lower-left, 5=upper-left."""
    hex6 = _canonical_hexagon()
    labeled = _label_corners_by_position(hex6, "A")
    # corner_0 should be the TOP point (smallest y)
    assert labeled[0][1] == min(p[1] for p in hex6)
    # corner_3 should be the BOTTOM point (largest y)
    assert labeled[3][1] == max(p[1] for p in hex6)
    # corner_1 (upper-right): smaller y of the 2 right-side points
    right_points = sorted(hex6, key=lambda p: p[0])[-2:]
    assert labeled[1] == min(right_points, key=lambda p: p[1])
    assert labeled[2] == max(right_points, key=lambda p: p[1])
    # corner_5 (upper-left), corner_4 (lower-left)
    left_points = sorted(hex6, key=lambda p: p[0])[:2]
    assert labeled[5] == min(left_points, key=lambda p: p[1])
    assert labeled[4] == max(left_points, key=lambda p: p[1])


def test_label_corners_side_b_uses_different_numbering():
    """Side B labels the SAME silhouette positions with DIFFERENT
    corner numbers (per `FACE_DEFS_BY_SIDE["B"]` derived convention).
    TOP → corner_3 on side B (vs corner_0 on side A); BOTTOM → corner_0
    (vs corner_3); etc. The 3 NEAR silhouette positions (upper-right,
    BOTTOM, upper-left) are the same on both sides but mapped to
    {1, 3, 5} (A) vs {2, 0, 4} (B), reflecting the cube's body-diagonal
    rotation between the two views."""
    hex6 = _canonical_hexagon()
    a = _label_corners_by_position(hex6, "A")
    b = _label_corners_by_position(hex6, "B")
    # Silhouette positions are the same on both sides
    top = min(hex6, key=lambda p: p[1])
    bottom = max(hex6, key=lambda p: p[1])
    # Side A: TOP = corner_0, BOTTOM = corner_3
    assert a[0] == top
    assert a[3] == bottom
    # Side B: TOP = corner_3, BOTTOM = corner_0
    assert b[3] == top
    assert b[0] == bottom
    # NEAR silhouette positions match across sides (upper-right,
    # BOTTOM, upper-left), but corner numbers differ:
    #   Side A NEAR = {1, 3, 5}; side B NEAR = {0, 2, 4}
    assert {a[1], a[3], a[5]} == {b[2], b[0], b[4]}
    # And FAR silhouette positions (TOP, lower-right, lower-left)
    # similarly map to {0,2,4} (A) vs {3,1,5} (B):
    assert {a[0], a[2], a[4]} == {b[3], b[1], b[5]}


def test_label_corners_rejects_wrong_count():
    """Defensive: <6 or >6 corners raises ValueError."""
    with pytest.raises(ValueError):
        _label_corners_by_position(_canonical_hexagon()[:5], "A")
    with pytest.raises(ValueError):
        _label_corners_by_position(_canonical_hexagon() + [(100.0, 100.0)], "A")


def test_label_corners_rejects_unknown_side():
    """A side outside the SILHOUETTE_TO_CORNER table raises KeyError —
    surfaces missing-mapping bugs loudly instead of silently producing
    wrong labels."""
    hex6 = _canonical_hexagon()
    with pytest.raises(KeyError):
        _label_corners_by_position(hex6, "C")


def test_label_corners_handles_mild_tilt():
    """Robustness check: rotate the hexagon by 5° and confirm labels
    still map to the same SET of physical positions (TOP still goes
    to the smallest-y corner, etc.). The CCW corner identities can
    shift but each named silhouette slot still picks the geometrically
    extreme point of its quadrant. Skips the test if rotation pushes
    a point past the rough left/right midline."""
    base = _canonical_hexagon()
    rad = math.radians(5.0)
    cx, cy = 500.0, 500.0
    rotated = [
        (cx + (p[0] - cx) * math.cos(rad) - (p[1] - cy) * math.sin(rad),
         cy + (p[0] - cx) * math.sin(rad) + (p[1] - cy) * math.cos(rad))
        for p in base
    ]
    labeled = _label_corners_by_position(rotated, "A")
    # TOP is still the lowest-y point
    assert labeled[0][1] == min(p[1] for p in rotated)
    # BOTTOM is still the highest-y point
    assert labeled[3][1] == max(p[1] for p in rotated)


# ---------------- _derive_vertex_from_corners ----------------


def test_derive_vertex_exact_for_iso_canonical_hexagon():
    """In canonical iso projection of a perfect cube, the front vertex
    sits exactly at the centroid of the 6 hexagon corners. The
    parallelogram-completion math must produce the centroid (within
    float precision) when fed a perfect hexagon."""
    hex6 = _canonical_hexagon(center=(500.0, 500.0), radius=200.0)
    labeled = _label_corners_by_position(hex6, "A")
    vertex, estimates = _derive_vertex_from_corners(labeled, "A")
    # All 3 face estimates should agree exactly
    assert all(
        abs(e[0] - estimates[0][0]) < 1e-6
        and abs(e[1] - estimates[0][1]) < 1e-6
        for e in estimates
    )
    # Mean equals centroid of hexagon (500, 500)
    centroid_x = sum(p[0] for p in hex6) / 6.0
    centroid_y = sum(p[1] for p in hex6) / 6.0
    assert abs(vertex[0] - centroid_x) < 1e-6
    assert abs(vertex[1] - centroid_y) < 1e-6


def test_derive_vertex_side_b_matches_side_a_on_canonical_hexagon():
    """On a perfect iso hexagon, side A and side B should derive the
    SAME vertex (the centroid) — they're the same physical cube viewed
    from opposite body-diagonal corners, but the vertex IS the centroid
    of the silhouette in both cases."""
    hex6 = _canonical_hexagon()
    v_a, _ = _derive_vertex_from_corners(_label_corners_by_position(hex6, "A"), "A")
    v_b, _ = _derive_vertex_from_corners(_label_corners_by_position(hex6, "B"), "B")
    assert abs(v_a[0] - v_b[0]) < 1e-6
    assert abs(v_a[1] - v_b[1]) < 1e-6


def test_derive_vertex_handles_perspective_perturbation():
    """Real iPhone shots have mild perspective: the hexagon corners
    deviate ~10-30 px from canonical iso positions. Confirm the
    parallelogram-completion still produces a sensible vertex (within
    50 px of centroid) under random perturbation."""
    import random
    random.seed(42)
    base = _canonical_hexagon(center=(500.0, 500.0), radius=200.0)
    perturbed = [(p[0] + random.gauss(0, 10), p[1] + random.gauss(0, 10))
                 for p in base]
    labeled = _label_corners_by_position(perturbed, "A")
    vertex, estimates = _derive_vertex_from_corners(labeled, "A")
    centroid_x = sum(p[0] for p in perturbed) / 6.0
    centroid_y = sum(p[1] for p in perturbed) / 6.0
    # 3 estimates now disagree a bit due to perturbation
    spreads_x = max(e[0] for e in estimates) - min(e[0] for e in estimates)
    spreads_y = max(e[1] for e in estimates) - min(e[1] for e in estimates)
    # Spread under 60 px on 10-px-sigma noise — reasonable bound.
    assert spreads_x < 60
    assert spreads_y < 60
    # Mean still within 30 px of centroid.
    assert abs(vertex[0] - centroid_x) < 30
    assert abs(vertex[1] - centroid_y) < 30


# ---------------- mapping table sanity ----------------


def test_silhouette_to_corner_table_covers_both_sides():
    """The per-side mapping table must cover both A and B with all 6
    silhouette positions assigned to distinct corner numbers 0-5."""
    for side in ("A", "B"):
        assert side in SILHOUETTE_TO_CORNER
        mapping = SILHOUETTE_TO_CORNER[side]
        assert set(mapping.keys()) == {
            "top", "upper_right", "lower_right",
            "bottom", "lower_left", "upper_left",
        }
        assert set(mapping.values()) == set(range(6))


def test_silhouette_to_corner_consistent_with_face_defs():
    """The per-side silhouette-to-corner mapping must produce
    geometrically valid face_quads — specifically: for each face, the
    3 hexagon corners assigned to that face must be 3 adjacent corners
    around the hexagon (not 3 random non-adjacent ones).

    Check by walking each side's FACE_DEFS and confirming the 3
    non-vertex corner numbers correspond to consecutive silhouette
    positions (going CCW: top → upper_right → lower_right → bottom
    → lower_left → upper_left → top)."""
    from tools.corner_conventions import FACE_DEFS_BY_SIDE
    ccw_positions = [
        "upper_left", "top", "upper_right",
        "lower_right", "bottom", "lower_left",
    ]
    for side in ("A", "B"):
        # Inverse mapping: corner_number → silhouette_position
        inv = {v: k for k, v in SILHOUETTE_TO_CORNER[side].items()}
        for slot, names in FACE_DEFS_BY_SIDE[side].items():
            corner_nums = [int(n.split("_")[1]) for n in names if n != "vertex"]
            positions = [inv[n] for n in corner_nums]
            # The 3 positions must be 3 consecutive items in ccw_positions
            # (allowing wrap-around).
            ccw_indices = sorted(ccw_positions.index(p) for p in positions)
            diffs = [
                (ccw_indices[1] - ccw_indices[0]) % 6,
                (ccw_indices[2] - ccw_indices[1]) % 6,
                (ccw_indices[0] - ccw_indices[2]) % 6,
            ]
            assert sorted(diffs) == [1, 1, 4], (
                f"side {side} face {slot} corners {corner_nums} → positions "
                f"{positions} → not 3-consecutive in CCW order"
            )
