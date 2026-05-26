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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.rectify_via_hull_labels import (  # noqa: E402
    SILHOUETTE_TO_CORNER,
    _derive_vertex_from_corners,
    _label_corners_by_position,
    _score_rectified_faces,
    choose_best_threshold_candidate,
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


# ---------------- mask threshold candidate selector ----------------


def test_choose_best_threshold_candidate_prefers_accepted_lowest_score():
    candidates = [
        {"threshold": 128, "accepted": True, "sticker_score_total": 900.0},
        {"threshold": 224, "accepted": False, "sticker_score_total": 700.0},
        {"threshold": 192, "accepted": True, "sticker_score_total": 800.0},
    ]

    assert choose_best_threshold_candidate(candidates, accepted_only=False)["threshold"] == 224
    assert choose_best_threshold_candidate(candidates, accepted_only=True)["threshold"] == 192


# ---------------- _choose_hybrid_vertex (affine ↔ projective switch) ----
# 2026-05-25 follow-up to PR #288: the projective vertex is used IFF
# the affine 3-estimate cloud spread exceeds the threshold (240 px by
# default), otherwise affine wins by lower variance from corner noise.


def test_hybrid_projective_threshold_is_resolution_independent():
    """Codex P2 on PR #289 head 48f5a66: the hybrid switch must gate
    on a normalized (resolution-independent) signal so it stays
    stable across processing scales. Pin both the constant name and
    a sensible value range so an accidental revert to raw-px would
    fail this test."""
    from tools.rectify_via_hull_labels import (
        HYBRID_PROJECTIVE_SPREAD_NORM_THRESHOLD,
    )
    # Normalized signal — should be a fraction in (0, 1].
    assert 0.0 < HYBRID_PROJECTIVE_SPREAD_NORM_THRESHOLD <= 1.0
    # Empirically calibrated at 0.26; allow some drift but don't
    # let it silently become a raw-px-shaped number again.
    assert 0.15 <= HYBRID_PROJECTIVE_SPREAD_NORM_THRESHOLD <= 0.35


def test_choose_hybrid_vertex_returns_affine_on_iso_input():
    """On a perfect iso hexagon the 3 affine estimates are tightly
    clustered (spread/diameter ≈ 0). Switch stays on affine."""
    from tools.rectify_via_hull_labels import (
        _choose_hybrid_vertex, _label_corners_by_position,
        HYBRID_PROJECTIVE_SPREAD_NORM_THRESHOLD,
    )
    import math as _m
    pts = [(500 + 200 * _m.cos(_m.radians(d)),
            500 + 200 * _m.sin(_m.radians(d)))
           for d in (-90, -30, 30, 90, 150, 210)]
    corners = _label_corners_by_position(pts, "A")
    vertex, tel = _choose_hybrid_vertex(corners, "A")
    assert tel["vertex_source"] == "affine"
    assert tel["vertex_cloud_spread_norm"] < HYBRID_PROJECTIVE_SPREAD_NORM_THRESHOLD
    assert vertex == tel["affine_vertex"]


def test_choose_hybrid_vertex_switches_to_projective_under_strong_perspective():
    """Build a 3D cube + tilted pinhole camera → project → solve.
    Strong perspective spreads the 3 affine estimates beyond 240 px →
    switch picks projective. Verifies (a) the switch fires, (b) the
    chosen vertex is the projective one, and (c) projective is exact
    on synthetic pinhole input (vertex_err < 1 px vs GT)."""
    import numpy as np
    import math as _m
    cube_3d = {
        "h_x": (-1., 0., 0.), "h_y": (0., -1., 0.), "h_z": (0., 0., -1.),
        "h_xy": (-1., -1., 0.), "h_xz": (-1., 0., -1.), "h_yz": (0., -1., -1.),
        "vertex": (0., 0., 0.),
    }
    side_a_map = {0: "h_xy", 1: "h_x", 2: "h_xz", 3: "h_z", 4: "h_yz", 5: "h_y"}
    cam_pos = np.array([1.4, 1.8, 2.2])
    target = np.array([-0.4, -0.5, -0.3])
    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0., 1., 0.]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, up, -fwd])
    f, cx, cy = 900.0, 500.0, 500.0

    def project(p3):
        pc = R @ (np.array(p3) - cam_pos)
        return (float(-f * pc[0] / pc[2] + cx),
                float(-f * pc[1] / pc[2] + cy))

    corners = {cn: project(cube_3d[side_a_map[cn]]) for cn in range(6)}
    from tools.rectify_via_hull_labels import _choose_hybrid_vertex
    # Use a lower normalized threshold for the unit test — the
    # production 0.26 default is calibrated to real iPhone shots,
    # but a synthetic unit-cube + close pinhole camera produces a
    # smaller normalized spread. The SWITCH BEHAVIOR is what this
    # test pins; the threshold value itself is empirically
    # calibrated and pinned separately.
    vertex, tel = _choose_hybrid_vertex(corners, "A", spread_norm_threshold=0.05)
    assert tel["vertex_cloud_spread_norm"] > 0.05, (
        f"test camera produces only {tel['vertex_cloud_spread_norm']:.4f} "
        f"normalized spread — tune cam_pos so perspective is stronger"
    )
    assert tel["vertex_source"] == "projective"
    assert vertex == tel["projective_vertex"]
    gt = project(cube_3d["vertex"])
    err = _m.hypot(vertex[0] - gt[0], vertex[1] - gt[1])
    assert err < 1.0, (
        f"projective vertex should be exact under pinhole, got {err:.2f}px"
    )


def test_choose_hybrid_vertex_telemetry_is_full_precision():
    """Codex P2 on PR #289 head 540d891: numeric telemetry fields are
    surfaced specifically so callers can route on them via
    `evaluate_hull_label_acceptance`. Rounding them here can flip
    near-threshold decisions (e.g. a raw 0.02504 stored as 0.0250
    won't trip a > 0.025 hard gate). Pin that the raw floats survive
    intact — round only in serialization layers.
    """
    from tools.rectify_via_hull_labels import (
        _choose_hybrid_vertex, _label_corners_by_position,
    )
    import math as _m
    # Tilt the canonical hexagon enough to produce non-trivial
    # values for all telemetry fields.
    pts = [(500 + 200 * _m.cos(_m.radians(d)) + (i % 2) * 3.7,
            500 + 200 * _m.sin(_m.radians(d)) - (i % 3) * 1.3)
           for i, d in enumerate((-90, -30, 30, 90, 150, 210))]
    corners = _label_corners_by_position(pts, "A")
    _vertex, tel = _choose_hybrid_vertex(corners, "A")
    for key in ("vertex_cloud_spread_px", "vertex_cloud_spread_norm",
                "hexagon_diameter_px", "projective_residual_norm"):
        val = tel[key]
        # Full-precision floats won't be exact multiples of any
        # nice rounding interval — if it equals its own round-to-3,
        # we (probably) rounded somewhere.
        assert val != round(val, 3), (
            f"telemetry field {key}={val} appears to be rounded; "
            f"surface full precision so downstream gates can compare "
            f"exactly against their thresholds"
        )


def test_choose_hybrid_vertex_telemetry_carries_both_candidates():
    """Whichever vertex the switch picks, BOTH candidates + decision
    metadata must be surfaced so callers can route on the same signals
    the switch used (e.g. hull_label_acceptance.py)."""
    from tools.rectify_via_hull_labels import (
        _choose_hybrid_vertex, _label_corners_by_position,
    )
    import math as _m
    pts = [(500 + 200 * _m.cos(_m.radians(d)),
            500 + 200 * _m.sin(_m.radians(d)))
           for d in (-90, -30, 30, 90, 150, 210)]
    corners = _label_corners_by_position(pts, "A")
    _vertex, tel = _choose_hybrid_vertex(corners, "A")
    for key in ("affine_vertex", "projective_vertex",
                "vertex_cloud_spread_px", "vertex_cloud_spread_norm",
                "hexagon_diameter_px", "projective_residual_norm",
                "projective_degeneracy", "vertex_source"):
        assert key in tel, f"telemetry missing {key}"


def test_score_rectified_faces_ignores_classifier_env(monkeypatch):
    """The diagnostic report says its score is canonical CIELAB distance.
    Keep that stable even when production classifier experiments are selected
    through CUBE_RECOGNIZER_CLASSIFIER.
    """
    from rubik_recognizer.colors import CLASSIFIER_CANONICAL, classify_rgb_with_mode

    rgb = (144, 72, 49)
    expected = round(classify_rgb_with_mode(rgb, CLASSIFIER_CANONICAL).distance * 27, 2)
    monkeypatch.setenv("CUBE_RECOGNIZER_CLASSIFIER", "knn5_lab_full")

    face = Image.new("RGB", (300, 300), rgb)
    score = _score_rectified_faces({"upper": face, "right": face, "front": face})

    assert score["total_distance"] == expected
