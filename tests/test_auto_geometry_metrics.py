"""Unit tests for the auto-geometry evaluator's metric functions."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from tools.evaluate_auto_geometry import (  # noqa: E402
    _best_match_face_iou,
    canonicalize_quad_order,
    mean_corner_error,
    polygon_containment,
    polygon_iou,
    polygon_mask,
)
from tools.propose_geometry_labels import (  # noqa: E402
    _fit_hexagon_to_hull,
    homography_4_points,
    warp,
)


# ---- polygon_iou / containment / mask ----


def test_polygon_iou_identical_squares():
    sq = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert polygon_iou(sq, sq, 120, 120) == pytest.approx(1.0, abs=0.01)


def test_polygon_iou_disjoint():
    a = [(0, 0), (10, 0), (10, 10), (0, 10)]
    b = [(50, 50), (60, 50), (60, 60), (50, 60)]
    assert polygon_iou(a, b, 100, 100) == 0.0


def test_polygon_iou_half_overlap():
    a = [(0, 0), (100, 0), (100, 100), (0, 100)]
    b = [(50, 0), (150, 0), (150, 100), (50, 100)]  # right half overlaps
    iou = polygon_iou(a, b, 200, 100)
    # Intersection = 50x100 = 5000; union = 100x100 + 100x100 - 5000 = 15000
    assert iou == pytest.approx(5000 / 15000, abs=0.02)


def test_polygon_iou_degenerate_returns_zero():
    assert polygon_iou([(0, 0), (10, 10)], [(0, 0), (10, 10)], 100, 100) == 0.0
    assert polygon_iou([], [(0, 0), (10, 0), (10, 10), (0, 10)], 100, 100) == 0.0


def test_polygon_containment_full_inside():
    inner = [(20, 20), (80, 20), (80, 80), (20, 80)]
    outer = [(0, 0), (100, 0), (100, 100), (0, 100)]
    # inner fully inside outer → containment = 1.0
    assert polygon_containment(inner, outer, 120, 120) == pytest.approx(1.0, abs=0.02)


def test_polygon_containment_partial():
    a = [(0, 0), (100, 0), (100, 100), (0, 100)]
    b = [(50, 0), (150, 0), (150, 100), (50, 100)]
    # half of a is inside b
    assert polygon_containment(a, b, 200, 100) == pytest.approx(0.5, abs=0.02)


# ---- mean_corner_error ----


def test_mean_corner_error_identical_quads():
    q = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert mean_corner_error(q, q) == pytest.approx(0.0)


def test_mean_corner_error_offset_quad():
    a = [(0, 0), (10, 0), (10, 10), (0, 10)]
    b = [(3, 4), (13, 4), (13, 14), (3, 14)]  # offset by (3, 4) = 5px each
    assert mean_corner_error(a, b) == pytest.approx(5.0)


def test_mean_corner_error_empty_proposed_is_inf():
    assert mean_corner_error([], [(0, 0)]) == float("inf")


# ---- canonicalize_quad_order ----


def test_canonicalize_quad_order_rhombus_starts_north():
    raw = [(50, 100), (0, 50), (50, 0), (100, 50)]  # S, W, N, E in some order
    ordered = canonicalize_quad_order(raw)
    assert ordered[0] == (50, 0)  # N first


def test_canonicalize_quad_order_preserves_4():
    quad = [(1, 1), (5, 1), (5, 5), (1, 5)]
    assert len(canonicalize_quad_order(quad)) == 4


# ---- homography_4_points / warp ----


def test_homography_identity():
    src = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dst = src
    H = homography_4_points(src, dst)
    for u, v in [(0.5, 0.5), (0.25, 0.75)]:
        x, y = warp(H, u, v)
        assert abs(x - u) < 1e-6 and abs(y - v) < 1e-6


def test_homography_translation_scale():
    src = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dst = [(10, 20), (110, 20), (110, 70), (10, 70)]
    H = homography_4_points(src, dst)
    x, y = warp(H, 0.5, 0.5)
    assert x == pytest.approx(60.0, abs=1e-4)
    assert y == pytest.approx(45.0, abs=1e-4)


# ---- _best_match_face_iou ----


def test_best_match_face_iou_perfect():
    quads = [[(0, 0), (10, 0), (10, 10), (0, 10)]]
    iou = _best_match_face_iou(quads, quads, 20, 20)
    assert iou == pytest.approx(1.0, abs=0.01)


def test_best_match_face_iou_unmatched_proposal_penalised():
    # 2 proposed, 1 truth — best assignment is to match 1 proposed to truth
    # (IoU=1), but score is divided by max(2, 1)=2 → 0.5
    truth = [[(0, 0), (10, 0), (10, 10), (0, 10)]]
    proposed = [truth[0], [(100, 100), (110, 100), (110, 110), (100, 110)]]
    iou = _best_match_face_iou(proposed, truth, 200, 200)
    assert iou == pytest.approx(0.5, abs=0.02)


def test_best_match_face_iou_swapped_labels():
    """Two faces but their labels are swapped — bestMatch should still
    score 1.0 since it ignores labels."""
    quad_a = [(0, 0), (10, 0), (10, 10), (0, 10)]
    quad_b = [(50, 50), (60, 50), (60, 60), (50, 60)]
    proposed = [quad_b, quad_a]  # swapped
    truth = [quad_a, quad_b]
    iou = _best_match_face_iou(proposed, truth, 100, 100)
    assert iou == pytest.approx(1.0, abs=0.02)


# ---- hexagon fit ----


def test_fit_hexagon_to_regular_hexagonal_hull():
    """Feed a regular hexagon — the fit should return its 6 vertices."""
    import math
    cx, cy, r = 100, 100, 50
    verts = [(cx + r * math.cos(math.pi / 2 + i * math.pi / 3),
              cy - r * math.sin(math.pi / 2 + i * math.pi / 3)) for i in range(6)]
    # Add 20 extra points along the hexagon edges to mimic a sampled hull
    hull_points = list(verts)
    for i in range(6):
        a = verts[i]
        b = verts[(i + 1) % 6]
        for t in (0.2, 0.5, 0.8):
            hull_points.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    hex_out = _fit_hexagon_to_hull(hull_points)
    assert hex_out is not None
    assert len(hex_out) == 6
    # Each output vertex should be near one of the 6 input vertices
    for v in hex_out:
        dist = min(((v[0] - vv[0]) ** 2 + (v[1] - vv[1]) ** 2) ** 0.5 for vv in verts)
        assert dist < 1.0


def test_fit_hexagon_to_tiny_hull_returns_none():
    assert _fit_hexagon_to_hull([(0, 0), (1, 0), (0, 1)]) is None


# ---- portability: discover_additional_tasks degrades gracefully on
#      clean environments without /Users/jhuber/Downloads ----


def test_discover_additional_tasks_returns_empty_when_downloads_missing(monkeypatch, tmp_path):
    """Devin PR-#127 review caught this: on a clean CI/VM without the
    local Downloads asset directory, tooling crashed with FileNotFoundError
    instead of returning empty. This regression test pins the fix."""
    from tools import extract_color_samples

    missing = tmp_path / "definitely-does-not-exist"
    monkeypatch.setattr(extract_color_samples, "DOWNLOADS", missing)
    result = extract_color_samples.discover_additional_tasks(set())
    assert result == []


# ---- recognizer_impact_diagnostics (smoke test only — full coverage
#      requires real images; integration-tested by the full sweep) ----


def test_recognizer_impact_diagnostics_smoke():
    """Verify the impact-diagnostics block has the expected schema when
    run against a real labeled pair. This is a smoke test guarding the
    output shape so downstream consumers (Devin's audit, the summary
    table) don't break silently if fields are renamed."""
    from tools.evaluate_auto_geometry import (
        LabelTarget,
        discover_label_targets,
        recognizer_impact_diagnostics,
    )
    from tools.propose_geometry_labels import PROPOSERS

    try:
        targets = discover_label_targets()
    except (FileNotFoundError, OSError):
        # Clean CI/VM environments lack /Users/jhuber/Downloads or
        # runs/labels; the test isn't applicable there.
        pytest.skip("Local image/label assets not available in this environment")
    if not targets:
        pytest.skip("No (setId, side) labels discoverable in this environment")
    set_15 = next((t for t in targets if t.set_id == "15" and t.side == "A"), None)
    if set_15 is None:
        pytest.skip("Set 15 image A label not available in this environment")
    set_15.load()
    proposal = PROPOSERS["recognizer_grids"].propose(set_15)
    impact = recognizer_impact_diagnostics(set_15, proposal)

    # Schema check: all expected keys present with correct types
    expected_keys = {
        "stickerCount", "outsideHullStickerCount", "outsideHullFraction",
        "outsideAllFacesStickerCount", "outsideAllFacesFraction",
        "stickersPerProposedFace", "recognizerBestGridContainment",
        "recognizerGridsAccepted", "recognizerGridsConsidered",
    }
    assert expected_keys.issubset(impact.keys())
    assert isinstance(impact["stickerCount"], int) and impact["stickerCount"] > 0
    assert 0.0 <= impact["outsideHullFraction"] <= 1.0
    assert isinstance(impact["stickersPerProposedFace"], dict)
    # recognizer_grids proposer trivially has all stickers inside its
    # self-derived hull (the hull IS the convex hull of those stickers).
    assert impact["outsideHullStickerCount"] == 0
