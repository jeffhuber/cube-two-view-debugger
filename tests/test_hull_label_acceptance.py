"""Tests for production-shaped hull-label acceptance gates."""
from __future__ import annotations

from tools.hull_label_acceptance import (
    EXPECTED_FACE_SLOTS,
    HullLabelGateThresholds,
    evaluate_hull_label_acceptance,
    side_convention_errors,
    vertex_cloud_spread,
)


def _clean_kwargs():
    return {
        "side": "A",
        "hexagon_corner_count": 6,
        "vertex_estimates": [(100.0, 100.0), (110.0, 100.0), (105.0, 108.0)],
        "rectified_face_slots": EXPECTED_FACE_SLOTS,
        "sticker_score_total": 500.0,
        "sticker_score_per_face": {
            "upper": 130.0,
            "right": 180.0,
            "front": 190.0,
        },
    }


def test_vertex_cloud_spread_is_max_pairwise_distance():
    assert vertex_cloud_spread([(0.0, 0.0), (3.0, 4.0), (0.0, 10.0)]) == 10.0


def test_vertex_cloud_spread_rejects_nonfinite_coordinates():
    assert vertex_cloud_spread([(0.0, 0.0), (float("nan"), 4.0)]) == float("inf")


def test_side_conventions_cover_current_capture_sides():
    assert side_convention_errors("A") == ()
    assert side_convention_errors("B") == ()


def test_unknown_side_is_hard_failure():
    kwargs = _clean_kwargs()
    kwargs["side"] = "C"
    decision = evaluate_hull_label_acceptance(**kwargs)
    assert decision.should_fallback
    assert any("unsupported side" in reason for reason in decision.hard_failures)


def test_clean_row_is_accepted_without_warnings():
    decision = evaluate_hull_label_acceptance(**_clean_kwargs())
    assert decision.accepted
    assert decision.hard_failures == ()
    assert decision.warnings == ()


def test_hexagon_corner_count_must_be_exactly_six():
    kwargs = _clean_kwargs()
    kwargs["hexagon_corner_count"] = 5
    decision = evaluate_hull_label_acceptance(**kwargs)
    assert decision.should_fallback
    assert any("hexagon_corner_count=5" in reason for reason in decision.hard_failures)


def test_vertex_cloud_spread_has_warning_and_hard_thresholds():
    thresholds = HullLabelGateThresholds(
        warn_vertex_cloud_spread_px=20.0,
        max_vertex_cloud_spread_px=40.0,
    )
    kwargs = _clean_kwargs()
    kwargs["vertex_estimates"] = [(0.0, 0.0), (25.0, 0.0), (10.0, 0.0)]
    warning = evaluate_hull_label_acceptance(**kwargs, thresholds=thresholds)
    assert warning.accepted
    assert any("vertex_cloud_spread_px=25.0" in w for w in warning.warnings)

    kwargs["vertex_estimates"] = [(0.0, 0.0), (41.0, 0.0), (10.0, 0.0)]
    hard = evaluate_hull_label_acceptance(**kwargs, thresholds=thresholds)
    assert hard.should_fallback
    assert any("vertex_cloud_spread_px=41.0" in r for r in hard.hard_failures)


def test_sticker_scores_have_total_and_per_face_gates():
    thresholds = HullLabelGateThresholds(
        warn_sticker_score_total=600.0,
        max_sticker_score_total=900.0,
        warn_sticker_score_per_face=250.0,
        max_sticker_score_per_face=400.0,
    )
    kwargs = _clean_kwargs()
    kwargs["sticker_score_total"] = 650.0
    kwargs["sticker_score_per_face"] = {
        "upper": 100.0,
        "right": 260.0,
        "front": 290.0,
    }
    warning = evaluate_hull_label_acceptance(**kwargs, thresholds=thresholds)
    assert warning.accepted
    assert any("sticker_score_total=650.0" in w for w in warning.warnings)
    assert any("sticker_score_worst_face=290.0" in w for w in warning.warnings)

    kwargs["sticker_score_per_face"]["right"] = 401.0
    hard = evaluate_hull_label_acceptance(**kwargs, thresholds=thresholds)
    assert hard.should_fallback
    assert any("sticker_score_worst_face=401.0" in r for r in hard.hard_failures)


def test_nonfinite_sticker_score_is_hard_failure():
    kwargs = _clean_kwargs()
    kwargs["sticker_score_per_face"]["right"] = float("nan")
    decision = evaluate_hull_label_acceptance(**kwargs)
    assert decision.should_fallback
    assert any("sticker_score_worst_face=inf" in r for r in decision.hard_failures)


def test_missing_face_slot_or_score_is_hard_failure():
    kwargs = _clean_kwargs()
    kwargs["rectified_face_slots"] = {"upper", "front"}
    kwargs["sticker_score_per_face"] = {"upper": 100.0, "front": 100.0}
    decision = evaluate_hull_label_acceptance(**kwargs)
    assert decision.should_fallback
    assert any("rectified_face_slots" in reason for reason in decision.hard_failures)
    assert any("missing per-face sticker scores" in reason for reason in decision.hard_failures)
