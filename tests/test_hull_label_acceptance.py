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


# -------- projective_residual_norm gate (PR #288 follow-up) ----------
# Added 2026-05-25 — surfaces 30_A-class bad-hull rows that spread
# and sticker gates miss.


def test_projective_residual_norm_omitted_keeps_existing_behavior():
    """If caller doesn't supply projective_residual_norm (legacy
    callers from before PR #288), the gate must still accept clean
    rows — backward-compat shim."""
    kwargs = _clean_kwargs()
    decision = evaluate_hull_label_acceptance(**kwargs)
    assert decision.accepted
    assert decision.warnings == ()
    assert "projective_residual_norm" not in decision.metrics


def test_projective_residual_norm_below_warn_is_accepted_quiet():
    kwargs = _clean_kwargs()
    decision = evaluate_hull_label_acceptance(
        **kwargs, projective_residual_norm=0.005,
    )
    assert decision.accepted
    assert decision.warnings == ()
    assert decision.metrics["projective_residual_norm"] == 0.005


def test_projective_residual_norm_in_warn_band_warns_but_accepts():
    kwargs = _clean_kwargs()
    # 0.022 is between warn (0.018) and hard (0.025) → accept with warning.
    # Also verify 0.0199 (the cited borderline 30_B residual) triggers
    # the warn band — Codex P3 on PR #289 head 540d891.
    decision = evaluate_hull_label_acceptance(
        **kwargs, projective_residual_norm=0.022,
    )
    assert decision.accepted
    assert any("projective_residual_norm=0.0220" in w for w in decision.warnings)

    decision_30b = evaluate_hull_label_acceptance(
        **kwargs, projective_residual_norm=0.0199,
    )
    assert decision_30b.accepted
    assert any("0.0199" in w for w in decision_30b.warnings), (
        "30_B borderline residual 0.0199 must trigger warn — if not, "
        "warn_projective_residual_norm has drifted too high"
    )


def test_projective_residual_norm_above_hard_fails():
    """30_A's 0.0315 is the corpus max and the canonical bad-hull row.
    The hard gate must fire on it."""
    kwargs = _clean_kwargs()
    decision = evaluate_hull_label_acceptance(
        **kwargs, projective_residual_norm=0.0315,
    )
    assert decision.should_fallback
    assert any("projective_residual_norm=0.0315" in r for r in decision.hard_failures)


def test_projective_residual_norm_nonfinite_is_hard_failure():
    """NaN / inf residual indicates a degenerate fit — fallback."""
    kwargs = _clean_kwargs()
    decision = evaluate_hull_label_acceptance(
        **kwargs, projective_residual_norm=float("inf"),
    )
    assert decision.should_fallback
    assert any("projective_residual_norm" in r for r in decision.hard_failures)
