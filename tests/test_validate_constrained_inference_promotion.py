from __future__ import annotations

from tools.validate_constrained_inference_promotion import (
    build_summary,
    evaluate_promotion_gate,
)
from tools.constrained_inference_gate import evaluate_runtime_payload_gate


def _candidate_trace(threshold: int = 160):
    return {
        "threshold": threshold,
        "accepted": True,
        "hardFailures": [],
        "warnings": [],
    }


def _combo(
    *,
    method: str = "canonical_count_repaired",
    rank=(0, 4, 0.0, 4, 0),
    valid: bool = True,
    confidence: str = "high",
    hamming: int | None = 0,
    reason: str = "kept_current_valid_repair",
    thresholds=None,
):
    return {
        "status": "assembled",
        "selectionReason": reason,
        "thresholds": thresholds or {"A": 160, "B": 160},
        "productionRank": list(rank),
        "yawInference": {"accepted": True, "yawQuarterTurns": 0},
        "summary": {
            "recommendedMethod": method,
            "recommended": {
                "validState": valid,
                "countBalanced": valid,
                "confidence": confidence,
                "hamming": hamming,
            },
        },
    }


def _row(*, selected=None, current=None):
    selected_combo = selected or _combo()
    current_combo = current or _combo()
    return {
        "setId": "99",
        "current": {
            "thresholds": current_combo["thresholds"],
            "summary": current_combo["summary"],
        },
        "pairSelected": selected_combo,
        "thresholdDiagnostics": {
            "A": {"candidates": [_candidate_trace(selected_combo["thresholds"]["A"])]},
            "B": {"candidates": [_candidate_trace(selected_combo["thresholds"]["B"])]},
        },
    }


def test_gate_accepts_high_confidence_canonical_count_repair():
    decision = evaluate_promotion_gate(_row())

    assert decision["accepted"] is True
    assert decision["decision"] == "auto_return_candidate"
    assert decision["rejectReasons"] == []


def test_gate_rejects_canonical_count_repair_with_too_many_moves():
    selected = _combo(rank=(0, 11, 0.0, 11, 0))

    decision = evaluate_promotion_gate(_row(selected=selected))

    assert decision["accepted"] is False
    assert "canonical_repair_moves_above_gate" in decision["rejectReasons"]


def test_gate_rejects_low_confidence_repair_even_if_otherwise_valid():
    selected = _combo(confidence="low")

    decision = evaluate_promotion_gate(_row(selected=selected))

    assert decision["accepted"] is False
    assert "recommended_confidence_low_or_missing" in decision["rejectReasons"]


def test_gate_allows_pair_threshold_switch_only_when_current_is_invalid():
    invalid_current = _combo(valid=False, hamming=None)
    selected = _combo(
        reason="current_invalid_selected_best_pair",
        thresholds={"A": 64, "B": 192},
    )

    decision = evaluate_promotion_gate(_row(selected=selected, current=invalid_current))

    assert decision["accepted"] is True
    assert decision["thresholds"]["switched"] is True


def test_gate_rejects_pair_threshold_switch_when_current_is_valid():
    selected = _combo(
        reason="current_invalid_selected_best_pair",
        thresholds={"A": 64, "B": 192},
    )

    decision = evaluate_promotion_gate(_row(selected=selected))

    assert decision["accepted"] is False
    assert "switched_pair_while_current_valid" in decision["rejectReasons"]


def test_gate_accepts_legal_repair_inside_state_delta_and_cost_limits():
    selected = _combo(
        method="guarded_broad_legal_repaired",
        rank=(2, 2, 10.0, 6, 0),
    )

    decision = evaluate_promotion_gate(_row(selected=selected))

    assert decision["accepted"] is True


def test_gate_rejects_legal_repair_outside_state_delta_or_cost_limits():
    selected = _combo(
        method="guarded_broad_legal_repaired",
        rank=(2, 5, 21.0, 7, 0),
    )

    decision = evaluate_promotion_gate(_row(selected=selected))

    assert decision["accepted"] is False
    assert "legal_state_delta_above_gate" in decision["rejectReasons"]
    assert "legal_repair_cost_above_gate" in decision["rejectReasons"]
    assert "legal_repair_changes_above_gate" in decision["rejectReasons"]


def test_gate_rejects_diagnostic_only_broad_repair():
    selected = _combo(method="broad_legal_repaired", rank=(2, 1, 1.0, 1, 0))

    decision = evaluate_promotion_gate(_row(selected=selected))

    assert decision["accepted"] is False
    assert "recommended_method_not_allowed" in decision["rejectReasons"]


def test_build_summary_counts_gate_outcomes():
    rows = [
        _row(selected=_combo(hamming=0)),
        _row(selected=_combo(rank=(0, 11, 0.0, 11, 0), hamming=2)),
    ]

    summary = build_summary(rows)

    assert summary["pairCount"] == 2
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["acceptedExact"] == 1
    assert summary["rejectReasonCounts"] == {"canonical_repair_moves_above_gate": 1}


def test_runtime_payload_gate_uses_app_payload_shape():
    decision = evaluate_runtime_payload_gate(
        repair={
            "status": "assembled",
            "yawQuarterTurns": 0,
            "recommendedMethod": "canonical_count_repaired",
            "recommended": {
                "validState": True,
                "countBalanced": True,
                "confidence": "medium",
                "repairMoveCount": 8,
            },
            "methods": {
                "canonical_count_repaired": {
                    "validState": True,
                    "countBalanced": True,
                    "repairMoveCount": 8,
                }
            },
        },
        pair_threshold_selection={
            "selectionReason": "kept_current_valid_repair",
            "currentRepairValid": True,
            "currentThresholds": {"A": 160, "B": 160},
            "selectedThresholds": {"A": 160, "B": 160},
        },
        side_traces={
            "A": {"status": "accepted", "hard_failures": []},
            "B": {"status": "accepted", "hard_failures": []},
        },
        yaw_inference={"accepted": True},
    )

    assert decision["accepted"] is True
    assert decision["productionRank"] == [0, 8, 0.0, 8, 0]
