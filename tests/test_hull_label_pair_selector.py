from __future__ import annotations

from tools.hull_label_pair_selector import (
    choose_guarded_pair,
    choose_pair_by_production_signals,
    repair_rank,
    repair_valid,
)


def _app_eval(*, valid: bool, moves: int, yaw: int = 0):
    return {
        "status": "assembled",
        "yawQuarterTurns": yaw,
        "repair": {
            "methods": {
                "canonical_count_repaired": {
                    "validState": valid,
                    "countBalanced": True,
                    "repairMoveCount": moves,
                },
            },
            "recommended": {
                "validState": valid,
                "repairMoveCount": moves,
            },
        },
    }


def _diagnostic_payload(*, valid: bool, moves: int, yaw: int = 0):
    return {
        "yawQuarterTurns": yaw,
        "methods": {
            "canonical_count_repaired": {
                "validState": valid,
                "countBalanced": True,
                "repairMoveCount": moves,
            },
        },
        "recommended": {
            "validState": valid,
            "repairMoveCount": moves,
        },
    }


def _legal_payload(*, conservative_delta=None, conservative_changes=None, guarded_delta=None, guarded_changes=None):
    methods = {
        "canonical_count_repaired": {
            "validState": False,
            "countBalanced": True,
            "repairMoveCount": 4,
        }
    }
    if conservative_delta is not None or conservative_changes is not None:
        methods["conservative_legal_repaired"] = {
            "validState": True,
            "repairCost": 7.0,
            "repairChanges": conservative_changes,
            "stateDeltaFromCanonical": {"count": conservative_delta},
        }
    if guarded_delta is not None or guarded_changes is not None:
        methods["guarded_broad_legal_repaired"] = {
            "validState": True,
            "repairCost": 8.0,
            "repairChanges": guarded_changes,
            "stateDeltaFromCanonical": {"count": guarded_delta},
        }
    return {"methods": methods}


def test_repair_valid_accepts_app_and_diagnostic_payload_shapes():
    assert repair_valid(_app_eval(valid=True, moves=4))
    assert repair_valid(_diagnostic_payload(valid=True, moves=4))
    assert not repair_valid(_app_eval(valid=False, moves=4))
    assert not repair_valid({"summary": {"recommended": {"validState": False}}})


def test_repair_rank_matches_app_and_diagnostic_payload_shapes():
    assert repair_rank(_app_eval(valid=True, moves=4, yaw=2)) == (0, 4, 0.0, 4, 2)
    assert repair_rank(_diagnostic_payload(valid=True, moves=4, yaw=2)) == (0, 4, 0.0, 4, 2)


def test_repair_rank_prefers_state_delta_for_legal_repair_candidates():
    conservative = _legal_payload(conservative_delta=2, conservative_changes=6)
    guarded = _legal_payload(guarded_delta=3, guarded_changes=1)

    assert repair_rank(conservative) == (1, 2, 7.0, 6, 0)
    assert repair_rank(guarded) == (2, 3, 8.0, 1, 0)


def test_choose_pair_by_production_signals_ranks_without_ground_truth():
    weak = {
        "thresholds": {"A": 224, "B": 224},
        "evaluation": _app_eval(valid=True, moves=6),
        "stickerScoreTotal": 100.0,
    }
    strong = {
        "thresholds": {"A": 128, "B": 128},
        "evaluation": _app_eval(valid=True, moves=2),
        "stickerScoreTotal": 900.0,
    }

    assert choose_pair_by_production_signals([weak, strong]) is strong


def test_choose_guarded_pair_falls_back_to_current_for_app_when_no_assembled_candidate():
    current = {
        "thresholds": {"A": 160, "B": 160},
        "evaluation": {"status": "yaw_unavailable", "repair": {"recommended": {"validState": False}}},
    }

    selected = choose_guarded_pair(
        current_combo=current,
        candidates=[current],
        fallback_to_current_without_alternative=True,
    )

    assert selected["thresholds"] == {"A": 160, "B": 160}
    assert selected["selectionReason"] == "kept_current_no_assembled_alternative"
