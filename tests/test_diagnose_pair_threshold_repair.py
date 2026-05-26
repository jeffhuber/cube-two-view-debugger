from __future__ import annotations

from tools.diagnose_pair_threshold_repair import choose_guarded_pair


def test_guarded_pair_keeps_current_when_current_repair_is_valid():
    current = {
        "thresholds": {"A": 224, "B": 192},
        "status": "assembled",
        "summary": {"recommended": {"validState": True, "hamming": 0}},
    }
    aggressive = {
        "thresholds": {"A": 224, "B": 128},
        "status": "assembled",
        "summary": {"recommended": {"validState": True, "hamming": 2}},
    }
    selected = choose_guarded_pair(
        current_combo=current,
        current_eval={"summary": {"recommended": {"validState": True}}},
        aggressive_pair=aggressive,
    )

    assert selected["thresholds"] == {"A": 224, "B": 192}
    assert selected["selectionReason"] == "kept_current_valid_repair"


def test_guarded_pair_switches_when_current_repair_is_invalid():
    current = {
        "thresholds": {"A": 160, "B": 160},
        "status": "assembled",
        "summary": {"recommended": {"validState": False, "hamming": 4}},
    }
    aggressive = {
        "thresholds": {"A": 64, "B": 192},
        "status": "assembled",
        "summary": {"recommended": {"validState": True, "hamming": 0}},
    }
    selected = choose_guarded_pair(
        current_combo=current,
        current_eval={"summary": {"recommended": {"validState": False}}},
        aggressive_pair=aggressive,
    )

    assert selected["thresholds"] == {"A": 64, "B": 192}
    assert selected["selectionReason"] == "current_invalid_selected_best_pair"
