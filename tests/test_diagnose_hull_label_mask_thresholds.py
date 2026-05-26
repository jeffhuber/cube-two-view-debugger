from __future__ import annotations

from tools.diagnose_hull_label_mask_thresholds import build_summary, choose_best_candidate


def test_choose_best_candidate_can_require_accepted_rows():
    candidates = [
        {"threshold": 128, "accepted": True, "sticker_score_total": 900.0},
        {"threshold": 224, "accepted": False, "sticker_score_total": 700.0},
        {"threshold": 192, "accepted": True, "sticker_score_total": 800.0},
    ]

    assert choose_best_candidate(candidates, accepted_only=False)["threshold"] == 224
    assert choose_best_candidate(candidates, accepted_only=True)["threshold"] == 192


def test_build_summary_counts_thresholds_and_score_improvement():
    rows = [{
        "setId": "70",
        "sides": {
            "A": {
                "bestAnyThreshold": 224,
                "bestAcceptedThreshold": 192,
                "candidates": [
                    {"threshold": 128, "accepted": False, "sticker_score_total": 922.5},
                    {"threshold": 192, "accepted": True, "sticker_score_total": 837.6},
                    {"threshold": 224, "accepted": False, "sticker_score_total": 740.0},
                ],
            },
            "B": {
                "bestAnyThreshold": 224,
                "bestAcceptedThreshold": 224,
                "candidates": [
                    {"threshold": 128, "accepted": True, "sticker_score_total": 645.1},
                    {"threshold": 224, "accepted": True, "sticker_score_total": 556.0},
                ],
            },
        },
    }]

    summary = build_summary(rows)

    assert summary["sideCount"] == 2
    assert summary["bestAnyThresholdCounts"] == {"224": 2}
    assert summary["bestAcceptedThresholdCounts"] == {"192": 1, "224": 1}
    assert summary["acceptedSideCountByThreshold"] == {"128": 1, "192": 1, "224": 1}
    assert summary["medianScoreImprovementVs128"] == 135.8
