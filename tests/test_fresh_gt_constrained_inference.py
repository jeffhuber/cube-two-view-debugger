import json
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "fresh_gt_constrained_inference_summary.json"


def test_fresh_gt_constrained_inference_summary_matches_rows():
    payload = json.loads(FIXTURE.read_text())
    rows = payload["rows"]

    assert payload["schema"] == "fresh_gt_constrained_inference_v1"
    assert payload["summary"]["rowCount"] == 20
    assert len(rows) == 20
    assert {row["setId"] for row in rows} == {
        "8",
        "9",
        "10",
        "11",
        "13",
        "16",
        "18",
        "19",
        "33",
        "34",
        "35",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "59",
        "60",
    }

    exact = sum(1 for row in rows if row["recommendedHamming"] == 0)
    within3 = sum(1 for row in rows if row["recommendedHamming"] <= 3)
    legal = sum(1 for row in rows if row["recommendedValidState"])

    constrained = payload["summary"]["constrainedRecommended"]
    assert constrained["exact"] == exact == 19
    assert constrained["within3"] == within3 == 20
    assert constrained["legal"] == legal == 19
    assert constrained["hammingDistribution"] == {"0": 19, "2": 1}


def test_fresh_gt_report_tracks_gan_and_remaining_tail():
    payload = json.loads(FIXTURE.read_text())

    assert payload["summary"]["baseline"]["categoryDistribution"] == {
        "needs_manual_review": 6,
        "reject_retake": 12,
        "success_clean": 2,
    }
    assert {row["setId"] for row in payload["summary"]["ganRows"]} == {"50", "51"}
    assert all(row["recommendedHamming"] == 0 for row in payload["summary"]["ganRows"])

    misses = payload["summary"]["remainingRecommendedMisses"]
    assert [row["setId"] for row in misses] == ["11"]
    assert misses[0]["recommendedHamming"] == 2
    assert misses[0]["broadLegalHamming"] == 0
    assert misses[0]["guardedBroadLegalStatus"] == "rejected_guarded_broad_legal_repair"
