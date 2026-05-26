from __future__ import annotations

import json
from pathlib import Path

from tools import diagnose_hull_label_yaw_source as diag
from tools.corner_conventions import wca_face_by_slot


def _observed_for_yaw(yaw: int):
    assignments = {
        "A": wca_face_by_slot("A", yaw),
        "B": wca_face_by_slot("B", yaw),
    }
    return [
        (side, slot, assignments[side][slot])
        for side in ("A", "B")
        for slot in ("upper", "right", "front")
    ]


def test_score_yaw_candidates_prefers_unique_center_match():
    result = diag.score_yaw_candidates(_observed_for_yaw(2))

    assert result["accepted"] is True
    assert result["yawQuarterTurns"] == 2
    assert result["bestScore"] == 6
    assert result["secondScore"] == 2
    assert result["margin"] == 4


def test_score_yaw_candidates_accepts_one_bad_center_with_margin():
    observed = _observed_for_yaw(3)
    observed[-1] = (observed[-1][0], observed[-1][1], "U")

    result = diag.score_yaw_candidates(observed)

    assert result["accepted"] is True
    assert result["yawQuarterTurns"] == 3
    assert result["bestScore"] == 5
    assert result["secondScore"] == 2


def test_score_yaw_candidates_rejects_ambiguous_centers():
    observed = [
        ("A", "upper", "U"),
        ("A", "right", "R"),
        ("A", "front", "B"),
        ("B", "upper", "D"),
        ("B", "right", "B"),
        ("B", "front", "R"),
    ]

    result = diag.score_yaw_candidates(observed)

    assert result["accepted"] is False
    assert result["yawQuarterTurns"] is None
    assert result["bestScore"] == 4
    assert result["secondScore"] == 4


def test_committed_slot_trace_supports_center_yaw_decision():
    trace = json.loads(Path("tests/fixtures/hull_label_slot_yaw_assignment.json").read_text())
    manifest = diag._manifest_by_set(Path("tests/fixtures/corpus_manifest.json"))

    rows = diag.evaluate_rows(trace["rows"], manifest)
    summary = diag.build_summary(rows)

    assert summary["rows"] == len(trace["rows"])
    assert summary["accepted"] + summary["rejected"] == summary["rows"]
    assert summary["manifestAgreement"] == summary["manifestKnown"]
    assert summary["legacyDetectedAgreement"] == summary["legacyDetectedAvailable"]
    assert summary["accepted"] / summary["rows"] >= 0.85

    accepted = [row for row in rows if row["accepted"]]
    rejected = [row for row in rows if not row["accepted"]]
    assert accepted
    assert all(row["inferredYawQuarterTurns"] in {0, 1, 2, 3} for row in accepted)
    assert all(row["bestScore"] >= 5 for row in accepted)
    assert all(row["margin"] >= 2 for row in accepted)
    assert all(row["rejectReason"] for row in rejected)
