from __future__ import annotations

import json
from pathlib import Path

from tools import diagnose_hull_label_yaw_source as diag


def _observed_for_yaw(yaw: int):
    assignments = {
        "A": diag.wca_face_by_slot("A", yaw),
        "B": diag.wca_face_by_slot("B", yaw),
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

    assert summary["rows"] == 41
    assert summary["accepted"] == 40
    assert summary["manifestKnown"] == 14
    assert summary["manifestAgreement"] == 14
    assert summary["legacyDetectedAvailable"] == 27
    assert summary["legacyDetectedAgreement"] == 27
    assert summary["legacyDetectedMissingButInferred"] == 13
    assert summary["rejectedReasons"] == {"fit_failed": 1}
