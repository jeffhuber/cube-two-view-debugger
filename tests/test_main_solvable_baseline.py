import json
from pathlib import Path

from tools.main_solvable_baseline import (
    build_baseline,
    render_report,
    summarize_row,
)


def test_summarize_row_marks_confident_wrong_only_for_confident_success():
    confident_wrong = summarize_row(
        "corpus",
        {
            "setId": "1",
            "status": "success",
            "category": "success_clean",
            "categoryReason": "direct_unique_legal_high_confidence",
            "score": 53,
            "hamming": 1,
            "recognizedState": "U" * 54,
            "contractPassed": True,
        },
    )
    manual_wrong = summarize_row(
        "corpus",
        {
            "setId": "2",
            "status": "success",
            "category": "needs_manual_review",
            "score": 53,
            "hamming": 1,
            "recognizedState": "U" * 54,
            "contractPassed": True,
        },
    )

    assert confident_wrong["legalState"] is True
    assert confident_wrong["confidentSolve"] is True
    assert confident_wrong["confidentWrong"] is True
    assert manual_wrong["legalState"] is True
    assert manual_wrong["confidentSolve"] is False
    assert manual_wrong["confidentWrong"] is False


def test_build_baseline_uses_scored_denominator_and_tracks_skips(tmp_path):
    payload = {
        "schemaVersion": 1,
        "manifest": str(tmp_path / "manifest.json"),
        "timingSummary": {"rowCount": 3},
        "environmentPolicyWarnings": [],
        "results": [
            {
                "setId": "1",
                "status": "success",
                "category": "success_clean",
                "score": 54,
                "hamming": 0,
                "recognizedState": "U" * 54,
            },
            {
                "setId": "2",
                "status": "rejected",
                "category": "reject_retake",
                "score": 0,
                "hamming": 54,
                "recognizedState": "",
            },
            {
                "setId": "3",
                "status": "skipped",
                "category": "missing_files",
                "missingFiles": ["/missing/file"],
            },
        ],
    }

    baseline = build_baseline(
        [("corpus", tmp_path / "probe.json", payload)],
        git_head="abc123",
        cwd=tmp_path,
    )
    summary = baseline["overall"]

    assert summary["rowCount"] == 3
    assert summary["scoredRowCount"] == 2
    assert summary["skippedRowCount"] == 1
    assert summary["scoreSum"] == 54
    assert summary["perStickerAccuracy"] == 0.5
    assert summary["perStickerAccuracyAllRowsSkippedAsZero"] == 0.333333
    assert summary["exactMatchRate"] == 0.5
    assert summary["legalStateRate"] == 0.5
    assert summary["confidentSolveRate"] == 0.5
    assert summary["confidentWrongRate"] == 0.0


def test_committed_fixture_is_internally_consistent():
    path = Path("tests/fixtures/main_solvable_baseline.json")
    if not path.exists():
        return

    baseline = json.loads(path.read_text(encoding="utf-8"))
    scored = [row for row in baseline["rows"] if row["scored"]]
    score_sum = sum(row["score"] for row in scored)
    confident_wrong = [row for row in scored if row["confidentWrong"]]

    assert baseline["overall"]["scoredRowCount"] == len(scored)
    assert baseline["overall"]["scoreSum"] == score_sum
    assert baseline["overall"]["confidentWrongCount"] == len(confident_wrong)
    assert baseline["overall"]["confidentWrongCount"] == 0
    assert render_report(baseline).startswith("# Current-main solvable baseline\n")
