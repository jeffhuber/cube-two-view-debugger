from __future__ import annotations

import json
from pathlib import Path

from tools.vertex_label_server import save_feedback, update_feedback_row


def test_update_feedback_row_sets_label_without_touching_candidates():
    feedback = {
        "allowedStatuses": ["unlabeled", "labeled", "ambiguous", "not_visible"],
        "rows": [
            {
                "setId": "15",
                "side": "A",
                "status": "unlabeled",
                "humanVertexPoint": None,
                "labelQuality": None,
                "notes": "",
                "topCandidates": [{"rank": 1, "vertexPoint": [10.0, 20.0]}],
            }
        ],
    }

    update_feedback_row(
        feedback,
        set_id="15",
        side="A",
        status="labeled",
        human_vertex_point=(12.345, 67.891),
        label_quality="high",
        notes="clear vertex",
    )

    row = feedback["rows"][0]
    assert row["status"] == "labeled"
    assert row["humanVertexPoint"] == [12.35, 67.89]
    assert row["labelQuality"] == "high"
    assert row["notes"] == "clear vertex"
    assert row["topCandidates"] == [{"rank": 1, "vertexPoint": [10.0, 20.0]}]


def test_save_feedback_writes_json_and_report(tmp_path: Path):
    feedback = {
        "schemaVersion": 1,
        "sourceCandidateSummary": "candidate.json",
        "distanceThresholdPx": 10.0,
        "allowedStatuses": ["unlabeled", "labeled", "ambiguous", "not_visible"],
        "rows": [
            {
                "setId": "15",
                "side": "A",
                "evaluationTier": "easy_corpus",
                "candidateStatus": "ok",
                "overlayPath": "/tmp/overlay.png",
                "status": "labeled",
                "humanVertexPoint": [10.0, 10.0],
                "topCandidates": [
                    {"rank": 1, "source": "center_refine", "vertexPoint": [12.0, 10.0]},
                ],
            }
        ],
    }
    feedback_path = tmp_path / "feedback.json"
    report_path = tmp_path / "report.md"

    evaluation = save_feedback(feedback_path, report_path, feedback)

    assert evaluation["summary"]["top1HitCount"] == 1
    assert json.loads(feedback_path.read_text(encoding="utf-8"))["rows"][0]["status"] == "labeled"
    assert "Top-1 hit rate: 100.0%" in report_path.read_text(encoding="utf-8")
