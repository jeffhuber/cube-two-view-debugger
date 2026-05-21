from __future__ import annotations

import json
from pathlib import Path

from tools.active_vertex_axis_label_queue_v0 import (
    DEFAULT_ACTIVE_FEEDBACK,
    DEFAULT_MODEL_SUMMARY,
    build_active_learning_feedback,
)


def test_active_queue_excludes_existing_full_labels_and_prioritizes_easy_rows(tmp_path: Path):
    model_summary = {
        "rows": [
            _model_row("1", "A", "easy_corpus", "ok", 0.85),
            _model_row("1", "B", "easy_corpus", "ok", 0.82),
            _model_row("2", "A", "hard_case_stress", "low_cell_inside", 0.55),
        ]
    }
    existing_feedback = {
        "rows": [
            {
                "key": "1_A",
                "status": "labeled",
                "humanVertexPoint": [10.0, 10.0],
                "humanAxisEndpoints": [[0.0, 10.0], [10.0, 20.0], [20.0, 20.0]],
            }
        ]
    }
    model_path = tmp_path / "models.json"
    feedback_path = tmp_path / "feedback.json"
    model_path.write_text(json.dumps(model_summary), encoding="utf-8")
    feedback_path.write_text(json.dumps(existing_feedback), encoding="utf-8")

    queue = build_active_learning_feedback(
        model_summary_path=model_path,
        existing_feedback_path=feedback_path,
    )

    assert [row["key"] for row in queue["rows"]] == ["1_B", "2_A"]
    assert queue["rows"][0]["activeLearning"]["priority"] == "tier1_easy_unlabeled"
    assert queue["rows"][1]["activeLearning"]["priority"] == "tier4_retake_or_model_boundary"
    assert queue["rows"][0]["currentModel"]["axes"][0]["status"] == "ok"


def test_committed_active_queue_is_strict_json_and_has_expected_counts():
    for path in (DEFAULT_MODEL_SUMMARY, DEFAULT_ACTIVE_FEEDBACK):
        text = path.read_text(encoding="utf-8")
        assert "Infinity" not in text
        assert "NaN" not in text
        json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )

    queue = json.loads(DEFAULT_ACTIVE_FEEDBACK.read_text(encoding="utf-8"))
    assert len(queue["rows"]) == 30
    assert all(row["status"] == "unlabeled" for row in queue["rows"])


def _model_row(set_id: str, side: str, tier: str, status: str, iou: float) -> dict:
    return {
        "setId": set_id,
        "side": side,
        "evaluationTier": tier,
        "status": status,
        "diagnosticDisposition": "model_ok" if status == "ok" else "geometry_retake_or_segmentation_candidate",
        "imagePath": f"/tmp/set_{set_id}_{side}.jpg",
        "fitDiagnostics": {"fitVersion": "v0.1-center-refine"},
        "model": {
            "cubeCenter": [100.0, 100.0],
            "axes": [[-30.0, 0.0], [0.0, 30.0], [30.0, 30.0]],
            "edgeLength": 30.0,
            "score": 2.0,
            "signChoice": [-1, 1, 1],
            "scoreComponents": {
                "silhouetteIoU": iou,
                "insideRatio": 0.95,
                "cellInsideRatio": 0.85 if status != "ok" else 1.0,
                "maskCoverage": 0.90,
                "detectorSignalQuality": 0.2,
            },
        },
    }
