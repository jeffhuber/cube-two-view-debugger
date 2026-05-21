from __future__ import annotations

import json
from pathlib import Path

from tools.vertex_axis_feedback import (
    build_feedback_scaffold,
    build_model_candidate_fixture,
    evaluate_feedback,
    update_feedback_row,
)
from tools.vertex_axis_label_server import save_feedback


def test_build_model_candidate_fixture_extracts_vertex_and_axes(tmp_path: Path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "set_1_A_data.json").write_text(
        json.dumps(
            {
                "cube_center_screen": [100, 100],
                "axis_x_2d": [100, 0],
                "axis_y_2d": [0, 100],
                "axis_z_2d": [-100, 0],
                "fit_quality": 0.9,
                "debug": {"fit_residual_rms_px": 12.3},
            }
        ),
        encoding="utf-8",
    )

    fixture = build_model_candidate_fixture(
        ground_truth={"1_A": {"current_vertex": [100, 100]}},
        model_data_dir=model_dir,
    )

    row = fixture["rows"][0]
    assert row["status"] == "ok"
    assert row["vertexPoint"] == [100.0, 100.0]
    assert [axis["endpoint"] for axis in row["axes"]] == [
        [200.0, 100.0],
        [100.0, 200.0],
        [0.0, 100.0],
    ]


def test_feedback_scaffold_preserves_vertex_labels_and_pending_axes():
    feedback = build_feedback_scaffold(
        ground_truth={
            "1_A": {
                "center_correct": False,
                "current_vertex": [100, 100],
                "true_vertex": [101, 102],
                "error_px": 2.2,
            }
        },
        model_candidates={"rows": [{"key": "1_A", "status": "missing_model_data"}]},
    )

    row = feedback["rows"][0]
    assert row["status"] == "vertex_labeled_axes_unlabeled"
    assert row["humanVertexPoint"] == [101.0, 102.0]
    assert row["humanAxisEndpoints"] == [None, None, None]
    assert evaluate_feedback(feedback)["summary"]["trihedralLabeledRowCount"] == 0


def test_update_feedback_row_and_save_report(tmp_path: Path):
    feedback = {
        "allowedStatuses": ["unlabeled", "vertex_labeled_axes_unlabeled", "labeled"],
        "rows": [
            {
                "key": "1_A",
                "status": "unlabeled",
                "humanVertexPoint": None,
                "humanAxisEndpoints": [None, None, None],
                "currentModel": None,
            }
        ],
    }

    update_feedback_row(
        feedback,
        key="1_A",
        status="labeled",
        human_vertex_point=(10.123, 20.456),
        human_axis_endpoints=[(110, 20), (10, 120), (-90, 20)],
        axis_label_quality="high",
        notes="clear",
    )

    row = feedback["rows"][0]
    assert row["humanVertexPoint"] == [10.12, 20.46]
    assert row["humanAxisEndpoints"] == [[110.0, 20.0], [10.0, 120.0], [-90.0, 20.0]]

    feedback_path = tmp_path / "feedback.json"
    report_path = tmp_path / "report.md"
    evaluation = save_feedback(feedback_path, report_path, feedback)
    assert evaluation["summary"]["trihedralLabeledRowCount"] == 1
    assert "Full trihedral labels: 1" in report_path.read_text(encoding="utf-8")
