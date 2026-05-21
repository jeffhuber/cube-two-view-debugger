from __future__ import annotations

import json
from pathlib import Path

from tools.trihedral_axis_fit_v0 import generate_trihedral_axis_fit_summary


def _row(
    *,
    key: str = "1_A",
    vertex=(100.0, 100.0),
    endpoints=((200.0, 100.0), (100.0, 200.0), (0.0, 100.0)),
    model_vertex=(102.0, 101.0),
    model_axes=((0.0, 100.0), (100.0, 0.0), (-100.0, 0.0)),
):
    return {
        "key": key,
        "setId": "1",
        "side": "A",
        "status": "labeled",
        "humanVertexPoint": list(vertex),
        "humanAxisEndpoints": [list(point) for point in endpoints],
        "currentModel": {
            "status": "ok",
            "vertexPoint": list(model_vertex),
            "axes": [
                {"name": f"axis_{idx}", "status": "ok", "vector": list(vector)}
                for idx, vector in enumerate(model_axes)
            ],
        },
    }


def test_trihedral_axis_fit_is_order_invariant(tmp_path: Path):
    fixture = {"rows": [_row()]}
    path = tmp_path / "feedback.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    document = generate_trihedral_axis_fit_summary(feedback_path=path)
    row = document["rows"][0]

    assert row["status"] == "strict_ready"
    assert row["vertexErrorPx"] < 3
    assert row["maxAxisAngleErrorDeg"] == 0.0
    assert row["axisAssignment"]["candidatePermutation"] == [1, 0, 2]


def test_trihedral_axis_fit_reports_pending_until_axes_are_labeled(tmp_path: Path):
    fixture = {
        "rows": [
            {
                **_row(),
                "status": "vertex_labeled_axes_unlabeled",
                "humanAxisEndpoints": [None, None, None],
            }
        ]
    }
    path = tmp_path / "feedback.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    document = generate_trihedral_axis_fit_summary(feedback_path=path)

    assert document["summary"]["trihedralLabeledRowCount"] == 0
    assert document["summary"]["axisLabelsPendingRowCount"] == 1
    assert document["rows"][0]["evaluationStatus"] == "axis_labels_pending"


def test_committed_trihedral_axis_summary_is_label_pending():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "trihedral_axis_fit_v0_summary.json"
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["summary"]["rowCount"] == 28
    assert document["summary"]["trihedralLabeledRowCount"] == 0
    assert document["summary"]["axisLabelsPendingRowCount"] == 23
