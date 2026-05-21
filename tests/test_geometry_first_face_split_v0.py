from __future__ import annotations

import json
from pathlib import Path

from tools.geometry_first_face_split_v0 import generate_geometry_split_bakeoff


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "tests" / "fixtures" / "geometry_first_face_split_v0_summary.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fit(vertex, *, image_size=(220, 220)):
    x, y = vertex
    return {
        "cube_center_screen": vertex,
        "imageSize": list(image_size),
        "visible_corners": {
            "front": vertex,
            "h_x": [x + 45, y - 25],
            "h_y": [x - 45, y - 25],
            "h_z": [x, y + 55],
            "h_xy": [x, y - 50],
            "h_xz": [x + 45, y + 30],
            "h_yz": [x - 45, y + 30],
        },
        "debug": {"refinement": "applied", "fit_residual_rms_px": 10.0},
    }


def test_generate_geometry_split_bakeoff_marks_vertex_as_gate(tmp_path: Path):
    feedback_path = tmp_path / "feedback.json"
    rembg_dir = tmp_path / "rembg"
    sam3_dir = tmp_path / "sam3"
    _write_json(feedback_path, {"1_A": {"true_vertex": [80, 80]}})
    _write_json(rembg_dir / "set_1_A_data.json", _fit([81, 80]))
    _write_json(sam3_dir / "set_1_A_data.json", _fit([170, 170]))

    document = generate_geometry_split_bakeoff(
        feedback_path=feedback_path,
        rembg_dir=rembg_dir,
        sam3_dir=sam3_dir,
    )

    row = document["rows"][0]
    assert row["sources"]["rembg"]["status"] == "split_ready_strict"
    assert row["sources"]["sam3"]["status"] == "vertex_error_blocked"
    assert row["bestSourceByVertexError"] == "rembg"


def test_committed_geometry_first_split_fixture_keeps_vertex_blocker_visible():
    document = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary = document["summary"]

    assert summary["rowCount"] == 23
    assert summary["sources"]["rembg"]["strictReadyCount"] == 0
    assert summary["sources"]["sam3"]["strictReadyCount"] == 6
    assert summary["sources"]["oracle_best_source"]["strictReadyCount"] == 6
    assert summary["sources"]["oracle_best_source"]["plausibleCount"] < 23
