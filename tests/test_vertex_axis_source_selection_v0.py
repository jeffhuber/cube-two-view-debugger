from __future__ import annotations

import json
from pathlib import Path

from tools.vertex_axis_source_selection_v0 import generate_source_selection_bakeoff


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "tests" / "fixtures" / "vertex_axis_source_selection_v0_summary.json"


def _source(error: float, residual: float, quality: float) -> dict:
    return {
        "vertexErrorPx": error,
        "fitResidualRmsPx": residual,
        "fitQuality": quality,
        "status": "split_ready_strict" if error <= 30 else "vertex_error_blocked",
        "vertex": [0, 0],
        "nondegenerate": True,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_source_selection_can_abstain_on_low_confidence(tmp_path: Path):
    fixture = {
        "rows": [
            {
                "key": "1_A",
                "bestSourceByVertexError": "sam3",
                "trueVertex": [0, 0],
                "sources": {
                    "rembg": _source(80, 45, 0.80),
                    "sam3": _source(20, 15, 0.91),
                },
            },
            {
                "key": "2_A",
                "bestSourceByVertexError": "rembg",
                "trueVertex": [0, 0],
                "sources": {
                    "rembg": _source(40, 30, 0.84),
                    "sam3": _source(120, 28, 0.85),
                },
            },
        ]
    }
    fixture_path = tmp_path / "geometry.json"
    _write_json(fixture_path, fixture)

    document = generate_source_selection_bakeoff(geometry_fixture_path=fixture_path)
    strict = document["summary"]["policySummaries"]["strict_residual_margin_confidence_v0"]

    assert strict["selectedCount"] == 1
    assert strict["abstainCount"] == 1
    assert strict["falseConfidentCount"] == 0


def test_committed_source_selection_fixture_preserves_no_wiring_conclusion():
    document = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summaries = document["summary"]["policySummaries"]

    assert document["summary"]["rowCount"] == 23
    assert summaries["global_model_score_v0"]["selectedCount"] == 23
    assert summaries["global_model_score_v0"]["bestSourceCorrectCount"] == 17
    assert summaries["global_model_score_v0"]["falseConfidentCount"] == 15
    assert summaries["strict_residual_margin_confidence_v0"]["selectedCount"] == 1
    assert summaries["strict_residual_margin_confidence_v0"]["falseConfidentCount"] == 0
    assert summaries["oracle_best_source"]["plausibleCount"] == 11
