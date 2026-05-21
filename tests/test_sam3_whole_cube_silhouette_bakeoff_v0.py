from __future__ import annotations

import json
from pathlib import Path

from tools.sam3_whole_cube_silhouette_bakeoff_v0 import generate_silhouette_bakeoff


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "tests" / "fixtures" / "sam3_whole_cube_silhouette_bakeoff_v0_summary.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_silhouette_bakeoff_scores_paired_fit_outputs(tmp_path: Path):
    feedback = {
        "1_A": {"true_vertex": [10, 10]},
        "2_B": {"true_vertex": [100, 100]},
        "3_A": {},
    }
    feedback_path = tmp_path / "feedback.json"
    rembg_dir = tmp_path / "rembg"
    sam3_dir = tmp_path / "sam3"
    _write_json(feedback_path, feedback)

    _write_json(rembg_dir / "set_1_A_data.json", {"cube_center_screen": [20, 10], "debug": {"refinement": "applied"}})
    _write_json(sam3_dir / "set_1_A_data.json", {"cube_center_screen": [11, 10], "debug": {"refinement": "applied"}})
    _write_json(rembg_dir / "set_2_B_data.json", {"cube_center_screen": [100, 101], "debug": {"refinement": "skipped"}})
    _write_json(sam3_dir / "set_2_B_data.json", {"cube_center_screen": [130, 100], "debug": {"refinement": "skipped"}})

    document = generate_silhouette_bakeoff(
        feedback_path=feedback_path,
        rembg_dir=rembg_dir,
        sam3_dir=sam3_dir,
    )

    assert document["summary"]["rowCount"] == 2
    assert document["summary"]["skippedCount"] == 1
    assert document["summary"]["sam3BetterCount"] == 1
    assert document["summary"]["rembgBetterCount"] == 1
    assert [row["winner"] for row in document["rows"]] == ["sam3", "rembg"]


def test_committed_silhouette_bakeoff_preserves_negative_wiring_conclusion():
    document = json.loads(SUMMARY.read_text(encoding="utf-8"))
    summary = document["summary"]

    assert summary["rowCount"] == 23
    assert summary["thresholdCounts"]["30"]["rembgBelow"] == 0
    assert summary["thresholdCounts"]["30"]["sam3Below"] == 6
    assert summary["sam3BetterCount"] == 14
    assert summary["rembgBetterCount"] == 7
    assert summary["tieCount"] == 2
    assert summary["sam3LargeRegressionCount"] == 4
