from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import oracle_color_evidence_report as report  # noqa: E402


SOLVED_STATE = (
    "U" * 9
    + "R" * 9
    + "F" * 9
    + "D" * 9
    + "L" * 9
    + "B" * 9
)


def _write_manifest(
    tmp_path: Path,
    state: str = SOLVED_STATE,
    sha: Optional[str] = None,
) -> Path:
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps([{"corrected": state}]), encoding="utf-8")
    expected_sha = sha
    if expected_sha is None:
        expected_sha = hashlib.sha256(gt_path.read_bytes()).hexdigest()
    manifest = {
        "pairs": [{
            "setId": "99",
            "groundTruthPath": str(gt_path),
            "groundTruth_sha256_expected": expected_sha,
        }]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _oracle_index(stickers):
    return {
        "schema": "oracle_rectified_faces_v1",
        "source": {"rows_glob": "99_A"},
        "rows": [{
            "key": "99_A",
            "set_id": "99",
            "side": "A",
            "yaw_quarter_turns": 0,
            "faces": [{
                "slot": "upper",
                "wca_face": "U",
                "stickers": stickers,
            }],
        }],
        "skipped": [],
    }


def _sticker(facelet_id: str, rgb):
    return {
        "facelet_id": facelet_id,
        "sticker_id": int(facelet_id[1:]),
        "row": 0,
        "col": 0,
        "rgb": list(rgb),
        "hsv": [0.0, 0.0, 0.0],
        "lab": [0.0, 0.0, 0.0],
        "sticker_png": f"by_observation/99_A/{facelet_id}.png",
        "patch_png": f"patch_png/99_A_{facelet_id}.png",
    }


def test_expected_color_uses_state_facelet_not_solved_face_name():
    state = "R" + SOLVED_STATE[1:]
    expected_face, expected_color = report.expected_color_for_facelet(state, "U1")
    assert expected_face == "R"
    assert expected_color == "red"


def test_analyze_reports_mode_accuracy_from_oracle_observations(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path)
    oracle_index = _oracle_index([
        _sticker("U1", (245, 245, 245)),
        _sticker("R1", (200, 45, 35)),
    ])

    payload = report.analyze_oracle_color_evidence(
        oracle_index,
        manifest_path=manifest_path,
    )

    assert payload["summary"]["observation_count"] == 2
    assert payload["summary"]["skipped_count"] == 0
    canonical = payload["modes"][report.CLASSIFIER_CANONICAL]
    assert canonical["correct"] == 2
    assert canonical["total"] == 2
    assert canonical["accuracy"] == 1.0
    assert canonical["per_expected_color_accuracy"]["white"] == 1.0
    assert canonical["per_expected_color_accuracy"]["red"] == 1.0


def test_mismatches_are_sorted_by_confidence(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path)
    oracle_index = _oracle_index([
        _sticker("R1", (60, 145, 82)),  # expected red, looks green.
    ])

    payload = report.analyze_oracle_color_evidence(
        oracle_index,
        manifest_path=manifest_path,
    )

    mismatches = payload["mismatches"][report.CLASSIFIER_CANONICAL]
    assert len(mismatches) == 1
    assert mismatches[0]["facelet_id"] == "R1"
    assert mismatches[0]["expected_color"] == "red"
    assert mismatches[0]["predicted_color"] == "green"


def test_weakest_observations_surface_low_confidence_samples(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path)
    oracle_index = _oracle_index([
        _sticker("U1", (245, 245, 245)),
        _sticker("U2", (120, 120, 120)),
    ])

    payload = report.analyze_oracle_color_evidence(
        oracle_index,
        manifest_path=manifest_path,
    )

    weakest = payload["weakest_observations"][report.CLASSIFIER_CANONICAL]
    assert len(weakest) == 2
    assert weakest[0]["confidence"] <= weakest[1]["confidence"]
    canonical = payload["modes"][report.CLASSIFIER_CANONICAL]
    assert "white" in canonical["confidence_stats_by_expected_color"]
    assert "median" in canonical["confidence_stats_by_expected_color"]["white"]


def test_manifest_ground_truth_hash_mismatch_skips_rows(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path, sha="0" * 64)
    oracle_index = _oracle_index([
        _sticker("U1", (245, 245, 245)),
    ])

    payload = report.analyze_oracle_color_evidence(
        oracle_index,
        manifest_path=manifest_path,
    )

    assert payload["summary"]["observation_count"] == 0
    assert payload["summary"]["skipped_count"] == 1
    assert "ground truth hash mismatch" in payload["skipped"][0]["reason"]


def test_render_markdown_report_contains_core_sections(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path)
    payload = report.analyze_oracle_color_evidence(
        _oracle_index([_sticker("U1", (245, 245, 245))]),
        manifest_path=manifest_path,
    )

    md = report.render_markdown_report(payload)

    assert "# Oracle color evidence report" in md
    assert "## Mode accuracy" in md
    assert "## Canonical confusion matrix" in md
    assert "## Lowest-confidence canonical observations" in md
    assert "Expected colors are read from canonical ground truth" in md


def test_render_markdown_report_sorts_mixed_set_ids(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path)
    payload = report.analyze_oracle_color_evidence(
        _oracle_index([_sticker("U1", (245, 245, 245))]),
        manifest_path=manifest_path,
    )
    for mode in report.MODE_ORDER:
        payload["modes"][mode]["per_set_accuracy"]["calibration"] = 1.0

    md = report.render_markdown_report(payload)

    assert "| 99 |" in md
    assert "| calibration |" in md
