from __future__ import annotations

import json
from pathlib import Path

from tools.diagnose_hull_label_color_repair import (
    _face_index,
    _metadata_yaw_source,
    build_summary,
    greedy_count_repair,
)
from tools.extract_color_samples import PairTask


def _balanced_faces() -> list[str]:
    return [face for face in "URFDLB" for _ in range(9)]


def test_greedy_count_repair_preserves_centers_and_repairs_counts():
    faces = _balanced_faces()
    index_to_repair = _face_index("D", 0)
    faces[index_to_repair] = "U"

    costs = [{face: 50.0 for face in "URFDLB"} for _ in range(54)]
    for index, face in enumerate(faces):
        costs[index][face] = 0.0
    costs[index_to_repair]["U"] = 5.0
    costs[index_to_repair]["D"] = 1.0
    costs[_face_index("U", 4)]["D"] = -100.0

    repaired, moves = greedy_count_repair(
        faces,
        costs,
        fixed_indices=[_face_index(face, 4) for face in "URFDLB"],
    )

    assert repaired[index_to_repair] == "D"
    assert repaired[_face_index("U", 4)] == "U"
    assert {face: repaired.count(face) for face in "URFDLB"} == {face: 9 for face in "URFDLB"}
    assert moves == [{"index": index_to_repair, "from": "U", "to": "D", "delta": -4.0}]


def test_metadata_yaw_source_prefers_ground_truth_capture_yaw(tmp_path: Path):
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(
        json.dumps([{
            "captureYawQuarterTurns": 2,
            "canonicalizationSource": "center-inference",
            "corrected": "U" * 54,
        }]),
        encoding="utf-8",
    )
    task = PairTask(
        set_id="69",
        image_a=tmp_path / "a.jpg",
        image_b=tmp_path / "b.jpg",
        ground_truth=gt_path,
        source="test",
    )

    source = _metadata_yaw_source(task, {"notes": "Human-labeled capture yaw=0."})

    assert source == {
        "source": "ground_truth_captureYaw",
        "yawQuarterTurns": 2,
        "status": "center-inference",
    }


def test_build_summary_counts_repair_methods():
    rows = [{
        "evaluations": {
            "white_up_default": {
                "status": "assembled",
                "methods": {
                    "canonical": {"hamming": 6, "exactMatch": False, "validState": False},
                    "canonical_center_forced": {"hamming": 6, "exactMatch": False, "validState": False},
                    "canonical_count_repaired": {"hamming": 0, "exactMatch": True, "validState": True},
                    "adaptive": {"hamming": 3, "exactMatch": False, "validState": False},
                    "adaptive_center_forced": {"hamming": 3, "exactMatch": False, "validState": False},
                    "adaptive_count_repaired": {"hamming": 0, "exactMatch": True, "validState": True},
                },
            }
        }
    }]

    summary = build_summary(rows)

    repaired = summary["yawSources"]["white_up_default"]["adaptive_count_repaired"]
    assert repaired["assembled"] == 1
    assert repaired["exact"] == 1
    assert repaired["legal"] == 1
    assert repaired["meanStickersCorrect"] == 54
