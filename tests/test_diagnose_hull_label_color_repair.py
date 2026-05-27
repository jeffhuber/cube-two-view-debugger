from __future__ import annotations

import json
from pathlib import Path

from tools.diagnose_hull_label_color_repair import (
    _metadata_yaw_source,
    build_summary,
    render_report,
)
from tools.hull_label_color_repair import (
    FACE_ORDER,
    CANONICAL_RGB,
    FACE_TO_COLOR,
    StickerObservation,
    assemble_color_repair_payload,
    choose_recommended_method,
    _face_index,
    _guarded_broad_payload,
    _two_view_consistency_payload,
    greedy_count_repair,
    state_delta_payload,
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


def test_render_report_highlights_hull_label_center_color_scoreboard():
    row = {
        "setId": "99",
        "evaluations": {
            "hull_label_center_colors": {
                "status": "assembled",
                "methods": {
                    "canonical": {"hamming": 2, "exactMatch": False, "validState": False},
                    "canonical_center_forced": {"hamming": 2, "exactMatch": False, "validState": False},
                    "canonical_count_repaired": {"hamming": 0, "exactMatch": True, "validState": True},
                    "adaptive": {"hamming": 1, "exactMatch": False, "validState": False},
                    "adaptive_count_repaired": {"hamming": 0, "exactMatch": True, "validState": True},
                },
            }
        },
    }
    payload = {
        "source": {"git_sha": "abc123", "generated_at_utc": "2026-05-26T00:00:00+00:00"},
        "summary": build_summary([row]),
        "rows": [row],
    }

    report = render_report(payload)

    assert "The production-like yaw source is `hull_label_center_colors`" in report
    assert "9-per-color count repair" in report
    assert "`canonical_count_repaired` is the stable deterministic baseline" in report
    assert "The payload's recommended-method selector is now" in report
    assert "Guarded two-view and cubie-legality repair are now part of the color-repair payload" in report


def test_assemble_color_repair_payload_exposes_repaired_draft():
    observations = []
    for face in FACE_ORDER:
        for face_index in range(9):
            rgb = CANONICAL_RGB[FACE_TO_COLOR[face]]
            observations.append(
                StickerObservation(
                    index=_face_index(face, face_index),
                    side="A" if face in "URF" else "B",
                    slot="upper",
                    wca_face=face,
                    face_index=face_index,
                    rgb=rgb,
                    raw_color=FACE_TO_COLOR[face],
                )
            )

    payload = assemble_color_repair_payload(
        observations=observations,
        panel_meta={"panels": []},
        yaw_quarter_turns=0,
    )

    assert payload["status"] == "assembled"
    assert payload["recommendedMethod"] == "canonical_count_repaired"
    assert payload["recommended"]["state"] == "".join(face * 9 for face in FACE_ORDER)
    assert payload["recommended"]["validState"] is True
    assert payload["recommended"]["repairMoveCount"] == 0
    assert payload["recommended"]["confidence"] == "high"
    assert payload["methods"]["conservative_legal_repaired"]["status"] == "already_valid_count_repair"
    assert payload["methods"]["two_view_consistency_repaired"]["status"] == "rejected_two_view_consistency_repair"
    assert "no_split_cubie_inconsistency" in payload["methods"]["two_view_consistency_repaired"]["gate"]["reasons"]
    assert payload["methods"]["guarded_broad_legal_repaired"]["gate"]["accepted"] is True
    assert payload["methods"]["guarded_broad_legal_repaired"]["gate"]["stateDeltaFromCanonical"]["count"] == 0
    assert payload["methods"]["broad_legal_repaired"]["diagnosticOnly"] is True


def test_state_delta_gate_uses_count_repaired_delta_not_repair_changes():
    baseline = "U" * 54
    set11_like = list(baseline)
    set11_like[20] = "F"
    set11_like[53] = "B"
    set14_like = list(baseline)
    for index in range(6):
        set14_like[index] = "R"

    admit = _guarded_broad_payload(
        {
            "status": "legal_repair_found",
            "state": "".join(set11_like),
            "validState": True,
            "repairCost": 10.15,
            "repairChanges": 6,
            "stateDeltaFromCanonical": state_delta_payload(baseline, "".join(set11_like)),
        },
        gt_state=None,
    )
    reject = _guarded_broad_payload(
        {
            "status": "legal_repair_found",
            "state": "".join(set14_like),
            "validState": True,
            "repairCost": 10.15,
            "repairChanges": 5,
            "stateDeltaFromCanonical": state_delta_payload(baseline, "".join(set14_like)),
        },
        gt_state=None,
    )

    assert admit["status"] == "accepted_guarded_broad_legal_repair"
    assert admit["gate"]["stateDeltaFromCanonical"]["count"] == 2
    assert admit["repairChanges"] == 6
    assert reject["status"] == "rejected_guarded_broad_legal_repair"
    assert reject["gate"]["stateDeltaFromCanonical"]["count"] == 6
    assert reject["rejectedRepairChanges"] == 5


def test_two_view_consistency_gate_requires_split_cubie_evidence():
    solved = "".join(face * 9 for face in FACE_ORDER)
    split_corrupt = list(solved)
    split_corrupt[_face_index("L", 2)] = "D"  # UFL sees U/F/D, not a real corner.
    split_corrupt_state = "".join(split_corrupt)
    in_image_corrupt = list(solved)
    in_image_corrupt[_face_index("F", 2)] = "D"  # URF is fully visible in image A.
    in_image_corrupt_state = "".join(in_image_corrupt)

    broad_payload = {
        "status": "legal_repair_found",
        "state": solved,
        "validState": True,
        "repairCost": 3.0,
        "repairChanges": 1,
        "stateDeltaFromCanonical": state_delta_payload(split_corrupt_state, solved),
        "sourceMode": "grid_sample",
    }
    accepted = _two_view_consistency_payload(
        broad_payload,
        baseline_state=split_corrupt_state,
        gt_state=None,
    )
    rejected = _two_view_consistency_payload(
        {
            **broad_payload,
            "stateDeltaFromCanonical": state_delta_payload(in_image_corrupt_state, solved),
        },
        baseline_state=in_image_corrupt_state,
        gt_state=None,
    )

    assert accepted["status"] == "accepted_two_view_consistency_repair"
    assert accepted["gate"]["baselineCubieConsistency"]["inconsistentSplitCount"] == 1
    assert accepted["gate"]["candidateCubieConsistency"]["inconsistentCount"] == 0
    assert rejected["status"] == "rejected_two_view_consistency_repair"
    assert "no_split_cubie_inconsistency" in rejected["gate"]["reasons"]


def test_choose_recommended_method_uses_guarded_legal_before_balanced_fallback():
    methods = {
        "canonical_count_repaired": {"validState": False, "countBalanced": True},
        "conservative_legal_repaired": {"validState": False, "countBalanced": False},
        "two_view_consistency_repaired": {"validState": False, "countBalanced": False},
        "guarded_broad_legal_repaired": {"validState": True, "countBalanced": True},
        "broad_legal_repaired": {"validState": True, "countBalanced": True},
    }

    assert choose_recommended_method(methods) == "guarded_broad_legal_repaired"


def test_choose_recommended_method_prefers_two_view_over_guarded_broad():
    methods = {
        "canonical_count_repaired": {"validState": False, "countBalanced": True},
        "conservative_legal_repaired": {"validState": False, "countBalanced": False},
        "two_view_consistency_repaired": {"validState": True, "countBalanced": True},
        "guarded_broad_legal_repaired": {"validState": True, "countBalanced": True},
    }

    assert choose_recommended_method(methods) == "two_view_consistency_repaired"
