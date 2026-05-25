from __future__ import annotations

from tools.diagnose_slot_yaw_assignment import (
    _assembled_state,
    _note_yaw_quarter_turns,
    _orientation_from_corner_map,
    build_summary,
    manifest_yaw_source,
    slot_face_assignments,
)


def test_note_yaw_quarter_turns_parses_documented_yaw():
    assert _note_yaw_quarter_turns("Human-confirmed capture yaw=3 (+270).") == 3
    assert _note_yaw_quarter_turns("Human-labeled capture yaw=0 (standard).") == 0
    assert _note_yaw_quarter_turns("No yaw metadata here.") is None


def test_manifest_yaw_source_prefers_expected_yaw_over_notes():
    row = {
        "expectedYaw": {
            "status": "nonstandard",
            "quarterTurns": 2,
            "normalizationApplied": True,
        },
        "notes": "Human-confirmed capture yaw=0.",
    }
    assert manifest_yaw_source(row) == {
        "source": "manifest_expectedYaw",
        "yawQuarterTurns": 2,
        "status": "nonstandard",
        "normalizationApplied": True,
    }


def test_manifest_yaw_source_falls_back_to_notes():
    row = {"notes": "Human-labeled capture yaw=1 (+90)."}
    assert manifest_yaw_source(row) == {
        "source": "manifest_notes",
        "yawQuarterTurns": 1,
        "status": "documented",
        "normalizationApplied": None,
    }


def test_slot_face_assignments_follow_shared_convention():
    assert slot_face_assignments(0) == {
        "A": {"upper": "U", "right": "R", "front": "F"},
        "B": {"upper": "D", "right": "B", "front": "L"},
    }
    assert slot_face_assignments(1) == {
        "A": {"upper": "U", "right": "B", "front": "R"},
        "B": {"upper": "D", "right": "L", "front": "F"},
    }


def test_assembled_state_requires_all_faces_in_urfdlb_order():
    chunks = {
        "U": list("U" * 9),
        "R": list("R" * 9),
        "F": list("F" * 9),
        "D": list("D" * 9),
        "L": list("L" * 9),
        "B": list("B" * 9),
    }
    assert _assembled_state(chunks) == "U" * 9 + "R" * 9 + "F" * 9 + "D" * 9 + "L" * 9 + "B" * 9
    assert _assembled_state({"U": list("U" * 9)}) is None


def test_orientation_from_corner_map_finds_row_major_identity():
    assert _orientation_from_corner_map({0: 0, 2: 2, 8: 8, 6: 6}) == (False, 0)


def test_build_summary_counts_sources_and_gt_aligned_scores():
    rows = [{
        "evaluations": {
            "assumed_zero": {
                "status": "assembled",
                "yawQuarterTurns": 0,
                "raw": {"exactMatch": False, "validState": False, "hamming": 9},
                "convention": {"exactMatch": True, "validState": True, "hamming": 0},
                "gtAligned": {"exactMatch": True, "validState": True, "hamming": 0},
            },
            "manifest_notes": {
                "status": "assembled",
                "yawQuarterTurns": 1,
                "raw": {"exactMatch": False, "validState": True, "hamming": 6},
                "convention": {"exactMatch": False, "validState": True, "hamming": 3},
                "gtAligned": {"exactMatch": False, "validState": True, "hamming": 3},
            },
        },
    }]

    summary = build_summary(rows)

    assert summary["pairCount"] == 1
    assert summary["byYawSource"]["assumed_zero"]["raw"]["meanStickersCorrect"] == 45
    assert summary["byYawSource"]["assumed_zero"]["convention"]["exact"] == 1
    assert summary["byYawSource"]["assumed_zero"]["gtAligned"]["exact"] == 1
    assert summary["byYawSource"]["manifest_notes"]["yawCounts"] == {"1": 1}
