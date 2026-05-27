from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from app import (
    _infer_yaw_from_rectified_fits,
    _parse_llm_rectified_yaw,
)
from rubik_recognizer.colors import CANONICAL_RGB, FACE_TO_COLOR
from tools.corner_conventions import wca_face_by_slot
from tools.hull_label_pair_selector import choose_guarded_pair


def _solid_face(wca_face: str) -> Image.Image:
    return Image.new("RGB", (300, 300), CANONICAL_RGB[FACE_TO_COLOR[wca_face]])


def _fits_for_yaw(yaw: int):
    return {
        side: SimpleNamespace(
            rectified_faces={
                slot: _solid_face(wca_face)
                for slot, wca_face in wca_face_by_slot(side, yaw).items()
            }
        )
        for side in ("A", "B")
    }


def test_parse_llm_rectified_yaw_accepts_auto_and_explicit_values():
    assert _parse_llm_rectified_yaw(None) is None
    assert _parse_llm_rectified_yaw("") is None
    assert _parse_llm_rectified_yaw("auto") is None
    assert _parse_llm_rectified_yaw("center-inference") is None
    assert _parse_llm_rectified_yaw("2") == 2
    assert _parse_llm_rectified_yaw("6") == 2

    with pytest.raises(ValueError, match="integer 0..3 or auto"):
        _parse_llm_rectified_yaw("sideways")


def test_infer_yaw_from_rectified_slot_center_faces():
    result = _infer_yaw_from_rectified_fits(_fits_for_yaw(2))

    assert result["accepted"] is True
    assert result["yawQuarterTurns"] == 2
    assert result["bestScore"] == 6
    assert result["margin"] == 4
    assert {item["centerFace"] for item in result["observedCenters"]} == set("URFDLB")


def test_infer_yaw_falls_back_to_side_faces_when_ud_centers_are_noisy():
    fits = _fits_for_yaw(0)
    fits["A"].rectified_faces["upper"] = _solid_face("B")
    fits["B"].rectified_faces["upper"] = _solid_face("F")

    result = _infer_yaw_from_rectified_fits(fits)

    assert result["accepted"] is True
    assert result["status"] == "accepted_side_faces_only"
    assert result["yawQuarterTurns"] == 0
    assert result["bestScore"] == 4
    assert result["secondScore"] == 0
    assert result["allCenterBestScore"] == 4


def _threshold_combo(a_threshold: int, b_threshold: int, *, valid: bool, moves: int, score: float = 100.0):
    return {
        "thresholds": {"A": a_threshold, "B": b_threshold},
        "evaluation": {
            "status": "assembled",
            "yawQuarterTurns": 0,
            "repair": {
                "methods": {
                    "canonical_count_repaired": {
                        "validState": valid,
                        "countBalanced": True,
                        "repairMoveCount": moves,
                    }
                },
                "recommended": {
                    "validState": valid,
                    "repairMoveCount": moves,
                },
            },
        },
        "stickerScoreTotal": score,
    }


def test_pair_threshold_selection_keeps_current_when_current_repair_valid():
    current = _threshold_combo(224, 192, valid=True, moves=10, score=200.0)
    tempting = _threshold_combo(224, 128, valid=True, moves=9, score=150.0)

    selected = choose_guarded_pair(
        current_combo=current,
        candidates=[current, tempting],
        fallback_to_current_without_alternative=True,
    )

    assert selected["thresholds"] == {"A": 224, "B": 192}
    assert selected["selectionReason"] == "kept_current_valid_repair"


def test_pair_threshold_selection_switches_when_current_repair_invalid():
    current = _threshold_combo(160, 160, valid=False, moves=4, score=100.0)
    better = _threshold_combo(64, 192, valid=True, moves=8, score=120.0)

    selected = choose_guarded_pair(
        current_combo=current,
        candidates=[current, better],
        fallback_to_current_without_alternative=True,
    )

    assert selected["thresholds"] == {"A": 64, "B": 192}
    assert selected["selectionReason"] == "current_invalid_selected_best_pair"
