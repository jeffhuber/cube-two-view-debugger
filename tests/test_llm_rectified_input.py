from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from app import _infer_yaw_from_rectified_fits, _parse_llm_rectified_yaw
from rubik_recognizer.colors import CANONICAL_RGB, FACE_TO_COLOR
from tools.corner_conventions import wca_face_by_slot


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
