from __future__ import annotations

import json
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "api_contract"
VALID_CATEGORIES = {
    "success_clean",
    "success_repaired_high_confidence",
    "needs_manual_review",
    "reject_retake",
}


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


def test_recognize_contract_fixture_pins_cube_snap_consumed_fields():
    payload = _load_fixture("recognize_success_repaired_v1.json")

    assert payload["status"] == "success"
    assert len(payload["state"]) == 54
    assert payload["recognitionCategory"] in VALID_CATEGORIES
    assert isinstance(payload["recognitionCategoryReason"], str)

    signals = payload["recognitionSignals"]
    assert signals["schemaVersion"] == 1
    assert signals["repairPathUsed"] is True
    assert isinstance(signals["repairCandidateCount"], int)
    assert set(signals["selectedGridQuality"]) == {"imageA", "imageB"}
    assert set(signals["selectedGridQuality"]["imageA"]) == {"U", "F", "R"}
    assert set(signals["selectedGridQuality"]["imageB"]) == {"D", "L", "B"}
    assert signals["selectedSidesByImage"]["imageA"] == {"left": "F", "right": "R"}
    assert signals["selectedSidesByImage"]["imageB"] == {"left": "L", "right": "B"}
    assert signals["captureYaw"]["quarterTurns"] == 2
    assert signals["captureYaw"]["normalizationApplied"] is True
    assert len(signals["captureYaw"]["captureFrameState"]) == 54
    assert len(signals["topRepairCandidates"][0]["state"]) == 54
    assert len(signals["selectedRepairCandidate"]["state"]) == 54


def test_constrained_success_contract_fixture_pins_cube_snap_consumed_fields():
    payload = _load_fixture("recognize_constrained_success_clean_v1.json")

    assert payload["status"] == "success"
    assert len(payload["state"]) == 54
    assert payload["recognitionCategory"] == "success_clean"
    assert payload["failedChecks"] == []

    signal = payload["recognitionSignals"]["constrainedInference"]
    assert signal["selected"] is True
    assert signal["fallbackToLegacy"] is False
    assert signal["status"] == "accepted"
    assert signal["recommendedMethod"] == "canonical_count_repaired"
    assert signal["promotionGate"]["accepted"] is True
    assert signal["performance"]["schema"] == "constrained_recognize_performance_v1"
    assert signal["performance"]["contactSheetsIncluded"] is False
    assert signal["performance"]["stageTimingsMs"]["recognizeTotal"] > 0


def test_constrained_fast_reject_contract_fixture_pins_cube_snap_consumed_fields():
    payload = _load_fixture("recognize_constrained_fast_reject_v1.json")

    assert payload["status"] == "rejected"
    assert payload["state"] is None
    assert payload["recognitionCategory"] == "reject_retake"
    assert "non_cube_image_fast_reject" in payload["failedChecks"]

    signal = payload["recognitionSignals"]["constrainedInference"]
    assert signal["selected"] is False
    assert signal["fallbackToLegacy"] is False
    assert signal["status"] == "fast_reject"
    assert signal["fastReject"]["source"] == "hull_label_center_yaw_inference"
    assert signal["performance"]["schema"] == "constrained_recognize_performance_v1"
    assert signal["performance"]["contactSheetsIncluded"] is False
    assert "legacyFallback" not in signal["performance"]["stageTimingsMs"]


def test_llm_rectified_contract_fixture_pins_deterministic_repair_fields():
    payload = _load_fixture("llm_rectified_input_success_v1.json")

    assert payload["status"] == "success"
    assert payload["prompt"] == "rectified"
    assert payload["yawQuarterTurns"] == 2
    assert {panel["wcaFace"] for panel in payload["panels"]} == set("URFDLB")
    assert all(panel["image"] in {"imageA", "imageB"} for panel in payload["panels"])

    repair = payload["deterministicColorRepair"]
    assert repair["schema"] == "hull_label_color_repair_v1"
    assert repair["status"] == "assembled"
    assert repair["recommendedMethod"] == "guarded_broad_legal_repaired"
    assert len(repair["recommended"]["state"]) == 54
    assert repair["recommended"]["validState"] is True
    assert repair["recommended"]["countBalanced"] is True
