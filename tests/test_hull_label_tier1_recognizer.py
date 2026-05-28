from __future__ import annotations

import io
import json
from typing import Dict, List, Optional

from PIL import Image

from rubik_recognizer.dataset import ImagePair, ImageUpload
from rubik_recognizer.image_pipeline import FaceGrid, ImageAnalysis
from rubik_recognizer.recognizer import (
    HULL_LABEL_DIRECT_OPTION_SCORE,
    RecognitionResult,
    WhiteUpRecognizer,
    _base_recognition_signals,
    _hull_label_direct_options,
)
from tools.corner_conventions import wca_face_by_slot


SOLVED_STATE = (
    "U" * 9
    + "R" * 9
    + "F" * 9
    + "D" * 9
    + "L" * 9
    + "B" * 9
)


def _result(
    *,
    state: Optional[str] = SOLVED_STATE,
    status: str = "success",
    selected_a: bool = False,
    selected_b: bool = False,
) -> RecognitionResult:
    recognition_signals = {
        "schemaVersion": 1,
        "hullLabelTier1": {
            "mode": "prefer",
            "images": {
                "imageA": {"status": "accepted", "selected": selected_a},
                "imageB": {"status": "accepted", "selected": selected_b},
            },
        },
        "selectedGridQuality": {
            "imageA": {"U": {"matchedCount": 9}},
            "imageB": {"D": {"matchedCount": 9}},
        },
        "topVisibleTripleQuality": {
            "imageA": {"available": True},
            "imageB": {"available": True},
        },
    }
    return RecognitionResult(
        status=status,
        state=state if status == "success" else None,
        confidence=0.9 if status == "success" else 0.0,
        reason="test result",
        failed_checks=[] if status == "success" else ["test_reject"],
        recognition_signals=recognition_signals,
    )


def test_prefer_mode_returns_hull_label_candidate_when_both_sides_selected(monkeypatch):
    calls: List[str] = []

    def fake_recognize(self, image_a, image_b, mode):
        calls.append(mode)
        if mode == "prefer":
            return _result(state="R" * 54, selected_a=True, selected_b=True)
        return _result(state=SOLVED_STATE)

    monkeypatch.setattr(WhiteUpRecognizer, "_recognize_with_hull_label_mode", fake_recognize)

    result = WhiteUpRecognizer().recognize(b"a", b"b", hull_label_tier1_mode="prefer")

    assert calls == ["off", "prefer"]
    assert result.state == "R" * 54
    assert result.recognition_signals["hullLabelTier1Prefer"]["selected"] is True
    assert result.recognition_signals["hullLabelTier1Prefer"]["fallbackToLegacy"] is False


def test_prefer_mode_falls_back_to_legacy_when_candidate_not_selected(monkeypatch):
    calls: List[str] = []

    def fake_recognize(self, image_a, image_b, mode):
        calls.append(mode)
        if mode == "prefer":
            return _result(state="R" * 54, selected_a=True, selected_b=False)
        return _result(state=SOLVED_STATE)

    monkeypatch.setattr(WhiteUpRecognizer, "_recognize_with_hull_label_mode", fake_recognize)

    result = WhiteUpRecognizer().recognize(b"a", b"b", hull_label_tier1_mode="prefer")

    assert calls == ["off", "prefer"]
    assert result.state == SOLVED_STATE
    decision = result.recognition_signals["hullLabelTier1Prefer"]
    assert decision["selected"] is False
    assert decision["fallbackToLegacy"] is True
    assert decision["candidateHullLabelTier1"]["images"]["imageA"]["selected"] is True
    assert decision["candidateHullLabelTier1"]["images"]["imageB"]["selected"] is False
    assert decision["candidateDiagnostics"]["selectedGridQuality"]["imageA"]["U"]["matchedCount"] == 9
    assert decision["candidateDiagnostics"]["topVisibleTripleQuality"]["imageB"]["available"] is True


def test_recognize_and_persist_forwards_hull_label_tier1_mode(tmp_path, monkeypatch):
    import app as app_module

    class FakeRecognizer:
        def __init__(self):
            self.mode = None

        def recognize(self, image_a, image_b, *, hull_label_tier1_mode=None):
            self.mode = hull_label_tier1_mode
            return RecognitionResult(status="rejected", reason="fake")

    monkeypatch.setattr(app_module, "RUNS", tmp_path / "runs", raising=False)

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="JPEG")
    data = buf.getvalue()
    pair = ImagePair(
        set_id="tier1-mode-forwarding",
        image_a=ImageUpload("a.jpg", data),
        image_b=ImageUpload("b.jpg", data),
    )
    recognizer = FakeRecognizer()

    app_module.recognize_and_persist(
        recognizer,  # type: ignore[arg-type]
        pair,
        hull_label_tier1_mode="shadow",
    )

    assert recognizer.mode == "shadow"


def test_constrained_env_default_runs_shadow_when_query_omitted(tmp_path, monkeypatch):
    import app as app_module

    class FakeRecognizer:
        def __init__(self):
            self.modes: List[Optional[str]] = []

        def recognize(self, image_a, image_b, *, hull_label_tier1_mode=None):
            self.modes.append(hull_label_tier1_mode)
            return RecognitionResult(status="success", state="U" * 54, confidence=0.8, reason="legacy")

    recognizer = FakeRecognizer()
    monkeypatch.setattr(app_module, "RUNS", tmp_path / "runs", raising=False)
    monkeypatch.setenv(app_module.CONSTRAINED_INFERENCE_MODE_ENV, "shadow")
    monkeypatch.setattr(
        app_module,
        "prepare_llm_rectified_input",
        lambda _a, _b, **_kwargs: _constrained_payload(SOLVED_STATE),
    )

    pair = ImagePair("env-constrained-shadow", ImageUpload("a.jpg", b"a"), ImageUpload("b.jpg", b"b"))
    payload = app_module.recognize_and_persist(
        recognizer,  # type: ignore[arg-type]
        pair,
    )

    assert recognizer.modes == ["off"]
    assert payload["state"] == "U" * 54
    signal = payload["recognitionSignals"]["constrainedInference"]
    assert signal["selected"] is False
    assert signal["promotionGate"]["accepted"] is True


def test_explicit_mode_overrides_constrained_env_default(tmp_path, monkeypatch):
    import app as app_module

    class FakeRecognizer:
        def __init__(self):
            self.modes: List[Optional[str]] = []

        def recognize(self, image_a, image_b, *, hull_label_tier1_mode=None):
            self.modes.append(hull_label_tier1_mode)
            return RecognitionResult(status="rejected", reason="legacy")

    recognizer = FakeRecognizer()
    monkeypatch.setattr(app_module, "RUNS", tmp_path / "runs", raising=False)
    monkeypatch.setenv(app_module.CONSTRAINED_INFERENCE_MODE_ENV, "constrained")

    pair = ImagePair("explicit-off", ImageUpload("a.jpg", b"a"), ImageUpload("b.jpg", b"b"))
    app_module.recognize_and_persist(
        recognizer,  # type: ignore[arg-type]
        pair,
        hull_label_tier1_mode="off",
    )

    assert recognizer.modes == ["off"]


def _constrained_payload(state: str = SOLVED_STATE, *, accepted: bool = True):
    return {
        "status": "success",
        "yawQuarterTurns": 0,
        "yawSource": "center-inference",
        "yawInference": {"accepted": True, "yawQuarterTurns": 0},
        "hullLabelPairThresholdSelection": {
            "selectionReason": "kept_current_valid_repair",
            "currentRepairValid": True,
            "selectedRepairValid": accepted,
            "currentThresholds": {"A": 160, "B": 160},
            "selectedThresholds": {"A": 160, "B": 160},
            "selectedProductionRank": [0, 2, 0.0, 2, 0],
        },
        "constrainedInferencePromotionGate": {
            "accepted": accepted,
            "decision": "auto_return_candidate" if accepted else "fallback_or_manual_review",
            "rejectReasons": [] if accepted else ["test_reject"],
        },
        "deterministicColorRepair": {
            "status": "assembled",
            "recommendedMethod": "canonical_count_repaired",
            "recommended": {
                "state": state,
                "validState": accepted,
                "countBalanced": accepted,
                "confidence": "high",
                "repairMoveCount": 2,
            },
            "methods": {
                "two_view_consistency_repaired": {
                    "status": "accepted_two_view_consistency_repair",
                    "validState": True,
                    "countBalanced": True,
                    "repairCost": 3.5,
                    "repairChanges": 2,
                    "gate": {
                        "accepted": True,
                        "reasons": [],
                        "stateDeltaFromCanonical": {
                            "available": True,
                            "count": 2,
                        },
                        "baselineCubieConsistency": {
                            "totalCubies": 20,
                            "consistentCount": 19,
                            "inconsistentCount": 1,
                            "inconsistentSplitCount": 1,
                            "inconsistentInImageCount": 0,
                            "inconsistentNames": ["UFL"],
                            "inconsistentCubies": [{"name": "UFL"}],
                        },
                        "candidateCubieConsistency": {
                            "totalCubies": 20,
                            "consistentCount": 20,
                            "inconsistentCount": 0,
                            "inconsistentSplitCount": 0,
                            "inconsistentInImageCount": 0,
                            "inconsistentNames": [],
                        },
                    },
                },
            },
        },
    }


def test_constrained_shadow_mode_returns_legacy_with_shadow_signal(tmp_path, monkeypatch):
    import app as app_module

    class FakeRecognizer:
        def __init__(self):
            self.modes: List[Optional[str]] = []

        def recognize(self, image_a, image_b, *, hull_label_tier1_mode=None):
            self.modes.append(hull_label_tier1_mode)
            return RecognitionResult(status="success", state="U" * 54, confidence=0.8, reason="legacy")

    monkeypatch.setattr(app_module, "RUNS", tmp_path / "runs", raising=False)
    monkeypatch.setattr(
        app_module,
        "prepare_llm_rectified_input",
        lambda _a, _b, **_kwargs: _constrained_payload(SOLVED_STATE),
    )

    pair = ImagePair("constrained-shadow", ImageUpload("a.jpg", b"a"), ImageUpload("b.jpg", b"b"))
    payload = app_module.recognize_and_persist(
        FakeRecognizer(),  # type: ignore[arg-type]
        pair,
        hull_label_tier1_mode="constrained-shadow",
        expected_state=SOLVED_STATE,
    )

    assert payload["state"] == "U" * 54
    signal = payload["recognitionSignals"]["constrainedInference"]
    assert signal["selected"] is False
    assert signal["promotionGate"]["accepted"] is True
    assert signal["twoViewConsistencyRepair"]["gate"]["accepted"] is True
    assert "inconsistentCubies" not in signal["twoViewConsistencyRepair"]["gate"]["baselineCubieConsistency"]

    log_path = tmp_path / "runs" / "constrained_inference_shadow.jsonl"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    event = events[0]
    assert event["schema"] == "constrained_inference_shadow_event_v1"
    assert event["mode"] == "shadow"
    assert event["setId"] == "constrained-shadow"
    assert event["runId"] == payload["runId"]
    assert event["result"]["status"] == "success"
    assert event["constrainedInference"]["selected"] is False
    assert event["constrainedInference"]["promotionGate"]["accepted"] is True
    assert event["constrainedInference"]["candidateEvaluation"]["available"] is True
    assert event["constrainedInference"]["candidateEvaluation"]["exact"] is True
    assert event["constrainedInference"]["candidateEvaluation"]["hamming"] == 0
    assert event["constrainedInference"]["pairThresholdSelection"]["selectedThresholds"] == {
        "A": 160,
        "B": 160,
    }


def test_constrained_shadow_log_can_be_disabled(tmp_path, monkeypatch):
    import app as app_module

    class FakeRecognizer:
        def recognize(self, image_a, image_b, *, hull_label_tier1_mode=None):
            return RecognitionResult(status="success", state="U" * 54, confidence=0.8, reason="legacy")

    monkeypatch.setattr(app_module, "RUNS", tmp_path / "runs", raising=False)
    monkeypatch.setenv(app_module.CONSTRAINED_SHADOW_LOG_ENV, "off")
    monkeypatch.setattr(
        app_module,
        "prepare_llm_rectified_input",
        lambda _a, _b, **_kwargs: _constrained_payload("R" * 54),
    )

    pair = ImagePair("constrained-shadow-disabled", ImageUpload("a.jpg", b"a"), ImageUpload("b.jpg", b"b"))
    app_module.recognize_and_persist(
        FakeRecognizer(),  # type: ignore[arg-type]
        pair,
        hull_label_tier1_mode="constrained-shadow",
    )

    assert not (tmp_path / "runs" / "constrained_inference_shadow.jsonl").exists()


def test_constrained_prefer_mode_returns_candidate_when_gate_accepts(tmp_path, monkeypatch):
    import app as app_module

    class FakeRecognizer:
        def __init__(self):
            self.modes: List[Optional[str]] = []

        def recognize(self, image_a, image_b, *, hull_label_tier1_mode=None):
            self.modes.append(hull_label_tier1_mode)
            return RecognitionResult(status="rejected", reason="legacy rejected")

    prepare_kwargs: List[Dict[str, object]] = []

    def fake_prepare(_a, _b, **kwargs):
        prepare_kwargs.append(kwargs)
        return _constrained_payload(SOLVED_STATE)

    recognizer = FakeRecognizer()
    monkeypatch.setattr(app_module, "RUNS", tmp_path / "runs", raising=False)
    monkeypatch.setattr(app_module, "prepare_llm_rectified_input", fake_prepare)

    pair = ImagePair("constrained-prefer", ImageUpload("a.jpg", b"a"), ImageUpload("b.jpg", b"b"))
    payload = app_module.recognize_and_persist(
        recognizer,  # type: ignore[arg-type]
        pair,
        hull_label_tier1_mode="constrained",
        expected_state=SOLVED_STATE,
    )

    assert recognizer.modes == []
    assert prepare_kwargs == [{"include_contact_sheets": False}]
    assert payload["status"] == "success"
    assert payload["state"] == SOLVED_STATE
    signal = payload["recognitionSignals"]["constrainedInference"]
    assert signal["selected"] is True
    assert signal["promotionGate"]["accepted"] is True
    assert signal["candidateEvaluation"]["exact"] is True
    assert signal["performance"]["schema"] == "constrained_recognize_performance_v1"
    assert signal["performance"]["contactSheetsIncluded"] is False
    assert "legacyFallback" not in signal["performance"]["stageTimingsMs"]

    event = json.loads((tmp_path / "runs" / "constrained_inference_shadow.jsonl").read_text(encoding="utf-8"))
    assert event["mode"] == "prefer"
    assert event["constrainedInference"]["selected"] is True
    assert event["constrainedInference"]["recommendedMethod"] == "canonical_count_repaired"
    assert event["constrainedInference"]["candidateEvaluation"]["exact"] is True


def test_constrained_prefer_mode_falls_back_when_gate_rejects(tmp_path, monkeypatch):
    import app as app_module

    class FakeRecognizer:
        def __init__(self):
            self.modes: List[Optional[str]] = []

        def recognize(self, image_a, image_b, *, hull_label_tier1_mode=None):
            self.modes.append(hull_label_tier1_mode)
            return RecognitionResult(status="success", state="U" * 54, confidence=0.8, reason="legacy")

    recognizer = FakeRecognizer()
    monkeypatch.setattr(app_module, "RUNS", tmp_path / "runs", raising=False)
    monkeypatch.setattr(
        app_module,
        "prepare_llm_rectified_input",
        lambda _a, _b, **_kwargs: _constrained_payload(accepted=False),
    )

    pair = ImagePair("constrained-fallback", ImageUpload("a.jpg", b"a"), ImageUpload("b.jpg", b"b"))
    payload = app_module.recognize_and_persist(
        recognizer,  # type: ignore[arg-type]
        pair,
        hull_label_tier1_mode="constrained-prefer",
    )

    assert recognizer.modes == ["off"]
    assert payload["state"] == "U" * 54
    signal = payload["recognitionSignals"]["constrainedInference"]
    assert signal["selected"] is False
    assert signal["promotionGate"]["accepted"] is False
    assert signal["performance"]["stageTimingsMs"]["legacyFallback"] >= 0


def _analysis_with_hull_label_centers(side: str, yaw: int) -> ImageAnalysis:
    assignments = wca_face_by_slot(side, yaw)
    return ImageAnalysis(
        width=10,
        height=10,
        roi=(0, 0, 10, 10),
        stickers=[],
        grids=[],
        overlay_data_url="",
        warnings=[],
        hull_label_tier1={
            "mode": "shadow",
            "side": side,
            "status": "accepted",
            "accepted": True,
            "selected": False,
            "slot_center_faces": {
                slot: {"face": face, "color": face, "rgb": [0, 0, 0]}
                for slot, face in assignments.items()
            },
        },
    )


def test_base_signals_include_hull_label_center_yaw_inference():
    signals = _base_recognition_signals(
        _analysis_with_hull_label_centers("A", 2),
        _analysis_with_hull_label_centers("B", 2),
    )

    yaw = signals["hullLabelTier1Yaw"]
    assert yaw["source"] == "hull_label_center_colors"
    assert yaw["status"] == "accepted"
    assert yaw["accepted"] is True
    assert yaw["yawQuarterTurns"] == 2
    assert yaw["bestScore"] == 6
    assert yaw["margin"] == 4


SLOT_QUADS = {
    "upper": [(0, 0), (10, 0), (10, 10), (0, 10)],
    "right": [(20, 0), (30, 0), (30, 10), (20, 10)],
    "front": [(-20, 0), (-10, 0), (-10, 10), (-20, 10)],
}


def _grid_for_slot(slot: str, center_x: float) -> FaceGrid:
    return FaceGrid(
        id=10_000,
        stickers=[
            [f"{slot}-0", f"{slot}-1", f"{slot}-2"],
            [f"{slot}-3", f"{slot}-4", f"{slot}-5"],
            [f"{slot}-6", f"{slot}-7", f"{slot}-8"],
        ],  # type: ignore[arg-type]
        points=[
            [(center_x - 1, 0), (center_x, 0), (center_x + 1, 0)],
            [(center_x - 1, 1), (center_x, 1), (center_x + 1, 1)],
            [(center_x - 1, 2), (center_x, 2), (center_x + 1, 2)],
        ],
        matched_count=9,
        fit_error=0.0,
        cube_hull_source="hull_label_tier1",
        hull_label_slot=slot,
    )


def _selected_hull_label_analysis(side: str, yaw: int) -> ImageAnalysis:
    assignments = wca_face_by_slot(side, yaw)
    return ImageAnalysis(
        width=10,
        height=10,
        roi=(0, 0, 10, 10),
        stickers=[],
        grids=[
            _grid_for_slot("front", 10),
            _grid_for_slot("upper", 20),
            _grid_for_slot("right", 30),
        ],
        overlay_data_url="",
        warnings=[],
        hull_label_tier1={
            "mode": "prefer",
            "side": side,
            "status": "accepted",
            "accepted": True,
            "selected": True,
            "face_quads_by_slot": SLOT_QUADS,
            "slot_center_faces": {
                slot: {"face": face, "color": face, "rgb": [0, 0, 0]}
                for slot, face in assignments.items()
            },
        },
    )


def test_hull_label_direct_options_use_inferred_yaw_and_slot_convention():
    option_a, option_b = _hull_label_direct_options(
        _selected_hull_label_analysis("A", 2),
        _selected_hull_label_analysis("B", 2),
    )

    assert option_a is not None
    assert option_b is not None
    assert {face for face in option_a if face in "URFDLB"} == {"U", "L", "B"}
    assert {face for face in option_b if face in "URFDLB"} == {"D", "F", "R"}
    assert option_a["_score"] == HULL_LABEL_DIRECT_OPTION_SCORE
    assert option_b["_score"] == HULL_LABEL_DIRECT_OPTION_SCORE
    assert option_a["_ordered_side_pair"] == ("B", "L")
    assert option_b["_ordered_side_pair"] == ("R", "F")
    assert option_a["U"][1][1] == "upper-4"
    assert option_b["D"][1][1] == "upper-4"
