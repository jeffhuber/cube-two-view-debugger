from __future__ import annotations

import io
from typing import List, Optional

from PIL import Image

from rubik_recognizer.dataset import ImagePair, ImageUpload
from rubik_recognizer.recognizer import RecognitionResult, WhiteUpRecognizer


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
