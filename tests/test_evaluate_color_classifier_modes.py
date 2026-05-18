from __future__ import annotations

from types import SimpleNamespace

from tools import evaluate_color_classifier_modes as evaluator


def test_runtime_classifier_modes_recompute_stale_cached_predictions(monkeypatch):
    calls = []

    def fake_classify(rgb, mode):
        calls.append((rgb, mode))
        return SimpleNamespace(color=f"fresh-{mode}")

    monkeypatch.setattr(evaluator, "classify_rgb_with_mode", fake_classify)
    row = {
        "rgb": [1, 2, 3],
        "classifierModes": {
            "canonical": "stale-canonical",
            "knn5_lab": "stale-knn5",
            "knn5_lab_full": "stale-knn5-full",
        },
        "gtColor": "red",
    }

    assert evaluator._mode_prediction(row, "canonical") == "fresh-canonical"
    assert evaluator._mode_prediction(row, "knn5_lab") == "fresh-knn5_lab"
    assert evaluator._mode_prediction(row, "knn5_lab_full") == "fresh-knn5_lab_full"
    assert calls == [
        ((1, 2, 3), evaluator.CLASSIFIER_CANONICAL),
        ((1, 2, 3), evaluator.CLASSIFIER_KNN5_LAB),
        ((1, 2, 3), evaluator.CLASSIFIER_KNN5_LAB_FULL),
    ]


def test_adaptive_modes_keep_extracted_cached_predictions(monkeypatch):
    def fail_if_called(rgb, mode):
        raise AssertionError("adaptive flat rows should use extracted cached predictions")

    monkeypatch.setattr(evaluator, "classify_rgb_with_mode", fail_if_called)
    row = {
        "rgb": [1, 2, 3],
        "calibratedClassifier": "cached-canonical-adaptive",
        "classifierModes": {
            "canonical_adaptive": "cached-canonical-adaptive-mode",
            "knn5_lab_adaptive": "cached-knn5-adaptive",
            "knn5_lab_full_adaptive": "cached-knn5-full-adaptive",
        },
        "gtColor": "red",
    }

    assert evaluator._mode_prediction(row, "canonical_adaptive") == "cached-canonical-adaptive-mode"
    assert evaluator._mode_prediction(row, "knn5_lab_adaptive") == "cached-knn5-adaptive"
    assert evaluator._mode_prediction(row, "knn5_lab_full_adaptive") == "cached-knn5-full-adaptive"
