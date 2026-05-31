from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import CLASSIFIER_KNN5_LAB, CLASSIFIER_MODE_ENV  # noqa: E402
from tools import train_color_classifier  # noqa: E402


def test_baseline_canonical_is_pinned_to_canonical_classifier(monkeypatch):
    monkeypatch.setenv(CLASSIFIER_MODE_ENV, CLASSIFIER_KNN5_LAB)
    model = train_color_classifier.BaselineCanonical()
    rgb = np.array([[144, 72, 49]], dtype=np.float64)

    prediction = model.predict(rgb)

    assert prediction.tolist() == [train_color_classifier.COLOR_INDEX["orange"]]
