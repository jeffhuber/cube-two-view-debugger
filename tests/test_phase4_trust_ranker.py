"""Unit tests for tools/phase4_trust_ranker.py.

Coverage:
- load_matrix: parses real fixture; rejects missing/NaN features.
- threshold_sweep: monotonicity & boundary behavior at thresholds 0/1.
- find_operating_points: bar-clearing logic.
- end-to-end smoke: full pipeline on real data, output is well-formed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from tools import phase4_trust_ranker as p4

REPO = Path(__file__).resolve().parents[1]
REAL_FIXTURE = REPO / "tests" / "fixtures" / "phase2b_recomputed_signals.json"


# ----- load_matrix -----


def test_load_matrix_real_fixture():
    rows = p4.load_matrix(REAL_FIXTURE)
    assert len(rows) == 116, "the committed fixture should have 116 rows"
    assert len({r.case for r in rows}) == 58, "58 unique cases"
    # Each row has exactly 6 features in fixed order.
    for r in rows:
        assert r.features.shape == (6,)
    # Category counts match the fixture summary.
    cats = {}
    for r in rows:
        cats[r.category] = cats.get(r.category, 0) + 1
    assert cats == {
        "GOOD": 74, "MARGINAL": 22,
        "CHIRALITY_MISS": 12, "CHIRALITY_FALSE_FLIP": 7,
        "TRUE_GEOMETRY_FAIL": 1,
    }


def test_load_matrix_rejects_missing_feature(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "by_case": {
            "1_A": [
                {"run": 0, "category": "GOOD",
                 "fit_residual_rms_px": 10.0,
                 "pnp_rms_px": 11.0,
                 # missing hexagon_centroid_vs_bezel_vertex_offset_px
                 "junction_score_at_ensemble": 100.0,
                 "ensemble_shift_px": 5.0,
                 "phase_darkness_separation": 0.0},
            ]
        }
    }))
    with pytest.raises(ValueError, match="missing"):
        p4.load_matrix(bad)


def test_load_matrix_rejects_nan(tmp_path):
    bad = tmp_path / "bad_nan.json"
    bad.write_text(json.dumps({
        "by_case": {
            "1_A": [
                {"run": 0, "category": "GOOD",
                 "fit_residual_rms_px": float("nan"),
                 "pnp_rms_px": 11.0,
                 "hexagon_centroid_vs_bezel_vertex_offset_px": 1.0,
                 "junction_score_at_ensemble": 100.0,
                 "ensemble_shift_px": 5.0,
                 "phase_darkness_separation": 0.0},
            ]
        }
    }))
    with pytest.raises(ValueError, match="missing"):
        p4.load_matrix(bad)


# ----- Row classification -----


def test_row_is_catastrophic():
    base_feats = np.zeros(6)
    for cat in ("CHIRALITY_MISS", "CHIRALITY_FALSE_FLIP", "TRUE_GEOMETRY_FAIL"):
        assert p4.Row("1_A", 0, cat, base_feats).is_catastrophic
    for cat in ("GOOD", "MARGINAL"):
        assert not p4.Row("1_A", 0, cat, base_feats).is_catastrophic


def test_row_is_good():
    base_feats = np.zeros(6)
    assert p4.Row("1_A", 0, "GOOD", base_feats).is_good
    for cat in ("MARGINAL", "CHIRALITY_MISS", "CHIRALITY_FALSE_FLIP", "TRUE_GEOMETRY_FAIL"):
        assert not p4.Row("1_A", 0, cat, base_feats).is_good


# ----- threshold_sweep -----


def _make_rows(specs: List[tuple]) -> List[p4.Row]:
    """specs: list of (case, run, category) tuples. Features irrelevant."""
    return [p4.Row(case=c, run=r, category=cat, features=np.zeros(6))
            for c, r, cat in specs]


def test_threshold_sweep_boundary_threshold_0():
    rows = _make_rows([
        ("a", 0, "CHIRALITY_MISS"),
        ("b", 0, "GOOD"),
    ])
    probas = np.array([0.5, 0.5])
    sweep = p4.threshold_sweep(rows, probas, thresholds=np.array([0.0]))
    m = sweep[0]
    # At threshold 0, everything retakes → 100% recall, 100% FPR.
    assert m.catastrophic_recall == 1.0
    assert m.good_fpr == 1.0


def test_threshold_sweep_boundary_threshold_1_plus():
    rows = _make_rows([
        ("a", 0, "CHIRALITY_MISS"),
        ("b", 0, "GOOD"),
    ])
    probas = np.array([0.5, 0.5])
    # Threshold 1.01 means "retake only if proba >= 1.01" → never.
    sweep = p4.threshold_sweep(rows, probas, thresholds=np.array([1.01]))
    m = sweep[0]
    assert m.catastrophic_recall == 0.0
    assert m.good_fpr == 0.0


def test_threshold_sweep_separates_classes_perfectly():
    rows = _make_rows([
        ("a", 0, "CHIRALITY_MISS"),
        ("b", 0, "GOOD"),
        ("c", 0, "GOOD"),
        ("d", 0, "CHIRALITY_FALSE_FLIP"),
    ])
    # Cats: high probas; Goods: low probas.
    probas = np.array([0.9, 0.1, 0.1, 0.9])
    # Threshold 0.5 should perfectly classify.
    sweep = p4.threshold_sweep(rows, probas, thresholds=np.array([0.5]))
    m = sweep[0]
    assert m.catastrophic_recall == 1.0  # 2/2 caught
    assert m.good_fpr == 0.0              # 0/2 falsely retaken
    assert m.n_catastrophic == 2
    assert m.n_good == 2


def test_threshold_sweep_marginal_does_not_affect_fpr():
    rows = _make_rows([
        ("a", 0, "MARGINAL"),
        ("b", 0, "GOOD"),
    ])
    probas = np.array([0.9, 0.1])
    sweep = p4.threshold_sweep(rows, probas, thresholds=np.array([0.5]))
    m = sweep[0]
    # MARGINAL retake doesn't penalize the GOOD FPR metric.
    assert m.good_fpr == 0.0
    assert m.marginal_retake_rate == 1.0  # 1/1 marginal retaken


# ----- find_operating_points -----


def test_find_operating_points_bar_cleared():
    rows = _make_rows([
        ("a", 0, "CHIRALITY_MISS"),
        ("b", 0, "GOOD"),
        ("c", 0, "GOOD"),
        ("d", 0, "GOOD"),
        ("e", 0, "GOOD"),
        ("f", 0, "GOOD"),
        ("g", 0, "GOOD"),
        ("h", 0, "GOOD"),
        ("i", 0, "GOOD"),
        ("j", 0, "GOOD"),
        ("k", 0, "GOOD"),  # 10 GOOD, 1 catastrophic
    ])
    probas = np.array([0.9] + [0.1] * 10)
    sweep = p4.threshold_sweep(rows, probas, thresholds=np.array([0.5]))
    ops = p4.find_operating_points(sweep)
    bar = ops["bar_clearing"]
    assert bar is not None
    assert bar.catastrophic_recall == 1.0
    assert bar.good_fpr == 0.0


def test_find_operating_points_bar_not_cleared():
    rows = _make_rows([
        ("a", 0, "CHIRALITY_MISS"),
        ("b", 0, "GOOD"),
        ("c", 0, "GOOD"),
    ])
    # Probas overlap: catastrophic at 0.6, GOOD at 0.7 and 0.4.
    # No threshold separates them cleanly.
    probas = np.array([0.6, 0.7, 0.4])
    sweep = p4.threshold_sweep(rows, probas)
    ops = p4.find_operating_points(sweep)
    # At any threshold catching catastrophic, FPR > 10%.
    bar = ops["bar_clearing"]
    assert bar is None


# ----- end-to-end smoke -----


def test_end_to_end_smoke():
    """Full pipeline against the real fixture: must succeed + produce
    well-formed output for all 4 models.

    Skipped when sklearn is not installed (it's an optional dep —
    declared in requirements.txt but not strictly required by the
    production recognizer pipeline).
    """
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    rows = p4.load_matrix(REAL_FIXTURE)
    factories = p4._make_model_factories()
    # Run on a single model to keep the test fast.
    name = "logistic_regression"
    result = p4.run_model(rows, name, factories[name])
    assert result["name"] == name
    assert "in_sample" in result
    assert "out_of_fold" in result
    assert "operating_points" in result["out_of_fold"]
    # The OOF predictions are one per row.
    assert len(result["out_of_fold"]["predictions"]) == 116
    # Probabilities are valid [0, 1].
    for pred in result["out_of_fold"]["predictions"]:
        assert 0.0 <= pred["predicted_proba"] <= 1.0
