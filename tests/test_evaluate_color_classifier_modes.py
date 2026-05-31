from __future__ import annotations

from pathlib import Path
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


def _sample_row(set_id: str, rgb: list[int], gt_color: str, *, row: int = 0, col: int = 0):
    return {
        "setId": set_id,
        "side": "A",
        "labelName": "U",
        "face": "U",
        "row": row,
        "col": col,
        "isCenter": False,
        "rgb": rgb,
        "calibratedClassifier": "white",
        "classifierModes": {
            "canonical_adaptive": "white",
            "knn5_lab_adaptive": "white",
            "knn5_lab_full_adaptive": "white",
        },
        "gtColor": gt_color,
    }


def _report_for_rows(rows):
    report = {
        "input": "test.jsonl",
        "sampleCount": len(rows),
        "setCount": len({row["setId"] for row in rows}),
        "colorCounts": {},
        "modes": {},
        "headToHead": {},
    }
    for mode in evaluator.MODE_ORDER:
        report["modes"][mode] = evaluator._mode_report(rows, mode)
    baseline = report["modes"]["canonical"]["perSetAccuracy"]
    for mode in evaluator.MODE_ORDER[1:]:
        report["headToHead"][mode] = {
            "perSetDelta": evaluator._deltas(report["modes"][mode]["perSetAccuracy"], baseline),
            **evaluator._head_to_head_samples(rows, mode),
            "wins": 0,
            "losses": 0,
            "ties": 0,
        }
    return report


def test_mode_report_surfaces_ranked_mismatches_and_confusion_pairs():
    rows = [
        _sample_row("1", [245, 245, 245], "white"),
        _sample_row("1", [245, 245, 245], "red", row=0, col=1),
        _sample_row("2", [144, 72, 49], "red", row=1, col=0),
    ]

    report = evaluator._mode_report(rows, "canonical")

    assert {"expected": "red", "predicted": "white", "count": 1} in report["confusionPairs"]
    assert report["highConfidenceMismatches"][0]["gtColor"] == "red"
    assert report["highConfidenceMismatches"][0]["predictedColor"] == "white"
    assert report["weakestSamples"][0]["sample"].startswith("2_A")


def test_markdown_report_and_contact_sheets_render(tmp_path: Path):
    rows = [
        _sample_row("1", [245, 245, 245], "white"),
        _sample_row("1", [245, 245, 245], "red", row=0, col=1),
        _sample_row("2", [144, 72, 49], "red", row=1, col=0),
    ]
    report = _report_for_rows(rows)

    sheets = evaluator.write_contact_sheets(report, tmp_path)
    report["contactSheets"] = sheets
    markdown = evaluator.render_markdown_report(report)

    assert "# Color classifier mode report" in markdown
    assert "Highest-confidence canonical mismatches" in markdown
    assert "Contact sheets" in markdown
    assert sheets
    for path in sheets.values():
        assert Path(path).exists()
