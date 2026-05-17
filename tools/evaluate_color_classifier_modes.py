#!/usr/bin/env python3
"""Evaluate production color-classifier modes on clean hull-label samples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubik_recognizer.colors import (  # noqa: E402
    CLASSIFIER_CANONICAL,
    CLASSIFIER_KNN5_LAB,
    COLOR_ORDER,
    classify_rgb_with_mode,
)


DEFAULT_INPUT = ROOT / "runs" / "color_samples_geom.jsonl"
DEFAULT_OUTPUT = ROOT / "runs" / "color_classifier_modes_report.json"
MODE_ORDER = (
    "canonical",
    "canonical_adaptive",
    "knn5_lab",
    "knn5_lab_adaptive",
)


def _mode_prediction(row: Dict[str, Any], mode: str) -> str:
    modes = row.get("classifierModes")
    if isinstance(modes, dict) and isinstance(modes.get(mode), str):
        return modes[mode]
    rgb = tuple(row["rgb"])
    if mode == "canonical":
        return classify_rgb_with_mode(rgb, CLASSIFIER_CANONICAL).color
    if mode == "knn5_lab":
        return classify_rgb_with_mode(rgb, CLASSIFIER_KNN5_LAB).color
    if mode == "canonical_adaptive" and isinstance(row.get("calibratedClassifier"), str):
        return row["calibratedClassifier"]
    if mode == "canonical" and isinstance(row.get("defaultClassifier"), str):
        return row["defaultClassifier"]
    raise ValueError(
        f"Input row lacks {mode!r}; regenerate with tools/extract_clean_dataset.py "
        "from this branch for adaptive-mode evaluation."
    )


def _accuracy(rows: Sequence[Dict[str, Any]], mode: str) -> float:
    if not rows:
        return 0.0
    correct = sum(1 for row in rows if _mode_prediction(row, mode) == row["gtColor"])
    return correct / len(rows)


def _confusion(rows: Sequence[Dict[str, Any]], mode: str) -> List[List[int]]:
    index = {color: idx for idx, color in enumerate(COLOR_ORDER)}
    matrix = [[0 for _ in COLOR_ORDER] for _ in COLOR_ORDER]
    for row in rows:
        truth = row["gtColor"]
        pred = _mode_prediction(row, mode)
        matrix[index[truth]][index[pred]] += 1
    return matrix


def _per_color_accuracy(rows: Sequence[Dict[str, Any]], mode: str) -> Dict[str, float]:
    by_color: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_color[row["gtColor"]].append(row)
    return {
        color: round(_accuracy(by_color.get(color, ()), mode), 4)
        for color in COLOR_ORDER
    }


def _per_set_accuracy(rows: Sequence[Dict[str, Any]], mode: str) -> Dict[str, float]:
    by_set: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_set[str(row["setId"])].append(row)
    return {
        set_id: round(_accuracy(by_set[set_id], mode), 4)
        for set_id in sorted(by_set, key=lambda value: int(value) if value.isdigit() else value)
    }


def _mode_report(rows: Sequence[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    return {
        "overallAccuracy": round(_accuracy(rows, mode), 4),
        "perColorAccuracy": _per_color_accuracy(rows, mode),
        "perSetAccuracy": _per_set_accuracy(rows, mode),
        "confusionLabels": list(COLOR_ORDER),
        "confusionMatrix": _confusion(rows, mode),
    }


def _deltas(candidate: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    return {
        key: round(candidate.get(key, 0.0) - baseline.get(key, 0.0), 4)
        for key in baseline
    }


def _print_table(report: Dict[str, Any]) -> None:
    print("Mode                  Accuracy  Delta vs canonical")
    print("--------------------  --------  ------------------")
    canonical = report["modes"]["canonical"]["overallAccuracy"]
    for mode in MODE_ORDER:
        accuracy = report["modes"][mode]["overallAccuracy"]
        print(f"{mode:20s}  {accuracy:8.4f}  {accuracy - canonical:+18.4f}")

    for mode in MODE_ORDER[1:]:
        deltas = report["headToHead"][mode]["perSetDelta"]
        regressions = sorted(deltas.items(), key=lambda item: item[1])[:3]
        wins = sorted(deltas.items(), key=lambda item: item[1])[-3:]
        print(f"\n{mode} vs canonical:")
        print(f"  worst regressions: {regressions}")
        print(f"  biggest wins:      {wins}")


def _validate_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validated = []
    for row in rows:
        if row.get("gtColor") not in COLOR_ORDER:
            raise ValueError(f"Invalid gtColor in row: {row!r}")
        if not isinstance(row.get("rgb"), list) or len(row["rgb"]) != 3:
            raise ValueError(f"Invalid rgb in row: {row!r}")
        validated.append(row)
    return validated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--json-output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = _validate_rows(json.loads(line) for line in input_path.read_text().splitlines() if line.strip())
    if not rows:
        raise SystemExit("No clean-label samples found.")

    report: Dict[str, Any] = {
        "input": str(input_path),
        "sampleCount": len(rows),
        "setCount": len({row["setId"] for row in rows}),
        "colorCounts": dict(Counter(row["gtColor"] for row in rows)),
        "modes": {},
        "headToHead": {},
    }
    for mode in MODE_ORDER:
        report["modes"][mode] = _mode_report(rows, mode)

    baseline = report["modes"]["canonical"]["perSetAccuracy"]
    for mode in MODE_ORDER[1:]:
        per_set = report["modes"][mode]["perSetAccuracy"]
        deltas = _deltas(per_set, baseline)
        report["headToHead"][mode] = {
            "perSetDelta": deltas,
            "wins": sum(1 for delta in deltas.values() if delta > 0),
            "losses": sum(1 for delta in deltas.values() if delta < 0),
            "ties": sum(1 for delta in deltas.values() if delta == 0),
        }

    output_path = Path(args.json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _print_table(report)
    print(f"\nwrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
