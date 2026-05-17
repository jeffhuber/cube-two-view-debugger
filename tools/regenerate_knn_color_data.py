#!/usr/bin/env python3
"""Regenerate the dependency-free KNN5 color-classifier constants."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubik_recognizer.colors import COLOR_ORDER, rgb_to_lab  # noqa: E402


DEFAULT_INPUT = ROOT / "runs" / "color_samples_geom.jsonl"
DEFAULT_OUTPUT = ROOT / "rubik_recognizer" / "knn_color_data.py"


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        if row.get("gtColor") not in COLOR_ORDER:
            raise ValueError(f"Invalid gtColor in row: {row!r}")
        rgb = row.get("rgb")
        if not isinstance(rgb, list) or len(rgb) != 3:
            raise ValueError(f"Invalid rgb in row: {row!r}")
    return rows


def _mean(columns: Sequence[Sequence[float]], index: int) -> float:
    return sum(row[index] for row in columns) / len(columns)


def _scale(columns: Sequence[Sequence[float]], index: int, mean: float) -> float:
    variance = sum((row[index] - mean) ** 2 for row in columns) / len(columns)
    scale = variance ** 0.5
    return scale + 1e-9 if scale > 0.0 else 1.0


def _chunks(text: str, size: int = 96) -> Iterable[str]:
    for index in range(0, len(text), size):
        yield text[index : index + size]


def _format_tuple(values: Sequence[float], indent: str = "    ") -> str:
    lines = [f"{indent}{value:.12f}," for value in values]
    return "(\n" + "\n".join(lines) + "\n)"


def _format_samples(samples: Sequence[Tuple[float, float, float]]) -> str:
    lines = [f"    ({sample[0]:.6f}, {sample[1]:.6f}, {sample[2]:.6f})," for sample in samples]
    return "(\n" + "\n".join(lines) + "\n)"


def build_module(rows: Sequence[Dict[str, Any]], source: Path) -> str:
    labs = [rgb_to_lab(tuple(row["rgb"])) for row in rows]
    means = tuple(_mean(labs, index) for index in range(3))
    scales = tuple(_scale(labs, index, means[index]) for index in range(3))
    samples = [
        tuple((lab[index] - means[index]) / scales[index] for index in range(3))
        for lab in labs
    ]
    color_index = {color: str(index) for index, color in enumerate(COLOR_ORDER)}
    labels = "".join(color_index[row["gtColor"]] for row in rows)
    counts = dict(sorted(Counter(row["gtColor"] for row in rows).items()))
    sets = tuple(sorted({str(row["setId"]) for row in rows}, key=lambda value: int(value) if value.isdigit() else value))

    label_lines = "\n".join(f"    {chunk!r}" for chunk in _chunks(labels))
    return f'''"""Generated KNN5 color-classifier constants.

Generated from `{source}`, produced by:
    .venv/bin/python tools/extract_clean_dataset.py --output {source}
    .venv/bin/python tools/regenerate_knn_color_data.py --input {source}
Source dataset: {len(rows)} clean hull-label samples across {len(sets)} sets.
Do not edit individual samples by hand; regenerate from the clean-label pipeline instead.
"""

from __future__ import annotations

KNN5_SOURCE_SETS = {sets!r}
KNN5_SAMPLE_COUNT = {len(rows)}
KNN5_COLOR_COUNTS = {counts!r}
KNN5_LAB_MEAN = {_format_tuple(means)}
KNN5_LAB_SCALE = {_format_tuple(scales)}
KNN5_LAB_SAMPLES = {_format_samples(samples)}
KNN5_LAB_LABELS = (
{label_lines}
)
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = _load_rows(input_path)
    output_path.write_text(build_module(rows, input_path) + "\n")
    print(f"wrote {output_path} from {len(rows)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
