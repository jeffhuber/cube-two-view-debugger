#!/usr/bin/env python3
"""Score hybrid overlay slots for cell-internal discontinuity diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_hybrid_pipeline import _load_processing_image, _proposer_face_quads  # noqa: E402
from tools.overlay_feedback import DEFAULT_OUTPUT as DEFAULT_LABELS  # noqa: E402
from tools.rectify_faces import DEFAULT_FACE_SIZE, rectify_face  # noqa: E402


DEFAULT_MANIFEST = ROOT / "tests" / "fixtures" / "hard_case_manifest.json"
DEFAULT_OUTPUT = ROOT / "tests" / "fixtures" / "hard_case_visual_feedback_discontinuity.json"
DEFAULT_REPORT = ROOT / "tools" / "OVERLAY_DISCONTINUITY_REPORT.md"

HIGH_STD_THRESHOLD = 35.0
HIGH_HALF_DELTA_THRESHOLD = 45.0


def cell_discontinuity_metrics(face_image: Any) -> Dict[str, Any]:
    """Measure within-cell color variance on a rectified 3x3 face image."""
    arr = np.asarray(face_image.convert("RGB"), dtype=float)
    height, width = arr.shape[:2]
    cell_w = width / 3.0
    cell_h = height / 3.0
    cells = []
    adjacent_deltas = []
    cell_means: List[List[np.ndarray]] = []
    for row in range(3):
        mean_row = []
        for col in range(3):
            patch = _cell_patch(arr, row, col, cell_w, cell_h)
            metrics = _patch_discontinuity(patch)
            metrics.update({"row": row, "col": col})
            cells.append(metrics)
            mean_row.append(np.array(metrics["meanRgb"], dtype=float))
        cell_means.append(mean_row)

    for row in range(3):
        for col in range(3):
            if col < 2:
                adjacent_deltas.append(_distance(cell_means[row][col], cell_means[row][col + 1]))
            if row < 2:
                adjacent_deltas.append(_distance(cell_means[row][col], cell_means[row + 1][col]))

    max_internal_std = max((cell["internalStd"] for cell in cells), default=0.0)
    max_half_delta = max((cell["maxHalfDelta"] for cell in cells), default=0.0)
    mean_internal_std = sum(cell["internalStd"] for cell in cells) / max(1, len(cells))
    mean_half_delta = sum(cell["maxHalfDelta"] for cell in cells) / max(1, len(cells))
    score = (
        max_internal_std / HIGH_STD_THRESHOLD
        + max_half_delta / HIGH_HALF_DELTA_THRESHOLD
        + sum(1 for cell in cells if cell["internalStd"] >= HIGH_STD_THRESHOLD) * 0.35
        + sum(1 for cell in cells if cell["maxHalfDelta"] >= HIGH_HALF_DELTA_THRESHOLD) * 0.35
    )
    return {
        "schemaVersion": 1,
        "policy": "diagnostics_only_no_behavior_change",
        "cellCount": len(cells),
        "meanInternalStd": round(mean_internal_std, 3),
        "maxInternalStd": round(max_internal_std, 3),
        "meanHalfDelta": round(mean_half_delta, 3),
        "maxHalfDelta": round(max_half_delta, 3),
        "cellsAboveInternalStdThreshold": sum(
            1 for cell in cells if cell["internalStd"] >= HIGH_STD_THRESHOLD
        ),
        "cellsAboveHalfDeltaThreshold": sum(
            1 for cell in cells if cell["maxHalfDelta"] >= HIGH_HALF_DELTA_THRESHOLD
        ),
        "maxAdjacentMeanDelta": round(max(adjacent_deltas, default=0.0), 3),
        "score": round(score, 3),
        "thresholds": {
            "highInternalStd": HIGH_STD_THRESHOLD,
            "highHalfDelta": HIGH_HALF_DELTA_THRESHOLD,
        },
        "cells": cells,
    }


def probe_overlay_discontinuity(
    labels: Dict[str, Any],
    manifest: Dict[str, Any],
    *,
    hull_guard: bool = False,
    fit_error_fallback: bool = False,
) -> Dict[str, Any]:
    manifest_by_set = {str(row["setId"]): row for row in manifest.get("pairs", [])}
    rows = []
    for item in labels.get("sets", []):
        set_id = str(item.get("setId"))
        manifest_row = manifest_by_set.get(set_id)
        if not manifest_row:
            rows.append({"setId": set_id, "error": "set_not_in_hard_case_manifest"})
            continue
        image_by_side = {
            "A": Path(manifest_row["imageAPath"]),
            "B": Path(manifest_row["imageBPath"]),
        }
        slots_by_side: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": []}
        for slot in item.get("slots", []):
            slots_by_side.setdefault(slot["side"], []).append(slot)

        for side, slots in slots_by_side.items():
            if not slots:
                continue
            image_path = image_by_side[side]
            try:
                image, _ = _load_processing_image(image_path)
                quads, debug = _proposer_face_quads(
                    image_path,
                    side,
                    hull_guard=hull_guard,
                    fit_error_fallback=fit_error_fallback,
                    processing_image=image,
                )
            except Exception as exc:  # pragma: no cover - exercised in CLI environments
                rows.append(
                    {
                        "setId": set_id,
                        "side": side,
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )
                continue
            for slot in slots:
                slot_label = slot["slot"]
                quad = quads.get(slot_label)
                selected = debug.get("selectedPerFace", {}).get(slot_label, {})
                if quad is None:
                    rows.append({**_label_fields(set_id, slot, selected), "error": "quad_not_found"})
                    continue
                try:
                    rectified = rectify_face(image, quad, output_size=DEFAULT_FACE_SIZE)
                    metrics = cell_discontinuity_metrics(rectified)
                except Exception as exc:  # pragma: no cover - exercised in CLI environments
                    rows.append(
                        {
                            **_label_fields(set_id, slot, selected),
                            "error": f"{exc.__class__.__name__}: {exc}",
                        }
                    )
                    continue
                rows.append(
                    {
                        **_label_fields(set_id, slot, selected),
                        "metrics": {
                            key: value
                            for key, value in metrics.items()
                            if key not in {"cells"}
                        },
                    }
                )

    summary = discontinuity_summary(rows)
    return {
        "schemaVersion": 1,
        "policy": "diagnostics_only_no_behavior_change",
        "labelsSource": labels.get("source", {}),
        "probeConfig": {
            "hullGuard": hull_guard,
            "fitErrorFallback": fit_error_fallback,
        },
        "summary": summary,
        "rows": rows,
    }


def discontinuity_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [row for row in rows if isinstance(row.get("metrics"), dict)]
    bad = [row for row in scored if row.get("humanBad")]
    good = [row for row in scored if not row.get("humanBad")]
    return {
        "rowCount": len(rows),
        "scoredRowCount": len(scored),
        "humanBadRowCount": len(bad),
        "humanGoodRowCount": len(good),
        "meanScoreHumanBad": _mean_metric(bad, "score"),
        "meanScoreHumanGood": _mean_metric(good, "score"),
        "topRows": [
            _compact_row(row)
            for row in sorted(
                scored,
                key=lambda item: (
                    -float(item.get("metrics", {}).get("score", 0.0)),
                    item.get("setId", ""),
                    item.get("side", ""),
                    item.get("slot", ""),
                ),
            )[:12]
        ],
    }


def render_discontinuity_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    lines = [
        "# Overlay Cell Discontinuity Probe",
        "",
        "Diagnostics-only scoring over the same hybrid overlay slots that received human visual feedback.",
        "High scores indicate cell-internal color variance or half-cell contrast that may be consistent with a multi-face span.",
        "",
        "## Summary",
        "",
        f"- Rows scored: {summary['scoredRowCount']} / {summary['rowCount']}",
        f"- Human-bad rows: {summary['humanBadRowCount']}",
        f"- Human-good rows: {summary['humanGoodRowCount']}",
        f"- Mean score, human-bad rows: {summary['meanScoreHumanBad']}",
        f"- Mean score, human-good rows: {summary['meanScoreHumanGood']}",
        "",
        "## Highest Discontinuity Rows",
        "",
        "| Rank | Set | Slot | Human label | Score | Max std | Max half-delta | Source | Failure modes |",
        "|---:|---:|---|---|---:|---:|---:|---|---|",
    ]
    for index, row in enumerate(summary["topRows"], 1):
        modes = ", ".join(f"`{mode}`" for mode in row.get("failureModes", [])) or "`ok`"
        human = "bad" if row.get("humanBad") else "good"
        lines.append(
            f"| {index} | {row['setId']} | `{row['side']}:{row['slot']}` | {human} | "
            f"{row['score']} | {row['maxInternalStd']} | {row['maxHalfDelta']} | "
            f"{row.get('selectedSourceFace') or '?'} | {modes} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This probe is not a production guard and does not change recognition behavior.",
            "- Treat high-score human-bad rows as candidates for a future multi-face-span ranker.",
            "- Treat high-score human-good rows as potential false positives that a future guard must avoid.",
            "- The current metric is intentionally simple: rectified-cell RGB variance plus half-cell contrast.",
            "",
        ]
    )
    return "\n".join(lines)


def _label_fields(set_id: str, slot: Dict[str, Any], selected: Dict[str, Any]) -> Dict[str, Any]:
    modes = [mode for mode in slot.get("failureModes", []) if mode != "ok"]
    return {
        "setId": set_id,
        "image": slot.get("image"),
        "side": slot.get("side"),
        "slot": slot.get("slot"),
        "quadQuality": slot.get("quadQuality"),
        "rectifiedQuality": slot.get("rectifiedQuality"),
        "humanRectifiedSourceFace": slot.get("rectifiedSourceFace"),
        "selectedSourceFace": selected.get("sourceCenterFace"),
        "selectedSourcePosition": selected.get("sourcePosition"),
        "failureModes": modes,
        "humanBad": bool(modes),
    }


def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    metrics = row.get("metrics") or {}
    return {
        "setId": row.get("setId"),
        "side": row.get("side"),
        "slot": row.get("slot"),
        "humanBad": row.get("humanBad"),
        "failureModes": row.get("failureModes", []),
        "score": metrics.get("score"),
        "maxInternalStd": metrics.get("maxInternalStd"),
        "maxHalfDelta": metrics.get("maxHalfDelta"),
        "selectedSourceFace": row.get("selectedSourceFace"),
    }


def _cell_patch(arr: np.ndarray, row: int, col: int, cell_w: float, cell_h: float) -> np.ndarray:
    margin_x = cell_w * 0.14
    margin_y = cell_h * 0.14
    x0 = int(round(col * cell_w + margin_x))
    x1 = int(round((col + 1) * cell_w - margin_x))
    y0 = int(round(row * cell_h + margin_y))
    y1 = int(round((row + 1) * cell_h - margin_y))
    return arr[max(0, y0) : max(y0 + 1, y1), max(0, x0) : max(x0 + 1, x1), :]


def _patch_discontinuity(patch: np.ndarray) -> Dict[str, Any]:
    mean_rgb = patch.reshape(-1, 3).mean(axis=0)
    std_rgb = patch.reshape(-1, 3).std(axis=0)
    half_deltas = []
    height, width = patch.shape[:2]
    if width >= 2:
        half_deltas.append(
            _distance(
                patch[:, : width // 2, :].mean(axis=(0, 1)),
                patch[:, width // 2 :, :].mean(axis=(0, 1)),
            )
        )
    if height >= 2:
        half_deltas.append(
            _distance(
                patch[: height // 2, :, :].mean(axis=(0, 1)),
                patch[height // 2 :, :, :].mean(axis=(0, 1)),
            )
        )
    if width >= 2 and height >= 2:
        half_deltas.append(
            _distance(
                patch[: height // 2, : width // 2, :].mean(axis=(0, 1)),
                patch[height // 2 :, width // 2 :, :].mean(axis=(0, 1)),
            )
        )
        half_deltas.append(
            _distance(
                patch[: height // 2, width // 2 :, :].mean(axis=(0, 1)),
                patch[height // 2 :, : width // 2, :].mean(axis=(0, 1)),
            )
        )
    return {
        "meanRgb": [round(float(value), 3) for value in mean_rgb],
        "internalStd": round(_distance(std_rgb, np.zeros(3)), 3),
        "maxHalfDelta": round(max(half_deltas, default=0.0), 3),
    }


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return float(math.sqrt(float(np.dot(delta, delta))))


def _mean_metric(rows: Sequence[Dict[str, Any]], key: str) -> float:
    values = [
        float(row.get("metrics", {}).get(key, 0.0))
        for row in rows
        if isinstance(row.get("metrics", {}).get(key), (int, float))
    ]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return data


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--hull-guard", action="store_true")
    parser.add_argument("--fit-error-fallback", action="store_true")
    args = parser.parse_args(argv)

    labels = _load_json(args.labels)
    manifest = _load_json(args.manifest)
    document = probe_overlay_discontinuity(
        labels,
        manifest,
        hull_guard=args.hull_guard,
        fit_error_fallback=args.fit_error_fallback,
    )
    _write_json(args.output, document)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_discontinuity_report(document), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
