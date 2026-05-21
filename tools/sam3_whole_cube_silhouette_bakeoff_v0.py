#!/usr/bin/env python3
"""Bake off SAM3 whole-cube masks against rembg silhouettes.

Diagnostics/data-only. This module does not alter recognizer behavior.

The expected inputs are paired global-model fit outputs produced from the
same images and human vertex labels:

* one directory for the rembg-driven silhouette fit
* one directory for the SAM3 whole-cube-mask-driven silhouette fit

Each row compares the fitted visible trihedral vertex against the human label.
The purpose is to decide whether SAM3 whole-cube masks are useful as a
geometry source or cross-check, not to promote them into production.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEEDBACK = Path("/Users/jhuber/Downloads/gcm_refined_feedback.json")
DEFAULT_REMBG_DIR = Path("/tmp/gcm_refined")
DEFAULT_SAM3_DIR = Path("/tmp/gcm_sam3")
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "sam3_whole_cube_silhouette_bakeoff_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "SAM3_WHOLE_CUBE_SILHOUETTE_BAKEOFF_V0_REPORT.md"
THRESHOLDS_PX = (30, 50, 75, 100)
MEANINGFUL_DELTA_PX = 5.0
LARGE_REGRESSION_PX = 30.0

Point = Tuple[float, float]


def generate_silhouette_bakeoff(
    *,
    feedback_path: Path = DEFAULT_FEEDBACK,
    rembg_dir: Path = DEFAULT_REMBG_DIR,
    sam3_dir: Path = DEFAULT_SAM3_DIR,
) -> Dict[str, Any]:
    feedback = _read_json(feedback_path)
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for key, feedback_row in sorted(feedback.items()):
        true_vertex = feedback_row.get("true_vertex")
        if not true_vertex:
            skipped.append({"key": key, "reason": "missing_true_vertex"})
            continue
        try:
            set_id, side = key.split("_", 1)
        except ValueError:
            skipped.append({"key": key, "reason": "bad_feedback_key"})
            continue

        rembg_path = rembg_dir / f"set_{set_id}_{side}_data.json"
        sam3_path = sam3_dir / f"set_{set_id}_{side}_data.json"
        if not rembg_path.exists() or not sam3_path.exists():
            skipped.append({
                "key": key,
                "reason": "missing_fit_output",
                "rembgPath": str(rembg_path),
                "sam3Path": str(sam3_path),
            })
            continue

        rembg_data = _read_json(rembg_path)
        sam3_data = _read_json(sam3_path)
        rembg_vertex = _point_or_none(rembg_data.get("cube_center_screen"))
        sam3_vertex = _point_or_none(sam3_data.get("cube_center_screen"))
        if rembg_vertex is None or sam3_vertex is None:
            skipped.append({
                "key": key,
                "reason": "missing_fitted_vertex",
                "rembgPath": str(rembg_path),
                "sam3Path": str(sam3_path),
            })
            continue

        truth = _point_or_none(true_vertex)
        if truth is None:
            skipped.append({"key": key, "reason": "invalid_true_vertex"})
            continue

        rembg_error = _distance(rembg_vertex, truth)
        sam3_error = _distance(sam3_vertex, truth)
        delta = sam3_error - rembg_error
        rows.append({
            "key": key,
            "setId": set_id,
            "side": side,
            "imagePath": sam3_data.get("imagePath") or rembg_data.get("imagePath"),
            "imageSize": sam3_data.get("imageSize") or rembg_data.get("imageSize"),
            "trueVertex": _round_point(truth),
            "rembgVertex": _round_point(rembg_vertex),
            "sam3Vertex": _round_point(sam3_vertex),
            "rembgErrorPx": round(rembg_error, 2),
            "sam3ErrorPx": round(sam3_error, 2),
            "deltaPx": round(delta, 2),
            "winner": _winner(delta),
            "rembgFit": _fit_summary(rembg_data),
            "sam3Fit": _fit_summary(sam3_data),
        })

    rows.sort(key=lambda row: float(row["deltaPx"]))
    return {
        "schemaVersion": 1,
        "probe": "sam3_whole_cube_silhouette_bakeoff_v0",
        "description": (
            "Diagnostics/data-only comparison of the same refined global-model "
            "fit when driven by rembg silhouettes versus SAM3 whole-cube masks."
        ),
        "source": {
            "feedbackPath": str(feedback_path),
            "rembgFitDir": str(rembg_dir),
            "sam3FitDir": str(sam3_dir),
            "note": (
                "The committed fixture preserves this local bakeoff result; "
                "the source fit directories are diagnostic artifacts, not "
                "runtime dependencies."
            ),
        },
        "config": {
            "thresholdsPx": list(THRESHOLDS_PX),
            "meaningfulDeltaPx": MEANINGFUL_DELTA_PX,
            "largeRegressionPx": LARGE_REGRESSION_PX,
        },
        "summary": _summarize(rows, skipped),
        "rows": rows,
        "skipped": skipped,
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    thresholds = summary["thresholdCounts"]
    lines = [
        "# SAM3 Whole-Cube Silhouette Bakeoff V0",
        "",
        "Diagnostics/data-only artifact. This does not alter recognition behavior.",
        "",
        "This report compares a refined global cube model when the silhouette source is `rembg` versus when it is a cached SAM3 `whole_cube` mask. The score is the fitted visible trihedral vertex distance to the human label.",
        "",
        "## Summary",
        "",
        f"- Compared rows: {summary['rowCount']}",
        f"- Skipped rows: {summary['skippedCount']}",
        f"- rembg mean / median error: {summary['rembgMeanErrorPx']:.1f} / {summary['rembgMedianErrorPx']:.1f} px",
        f"- SAM3 mean / median error: {summary['sam3MeanErrorPx']:.1f} / {summary['sam3MedianErrorPx']:.1f} px",
        f"- Mean delta, SAM3 minus rembg: {summary['meanDeltaPx']:+.1f} px",
        f"- SAM3 better by >5 px: {summary['sam3BetterCount']}",
        f"- rembg better by >5 px: {summary['rembgBetterCount']}",
        f"- Within 5 px tie: {summary['tieCount']}",
        f"- SAM3 large regressions (>30 px worse): {summary['sam3LargeRegressionCount']}",
        "",
        "## Thresholds",
        "",
        "| Threshold | rembg rows below | SAM3 rows below |",
        "|---:|---:|---:|",
    ]
    for threshold in THRESHOLDS_PX:
        row = thresholds[str(threshold)]
        lines.append(
            f"| <{threshold} px | {row['rembgBelow']} | {row['sam3Below']} |"
        )

    lines.extend([
        "",
        "## Rows",
        "",
        "| Row | rembg err | SAM3 err | Delta | Winner | rembg refine | SAM3 refine |",
        "|---|---:|---:|---:|---|---|---|",
    ])
    for row in document["rows"]:
        lines.append(
            f"| `{row['key']}` | {row['rembgErrorPx']:.0f} | {row['sam3ErrorPx']:.0f} | "
            f"{row['deltaPx']:+.0f} | `{row['winner']}` | "
            f"`{row['rembgFit'].get('refinement', '')}` | `{row['sam3Fit'].get('refinement', '')}` |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- SAM3 whole-cube masks are materially more promising than SAM3 face text/box prompts: they improve mean and median vertex error in this paired refined-model bakeoff.",
        "- They are still not safe as a sole silhouette source. The regression tail is real, including several rows where SAM3 is more than 30 px worse than rembg.",
        "- Treat SAM3 whole-cube masks as an alternate geometry hypothesis or cross-check, not production wiring.",
        "- The next path should be geometry-first face splitting from a trusted whole-cube silhouette and vertex/axis model, because SAM3 is useful for object isolation but not for semantic face separation.",
        "",
    ])
    return "\n".join(lines)


def _summarize(rows: Sequence[Dict[str, Any]], skipped: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rembg_errors = [float(row["rembgErrorPx"]) for row in rows]
    sam3_errors = [float(row["sam3ErrorPx"]) for row in rows]
    deltas = [float(row["deltaPx"]) for row in rows]
    thresholds: Dict[str, Dict[str, int]] = {}
    for threshold in THRESHOLDS_PX:
        thresholds[str(threshold)] = {
            "rembgBelow": sum(1 for error in rembg_errors if error < threshold),
            "sam3Below": sum(1 for error in sam3_errors if error < threshold),
        }
    return {
        "rowCount": len(rows),
        "skippedCount": len(skipped),
        "rembgMeanErrorPx": _mean(rembg_errors),
        "sam3MeanErrorPx": _mean(sam3_errors),
        "meanDeltaPx": _mean(deltas),
        "rembgMedianErrorPx": _median(rembg_errors),
        "sam3MedianErrorPx": _median(sam3_errors),
        "rembgP95ErrorPx": _percentile(rembg_errors, 95),
        "sam3P95ErrorPx": _percentile(sam3_errors, 95),
        "rembgMaxErrorPx": max(rembg_errors) if rembg_errors else None,
        "sam3MaxErrorPx": max(sam3_errors) if sam3_errors else None,
        "sam3BetterCount": sum(1 for delta in deltas if delta < -MEANINGFUL_DELTA_PX),
        "rembgBetterCount": sum(1 for delta in deltas if delta > MEANINGFUL_DELTA_PX),
        "tieCount": sum(1 for delta in deltas if abs(delta) <= MEANINGFUL_DELTA_PX),
        "sam3LargeRegressionCount": sum(1 for delta in deltas if delta > LARGE_REGRESSION_PX),
        "thresholdCounts": thresholds,
    }


def _fit_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    debug = data.get("debug") or {}
    return {
        "fitQuality": data.get("fit_quality"),
        "fitResidualRmsPx": debug.get("fit_residual_rms_px"),
        "refinement": debug.get("refinement"),
        "cubeCenterSource": debug.get("cube_center_source"),
        "bezelVsFitCubeCenterOffsetPx": debug.get("bezel_vs_fit_cube_center_offset_px"),
        "hexagonCentroidVsBezelVertexOffsetPx": debug.get("hexagon_centroid_vs_bezel_vertex_offset_px"),
    }


def _winner(delta: float) -> str:
    if delta < -MEANINGFUL_DELTA_PX:
        return "sam3"
    if delta > MEANINGFUL_DELTA_PX:
        return "rembg"
    return "tie"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _point_or_none(value: Any) -> Optional[Point]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def _distance(left: Point, right: Point) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _round_point(point: Point) -> List[float]:
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(float(statistics.fmean(values)), 4) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    return round(float(statistics.median(values)), 4) if values else None


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    rank = (len(ordered) - 1) * percentile / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return round(ordered[int(rank)], 4)
    fraction = rank - low
    return round(ordered[low] * (1.0 - fraction) + ordered[high] * fraction, 4)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--rembg-dir", type=Path, default=DEFAULT_REMBG_DIR)
    parser.add_argument("--sam3-dir", type=Path, default=DEFAULT_SAM3_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    document = generate_silhouette_bakeoff(
        feedback_path=args.feedback,
        rembg_dir=args.rembg_dir,
        sam3_dir=args.sam3_dir,
    )
    _write_json(args.summary, document)
    _write_text(args.report, render_report(document))
    print(f"wrote {args.summary}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
