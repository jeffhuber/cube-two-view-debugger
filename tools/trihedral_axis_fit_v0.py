#!/usr/bin/env python3
"""Score current trihedral model hypotheses against vertex+axis labels.

Diagnostics/data-only. This module does not alter recognizer behavior.

Once humans label the visible trihedral vertex and three outgoing cube-edge
rays, this scorer evaluates whether a model's vertex + projected axes match
that labeled trihedral. Axis endpoint order does not matter: the deployable
model supplies three axes, while the human labels supply three rays, and the
scorer reports the best assignment by angular error.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEEDBACK = ROOT / "tests" / "fixtures" / "vertex_axis_human_feedback_v0.json"
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "trihedral_axis_fit_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "TRIHEDRAL_AXIS_FIT_V0_REPORT.md"

STRICT_VERTEX_PX = 30.0
PLAUSIBLE_VERTEX_PX = 50.0
STRICT_MAX_AXIS_ANGLE_DEG = 8.0
PLAUSIBLE_MAX_AXIS_ANGLE_DEG = 15.0
AXIS_GOOD_FOR_TAXONOMY_DEG = PLAUSIBLE_MAX_AXIS_ANGLE_DEG
VERTEX_GOOD_FOR_TAXONOMY_PX = PLAUSIBLE_VERTEX_PX

Point = Tuple[float, float]


def generate_trihedral_axis_fit_summary(
    *,
    feedback_path: Path = DEFAULT_FEEDBACK,
) -> Dict[str, Any]:
    feedback = _read_json(feedback_path)
    rows = [evaluate_row(row) for row in feedback.get("rows", [])]
    return {
        "schemaVersion": 1,
        "probe": "trihedral_axis_fit_v0",
        "description": (
            "Diagnostics/data-only scorer for the visible trihedral: vertex "
            "point plus three outgoing cube-edge rays."
        ),
        "sourceFeedback": str(feedback_path),
        "config": {
            "strictVertexPx": STRICT_VERTEX_PX,
            "plausibleVertexPx": PLAUSIBLE_VERTEX_PX,
            "strictMaxAxisAngleDeg": STRICT_MAX_AXIS_ANGLE_DEG,
            "plausibleMaxAxisAngleDeg": PLAUSIBLE_MAX_AXIS_ANGLE_DEG,
            "taxonomyVertexGoodPx": VERTEX_GOOD_FOR_TAXONOMY_PX,
            "taxonomyAxisGoodDeg": AXIS_GOOD_FOR_TAXONOMY_DEG,
        },
        "summary": summarize_rows(rows),
        "rows": rows,
    }


def evaluate_row(row: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "key": row.get("key"),
        "setId": row.get("setId"),
        "side": row.get("side"),
        "labelStatus": row.get("status"),
    }
    human_vertex = _point_or_none(row.get("humanVertexPoint"))
    human_endpoints = [_point_or_none(point) for point in row.get("humanAxisEndpoints", [])]
    human_endpoints = [point for point in human_endpoints if point is not None]
    model = row.get("currentModel") or {}
    model_vertex = _point_or_none(model.get("vertexPoint"))
    model_axes = [
        axis
        for axis in model.get("axes", [])
        if axis.get("status") == "ok" and _point_or_none(axis.get("vector")) is not None
    ]
    if human_vertex is None:
        return {**base, "evaluationStatus": "missing_human_vertex"}
    if len(human_endpoints) != 3:
        return {
            **base,
            "evaluationStatus": "axis_labels_pending",
            "axisEndpointCount": len(human_endpoints),
            "hasCurrentModel": model_vertex is not None and len(model_axes) == 3,
            "vertexErrorPx": (
                round(_distance(human_vertex, model_vertex), 2)
                if model_vertex is not None
                else None
            ),
        }
    if model_vertex is None or len(model_axes) != 3:
        return {**base, "evaluationStatus": "missing_model_trihedral"}

    human_vectors = [(endpoint[0] - human_vertex[0], endpoint[1] - human_vertex[1]) for endpoint in human_endpoints]
    candidate_vectors = [_point_or_none(axis.get("vector")) for axis in model_axes]
    candidate_vectors = [vector for vector in candidate_vectors if vector is not None]
    assignment = _best_axis_assignment(candidate_vectors, human_vectors)
    vertex_error = _distance(human_vertex, model_vertex)
    max_axis_error = max(assignment["angleErrorsDeg"])
    mean_axis_error = statistics.mean(assignment["angleErrorsDeg"])
    status = _status(vertex_error, max_axis_error)
    failure_category = _failure_category(status, vertex_error, max_axis_error)
    return {
        **base,
        "evaluationStatus": "trihedral_labeled",
        "status": status,
        "failureCategory": failure_category,
        "vertexErrorPx": round(vertex_error, 2),
        "meanAxisAngleErrorDeg": round(mean_axis_error, 2),
        "maxAxisAngleErrorDeg": round(max_axis_error, 2),
        "axisAssignment": assignment,
        "modelFitQuality": model.get("fitQuality"),
    }


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    trihedral = [row for row in rows if row.get("evaluationStatus") == "trihedral_labeled"]
    pending = [row for row in rows if row.get("evaluationStatus") == "axis_labels_pending"]
    vertex_errors = [float(row["vertexErrorPx"]) for row in trihedral if row.get("vertexErrorPx") is not None]
    axis_errors = [
        float(row["maxAxisAngleErrorDeg"])
        for row in trihedral
        if row.get("maxAxisAngleErrorDeg") is not None
    ]
    failure_categories = _count_by_key(trihedral, "failureCategory")
    return {
        "rowCount": len(rows),
        "trihedralLabeledRowCount": len(trihedral),
        "axisLabelsPendingRowCount": len(pending),
        "missingHumanVertexRowCount": sum(1 for row in rows if row.get("evaluationStatus") == "missing_human_vertex"),
        "missingModelTrihedralRowCount": sum(1 for row in rows if row.get("evaluationStatus") == "missing_model_trihedral"),
        "strictReadyCount": sum(1 for row in trihedral if row.get("status") == "strict_ready"),
        "plausibleCount": sum(1 for row in trihedral if row.get("status") in {"strict_ready", "plausible"}),
        "blockedCount": sum(1 for row in trihedral if row.get("status") == "blocked"),
        "meanVertexErrorPx": _mean(vertex_errors),
        "medianVertexErrorPx": _median(vertex_errors),
        "meanMaxAxisAngleErrorDeg": _mean(axis_errors),
        "medianMaxAxisAngleErrorDeg": _median(axis_errors),
        "failureCategoryCounts": failure_categories,
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    lines = [
        "# Trihedral Axis Fit V0",
        "",
        "Diagnostics/data-only artifact. This does not alter recognition behavior.",
        "",
        "This scorer evaluates a full visible-trihedral fit: vertex position plus three outgoing cube-edge ray directions.",
        "",
        "## Summary",
        "",
        f"- Rows: {summary['rowCount']}",
        f"- Full trihedral labels: {summary['trihedralLabeledRowCount']}",
        f"- Axis labels pending: {summary['axisLabelsPendingRowCount']}",
        f"- Missing human vertex: {summary['missingHumanVertexRowCount']}",
        f"- Missing model trihedral: {summary['missingModelTrihedralRowCount']}",
        f"- Strict-ready: {summary['strictReadyCount']}",
        f"- Plausible: {summary['plausibleCount']}",
        f"- Blocked: {summary['blockedCount']}",
        f"- Mean vertex error: {_fmt(summary['meanVertexErrorPx'], 'px')}",
        f"- Median vertex error: {_fmt(summary['medianVertexErrorPx'], 'px')}",
        f"- Mean max-axis angle error: {_fmt(summary['meanMaxAxisAngleErrorDeg'], 'deg')}",
        f"- Median max-axis angle error: {_fmt(summary['medianMaxAxisAngleErrorDeg'], 'deg')}",
        "- Failure categories: " + _format_category_counts(summary.get("failureCategoryCounts", {})),
        "",
        "## Rows",
        "",
        "| Row | Status | Failure category | Vertex error | Max axis angle | Mean axis angle |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in document["rows"]:
        lines.append(
            f"| `{row.get('key')}` | `{row.get('evaluationStatus') if row.get('evaluationStatus') != 'trihedral_labeled' else row.get('status')}` | "
            f"`{row.get('failureCategory', 'n/a')}` | "
            f"{_fmt(row.get('vertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('maxAxisAngleErrorDeg'), 'deg')} | "
            f"{_fmt(row.get('meanAxisAngleErrorDeg'), 'deg')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if summary["trihedralLabeledRowCount"] == 0:
        lines.extend(
            [
                "- No row currently has all three human axis endpoints, so this PR cannot claim axis-fit quality yet.",
                "- This is intentional: previous conclusions were limited by vertex-only labels. The scorer is now ready for the next human labeling pass.",
            ]
        )
    else:
        lines.extend(
            [
                "- Rows are `strict_ready` only when both the vertex and all three axis directions are within the configured thresholds.",
                "- Axis assignment is order-invariant, so high errors point at geometry, not label ordering.",
                "- `vertex_localization_blocked` means axis directions are plausible but the visible trihedral vertex is too far from the human label.",
                "- `axis_correspondence_blocked` means the vertex is plausible but the outgoing axis family is wrong.",
                "- `both_blocked` means neither the vertex nor the axis family is currently usable.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _best_axis_assignment(candidate_vectors: Sequence[Point], human_vectors: Sequence[Point]) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    for perm in itertools.permutations(range(3)):
        errors = [
            _axis_angle_error_deg(candidate_vectors[candidate_index], human_vectors[human_index])
            for human_index, candidate_index in enumerate(perm)
        ]
        score = (max(errors), sum(errors))
        if best is None or score < best["score"]:
            best = {
                "candidatePermutation": list(perm),
                "angleErrorsDeg": [round(error, 2) for error in errors],
                "score": score,
            }
    assert best is not None
    return {
        "candidatePermutation": best["candidatePermutation"],
        "angleErrorsDeg": best["angleErrorsDeg"],
    }


def _axis_angle_error_deg(a: Point, b: Point) -> float:
    an = math.hypot(a[0], a[1])
    bn = math.hypot(b[0], b[1])
    if an <= 1e-6 or bn <= 1e-6:
        return 180.0
    dot = (a[0] * b[0] + a[1] * b[1]) / (an * bn)
    dot = max(-1.0, min(1.0, dot))
    return abs(math.degrees(math.acos(dot)))


def _status(vertex_error: float, max_axis_error: float) -> str:
    if vertex_error <= STRICT_VERTEX_PX and max_axis_error <= STRICT_MAX_AXIS_ANGLE_DEG:
        return "strict_ready"
    if vertex_error <= PLAUSIBLE_VERTEX_PX and max_axis_error <= PLAUSIBLE_MAX_AXIS_ANGLE_DEG:
        return "plausible"
    return "blocked"


def _failure_category(status: str, vertex_error: float, max_axis_error: float) -> str:
    if status == "strict_ready":
        return "strict_ready"
    if status == "plausible":
        return "plausible"
    vertex_good = vertex_error <= VERTEX_GOOD_FOR_TAXONOMY_PX
    axis_good = max_axis_error <= AXIS_GOOD_FOR_TAXONOMY_DEG
    if axis_good and not vertex_good:
        return "vertex_localization_blocked"
    if vertex_good and not axis_good:
        return "axis_correspondence_blocked"
    return "both_blocked"


def _point_or_none(value: Any) -> Optional[Point]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _distance(a: Point, b: Point) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return round(values[mid], 4)
    return round((values[mid - 1] + values[mid]) / 2.0, 4)


def _count_by_key(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _format_category_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return "n/a"
    return ", ".join(f"`{key}` {value}" for key, value in sorted(counts.items()))


def _fmt(value: Any, unit: str) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f} {unit}"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    document = generate_trihedral_axis_fit_summary(feedback_path=args.feedback)
    _write_json(args.summary_out, document)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(document), encoding="utf-8")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.report_out}")
    print(
        f"trihedral-labeled rows: {document['summary']['trihedralLabeledRowCount']} / "
        f"{document['summary']['rowCount']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
