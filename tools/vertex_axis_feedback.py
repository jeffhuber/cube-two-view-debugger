#!/usr/bin/env python3
"""Human-feedback scaffold for visible trihedral vertex + axis labels.

Diagnostics/data-only. This module does not alter recognizer behavior.

The global cube path now needs labels for the complete visible trihedral:
one front vertex plus three outgoing cube-edge rays. Existing durable labels
only mark the vertex point, so this tool preserves those vertex labels and
adds an explicit scaffold for three human axis endpoints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUND_TRUTH = ROOT / "tests" / "fixtures" / "gcm_vertex_ground_truth.json"
DEFAULT_MODEL_CANDIDATES = ROOT / "tests" / "fixtures" / "trihedral_model_candidates_v0.json"
DEFAULT_FEEDBACK = ROOT / "tests" / "fixtures" / "vertex_axis_human_feedback_v0.json"
DEFAULT_REPORT = ROOT / "tools" / "VERTEX_AXIS_HUMAN_FEEDBACK_V0_REPORT.md"

Point = Tuple[float, float]

ALLOWED_STATUSES = [
    "unlabeled",
    "vertex_labeled_axes_unlabeled",
    "labeled",
    "ambiguous",
    "not_visible",
    "judgment_only",
]


def build_model_candidate_fixture(
    *,
    ground_truth: Dict[str, Any],
    model_data_dir: Path,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for key in sorted(ground_truth):
        model_path = model_data_dir / f"set_{key}_data.json"
        if not model_path.exists():
            rows.append({"key": key, "status": "missing_model_data", "modelPath": str(model_path)})
            continue
        data = _read_json(model_path)
        vertex = _point_or_none(data.get("cube_center_screen"))
        axes = [
            _axis_item("axis_x", data.get("axis_x_2d"), vertex),
            _axis_item("axis_y", data.get("axis_y_2d"), vertex),
            _axis_item("axis_z", data.get("axis_z_2d"), vertex),
        ]
        rows.append(
            {
                "key": key,
                "status": "ok" if vertex and all(item["status"] == "ok" for item in axes) else "incomplete",
                "modelPath": str(model_path),
                "modelSource": "global_cube_model_rembg_refined",
                "vertexPoint": _round_point(vertex) if vertex else None,
                "axes": axes,
                "fitQuality": data.get("fit_quality"),
                "fitLoss": data.get("fit_loss"),
                "debug": {
                    "cubeCenterSource": (data.get("debug") or {}).get("cube_center_source"),
                    "fitResidualRmsPx": (data.get("debug") or {}).get("fit_residual_rms_px"),
                    "refinement": (data.get("debug") or {}).get("refinement"),
                },
            }
        )
    return {
        "schemaVersion": 1,
        "artifact": "trihedral_model_candidates_v0",
        "description": (
            "Diagnostics/data-only snapshot of current global-cube-model "
            "trihedral vertex and axis hypotheses used by the axis labeler."
        ),
        "sourceGroundTruth": str(DEFAULT_GROUND_TRUTH),
        "sourceModelDataDir": str(model_data_dir),
        "rows": rows,
    }


def build_feedback_scaffold(
    *,
    ground_truth: Dict[str, Any],
    model_candidates: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    manifests = _load_manifests()
    candidates_by_key = {
        str(row.get("key")): row
        for row in (model_candidates or {}).get("rows", [])
    }
    rows: List[Dict[str, Any]] = []
    for key, item in sorted(ground_truth.items()):
        set_id, side = _split_key(key)
        true_vertex = _point_or_none(item.get("true_vertex"))
        current_vertex = _point_or_none(item.get("current_vertex"))
        candidate = candidates_by_key.get(key)
        if true_vertex is not None:
            status = "vertex_labeled_axes_unlabeled"
        elif bool(item.get("center_correct")):
            status = "judgment_only"
        else:
            status = "unlabeled"
        rows.append(
            {
                "key": key,
                "setId": set_id,
                "side": side,
                "imagePath": _resolve_image_path(manifests, set_id, side),
                "status": status,
                "humanVertexPoint": _round_point(true_vertex) if true_vertex else None,
                "humanAxisEndpoints": [None, None, None],
                "axisLabelQuality": None,
                "notes": item.get("notes") or "",
                "sourceJudgment": {
                    "centerCorrect": bool(item.get("center_correct")),
                    "currentVertex": _round_point(current_vertex) if current_vertex else None,
                    "currentVertexErrorPx": item.get("error_px"),
                },
                "currentModel": _candidate_for_feedback(candidate),
                "target": (
                    "Mark the true visible trihedral vertex and one endpoint on "
                    "each of the three outgoing cube-edge rays. Endpoint order "
                    "does not matter; the scorer uses best ray assignment."
                ),
            }
        )

    return {
        "schemaVersion": 1,
        "artifact": "vertex_axis_human_feedback_v0",
        "description": (
            "Human-label scaffold for the complete visible trihedral: vertex "
            "point plus three outgoing cube-edge rays. Existing vertex labels "
            "are preserved; axis endpoints remain unlabeled until a human "
            "marks them in the labeler."
        ),
        "allowedStatuses": ALLOWED_STATUSES,
        "sourceGroundTruth": str(DEFAULT_GROUND_TRUTH),
        "sourceModelCandidates": str(DEFAULT_MODEL_CANDIDATES),
        "rows": rows,
    }


def evaluate_feedback(feedback: Dict[str, Any]) -> Dict[str, Any]:
    rows = [evaluate_feedback_row(row) for row in feedback.get("rows", [])]
    vertex_labeled = [row for row in rows if row["hasHumanVertex"]]
    axis_labeled = [row for row in rows if row["axisEndpointCount"] == 3]
    full = [row for row in rows if row["evaluationStatus"] == "trihedral_labeled"]
    vertex_errors = [
        float(row["currentModelVertexErrorPx"])
        for row in rows
        if row.get("currentModelVertexErrorPx") is not None
    ]
    return {
        "schemaVersion": 1,
        "probe": "vertex_axis_human_feedback_v0",
        "summary": {
            "rowCount": len(rows),
            "vertexLabeledRowCount": len(vertex_labeled),
            "axisLabeledRowCount": len(axis_labeled),
            "trihedralLabeledRowCount": len(full),
            "currentModelRows": sum(1 for row in rows if row.get("hasCurrentModel")),
            "meanCurrentModelVertexErrorPx": _mean(vertex_errors),
            "medianCurrentModelVertexErrorPx": _median(vertex_errors),
        },
        "rows": rows,
    }


def evaluate_feedback_row(row: Dict[str, Any]) -> Dict[str, Any]:
    human_vertex = _point_or_none(row.get("humanVertexPoint"))
    endpoints = [_point_or_none(point) for point in row.get("humanAxisEndpoints", [])]
    endpoint_count = sum(1 for point in endpoints if point is not None)
    model = row.get("currentModel") or {}
    model_vertex = _point_or_none(model.get("vertexPoint"))
    model_error = None
    if human_vertex is not None and model_vertex is not None:
        model_error = round(_distance(human_vertex, model_vertex), 2)
    if human_vertex is not None and endpoint_count == 3:
        status = "trihedral_labeled"
    elif human_vertex is not None:
        status = "vertex_labeled_axes_pending"
    elif row.get("status") == "judgment_only":
        status = "judgment_only"
    else:
        status = str(row.get("status") or "unlabeled")
    return {
        "key": row.get("key"),
        "setId": row.get("setId"),
        "side": row.get("side"),
        "labelStatus": row.get("status"),
        "evaluationStatus": status,
        "hasHumanVertex": human_vertex is not None,
        "axisEndpointCount": endpoint_count,
        "hasCurrentModel": bool(model.get("status") == "ok"),
        "currentModelVertexErrorPx": model_error,
    }


def render_report(feedback: Dict[str, Any], evaluation: Dict[str, Any]) -> str:
    summary = evaluation["summary"]
    lines = [
        "# Vertex + Axis Human Feedback V0",
        "",
        "Diagnostics/data-only artifact. This does not alter recognizer behavior.",
        "",
        "This fixture upgrades the vertex-only labels into a scaffold for complete visible-trihedral labels: vertex point plus three outgoing cube-edge rays.",
        "",
        "## Summary",
        "",
        f"- Rows: {summary['rowCount']}",
        f"- Rows with human vertex labels: {summary['vertexLabeledRowCount']}",
        f"- Rows with all three axis endpoints: {summary['axisLabeledRowCount']}",
        f"- Full trihedral labels: {summary['trihedralLabeledRowCount']}",
        f"- Rows with current model axes attached: {summary['currentModelRows']}",
        f"- Mean current-model vertex error: {_fmt_px(summary['meanCurrentModelVertexErrorPx'])}",
        f"- Median current-model vertex error: {_fmt_px(summary['medianCurrentModelVertexErrorPx'])}",
        "",
        "## Labeling Target",
        "",
        "For each row, mark the visible trihedral vertex and one endpoint on each of the three outgoing cube-edge rays. Axis endpoint order does not matter; the scorer will use best assignment.",
        "",
        "Run the labeler with:",
        "",
        "```bash",
        ".venv/bin/python tools/vertex_axis_label_server.py --port 8778",
        "```",
        "",
        "## Rows",
        "",
        "| Row | Status | Vertex label | Axis endpoints | Current model vertex error | Notes |",
        "|---|---|---|---:|---:|---|",
    ]
    row_by_key = {row.get("key"): row for row in feedback.get("rows", [])}
    for row in evaluation["rows"]:
        source = row_by_key.get(row.get("key"), {})
        lines.append(
            f"| `{row.get('key')}` | `{row.get('evaluationStatus')}` | "
            f"{'yes' if row['hasHumanVertex'] else 'no'} | "
            f"{row['axisEndpointCount']} | "
            f"{_fmt_px(row.get('currentModelVertexErrorPx'))} | "
            f"{source.get('notes') or ''} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The scaffold is ready for human axis labels, but the committed durable labels are still vertex-only.",
            "- The next scorer can evaluate current model axes as soon as full trihedral labels exist; until then, axis-quality conclusions should stay pending.",
            "",
        ]
    )
    return "\n".join(lines)


def update_feedback_row(
    feedback: Dict[str, Any],
    *,
    key: str,
    status: str,
    human_vertex_point: Optional[Point],
    human_axis_endpoints: Sequence[Optional[Point]],
    axis_label_quality: Optional[str],
    notes: str,
) -> None:
    if status not in set(feedback.get("allowedStatuses", ALLOWED_STATUSES)):
        raise ValueError(f"unsupported status: {status}")
    if status == "labeled" and (human_vertex_point is None or len([p for p in human_axis_endpoints if p]) != 3):
        raise ValueError("labeled rows require a vertex point and 3 axis endpoints")
    for row in feedback.get("rows", []):
        if str(row.get("key")) == str(key):
            row["status"] = status
            row["humanVertexPoint"] = _round_point(human_vertex_point) if human_vertex_point else None
            padded = list(human_axis_endpoints[:3]) + [None, None, None]
            row["humanAxisEndpoints"] = [
                _round_point(point) if point is not None else None
                for point in padded[:3]
            ]
            row["axisLabelQuality"] = axis_label_quality
            row["notes"] = notes
            return
    raise KeyError(f"feedback row not found: {key}")


def _candidate_for_feedback(candidate: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidate:
        return None
    return {
        "status": candidate.get("status"),
        "modelSource": candidate.get("modelSource"),
        "vertexPoint": candidate.get("vertexPoint"),
        "axes": candidate.get("axes") or [],
        "fitQuality": candidate.get("fitQuality"),
        "debug": candidate.get("debug") or {},
    }


def _axis_item(name: str, vector_value: Any, vertex: Optional[Point]) -> Dict[str, Any]:
    vector = _point_or_none(vector_value)
    if vector is None or vertex is None:
        return {"name": name, "status": "missing"}
    length = math.hypot(vector[0], vector[1])
    endpoint = (vertex[0] + vector[0], vertex[1] + vector[1])
    return {
        "name": name,
        "status": "ok",
        "vector": _round_point(vector),
        "endpoint": _round_point(endpoint),
        "lengthPx": round(length, 2),
        "angleDeg": round(math.degrees(math.atan2(vector[1], vector[0])), 2),
    }


def _load_manifests() -> List[Dict[str, Any]]:
    manifests = []
    for filename in ("hard_case_manifest.json", "corpus_manifest.json"):
        path = ROOT / "tests" / "fixtures" / filename
        if path.exists():
            manifests.append(_read_json(path))
    return manifests


def _resolve_image_path(manifests: Sequence[Dict[str, Any]], set_id: str, side: str) -> Optional[str]:
    side_key = "imageAPath" if side == "A" else "imageBPath"
    for manifest in manifests:
        for pair in manifest.get("pairs", []):
            if str(pair.get("setId")) == str(set_id):
                return pair.get(side_key)
    return None


def _split_key(key: str) -> Tuple[str, str]:
    if "_" not in key:
        return key, ""
    set_id, side = key.split("_", 1)
    return set_id, side


def _point_or_none(value: Any) -> Optional[Point]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _round_point(point: Optional[Point]) -> Optional[List[float]]:
    if point is None:
        return None
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


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


def _fmt_px(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.1f} px"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--model-candidates", type=Path, default=DEFAULT_MODEL_CANDIDATES)
    parser.add_argument("--model-data-dir", type=Path, default=None)
    parser.add_argument("--feedback-out", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    ground_truth = _read_json(args.ground_truth)
    if args.model_data_dir is not None:
        candidates = build_model_candidate_fixture(
            ground_truth=ground_truth,
            model_data_dir=args.model_data_dir,
        )
        _write_json(args.model_candidates, candidates)
    elif args.model_candidates.exists():
        candidates = _read_json(args.model_candidates)
    else:
        candidates = {"rows": []}

    feedback = build_feedback_scaffold(
        ground_truth=ground_truth,
        model_candidates=candidates,
    )
    evaluation = evaluate_feedback(feedback)
    _write_json(args.feedback_out, feedback)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(feedback, evaluation), encoding="utf-8")
    print(f"wrote {args.feedback_out}")
    print(f"wrote {args.report_out}")
    if args.model_data_dir is not None:
        print(f"wrote {args.model_candidates}")
    print(
        f"axis-labeled rows: {evaluation['summary']['axisLabeledRowCount']} / "
        f"{evaluation['summary']['rowCount']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
