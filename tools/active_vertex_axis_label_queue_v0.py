#!/usr/bin/env python3
"""Build the next active-learning queue for vertex + axis labels.

Diagnostics/data-only. This does not alter recognizer behavior.

The KNN vertex localizer V0 showed that the local candidate grid can reach the
human visible-trihedral vertex on every currently labeled row, but ranking and
confidence need more supervision. This tool turns the manifest-wide global
cube model pool into a focused label queue for rows that do not yet have full
human vertex+axis labels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools.vertex_axis_feedback import (
    ALLOWED_STATUSES,
    DEFAULT_FEEDBACK,
    evaluate_feedback,
    render_report as render_feedback_report,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_SUMMARY = ROOT / "tests" / "fixtures" / "active_vertex_global_model_v01_summary.json"
DEFAULT_ACTIVE_FEEDBACK = ROOT / "tests" / "fixtures" / "vertex_axis_active_learning_feedback_v0.json"
DEFAULT_REPORT = ROOT / "tools" / "VERTEX_AXIS_ACTIVE_LEARNING_QUEUE_V0_REPORT.md"
DEFAULT_LABEL_REPORT = ROOT / "tools" / "VERTEX_AXIS_ACTIVE_LEARNING_FEEDBACK_V0_REPORT.md"

Point = Tuple[float, float]


def build_active_learning_feedback(
    *,
    model_summary_path: Path = DEFAULT_MODEL_SUMMARY,
    existing_feedback_path: Path = DEFAULT_FEEDBACK,
) -> Dict[str, Any]:
    model_summary = _read_json(model_summary_path)
    existing_feedback = _read_json(existing_feedback_path)
    existing_rows = {str(row.get("key")): row for row in existing_feedback.get("rows", [])}
    existing_labeled_keys = {
        key
        for key, row in existing_rows.items()
        if row.get("status") == "labeled"
        and _point_or_none(row.get("humanVertexPoint")) is not None
        and sum(1 for point in row.get("humanAxisEndpoints", []) if _point_or_none(point) is not None) == 3
    }
    rows: List[Dict[str, Any]] = []
    for model_row in model_summary.get("rows", []):
        key = _row_key(model_row)
        if not key or key in existing_labeled_keys:
            continue
        current_model = _current_model_from_global_row(model_row)
        if current_model is None:
            continue
        reasons = _active_reasons(model_row, key, existing_rows)
        rows.append(
            {
                "key": key,
                "setId": str(model_row.get("setId")),
                "side": str(model_row.get("side")),
                "imagePath": model_row.get("imagePath"),
                "status": "unlabeled",
                "humanVertexPoint": None,
                "humanAxisEndpoints": [None, None, None],
                "axisLabelQuality": None,
                "notes": "",
                "activeLearning": {
                    "priority": _priority_bucket(model_row, reasons),
                    "score": _active_score(model_row, reasons),
                    "reasons": reasons,
                    "sourceModelStatus": model_row.get("status"),
                    "diagnosticDisposition": model_row.get("diagnosticDisposition"),
                    "evaluationTier": model_row.get("evaluationTier"),
                },
                "currentModel": current_model,
                "target": (
                    "Mark the true visible trihedral vertex and one endpoint on "
                    "each of the three outgoing cube-edge rays. Endpoint order "
                    "does not matter; the scorer uses best ray assignment."
                ),
            }
        )
    rows.sort(key=_sort_key)
    return {
        "schemaVersion": 1,
        "artifact": "vertex_axis_active_learning_feedback_v0",
        "description": (
            "Active-learning queue for additional visible-trihedral labels. "
            "Rows are unlabeled manifest images not already present as full "
            "human vertex+axis labels in vertex_axis_human_feedback_v0."
        ),
        "allowedStatuses": ALLOWED_STATUSES,
        "sourceModelSummary": str(model_summary_path),
        "sourceExistingFeedback": str(existing_feedback_path),
        "labelerCommand": (
            ".venv/bin/python tools/vertex_axis_label_server.py "
            "--feedback tests/fixtures/vertex_axis_active_learning_feedback_v0.json "
            "--report tools/VERTEX_AXIS_ACTIVE_LEARNING_FEEDBACK_V0_REPORT.md "
            "--port 8778"
        ),
        "rows": rows,
    }


def render_queue_report(feedback: Dict[str, Any], existing_feedback: Dict[str, Any]) -> str:
    evaluation = evaluate_feedback(feedback)
    summary = evaluation["summary"]
    priorities: Dict[str, int] = {}
    reasons: Dict[str, int] = {}
    tiers: Dict[str, int] = {}
    statuses: Dict[str, int] = {}
    for row in feedback.get("rows", []):
        active = row.get("activeLearning") or {}
        priorities[str(active.get("priority"))] = priorities.get(str(active.get("priority")), 0) + 1
        tiers[str(active.get("evaluationTier"))] = tiers.get(str(active.get("evaluationTier")), 0) + 1
        statuses[str(active.get("sourceModelStatus"))] = statuses.get(str(active.get("sourceModelStatus")), 0) + 1
        for reason in active.get("reasons", []):
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    existing_eval = evaluate_feedback(existing_feedback)
    lines = [
        "# Vertex + Axis Active-Learning Queue V0",
        "",
        "Diagnostics/data-only artifact. This does not alter recognizer behavior.",
        "",
        "This queue selects additional manifest image rows for human visible-trihedral labeling after the KNN V0 result: candidate generation is strong, but ranking/confidence needs more supervision.",
        "",
        "## Summary",
        "",
        f"- Queue rows: {summary['rowCount']}",
        f"- Already completed canonical trihedral labels: {existing_eval['summary']['trihedralLabeledRowCount']}",
        f"- Queue rows with current model attached: {summary['currentModelRows']}",
        f"- Queue rows already labeled: {summary['trihedralLabeledRowCount']}",
        f"- Priority buckets: {_format_counts(priorities)}",
        f"- Evaluation tiers: {_format_counts(tiers)}",
        f"- Source model statuses: {_format_counts(statuses)}",
        f"- Active reasons: {_format_counts(reasons)}",
        "",
        "## Labeling Command",
        "",
        "```bash",
        feedback.get("labelerCommand", ""),
        "```",
        "",
        "## Labeling Instructions",
        "",
        "- Start with `tier1_easy_unlabeled` rows. These are the cleanest way to make the localizer boringly consistent before hard backgrounds.",
        "- For each row, click the visible trihedral vertex first, then one endpoint on each outgoing cube-edge ray. Axis order does not matter.",
        "- Use `ambiguous` or `not_visible` if the vertex/rays cannot be marked honestly.",
        "- Keep notes short when the model overlay is misleading, the cube is occluded, or the background should be treated as a retake candidate.",
        "",
        "## Rows",
        "",
        "| Row | Priority | Tier | Source status | Score | Reasons | Model score | IoU | Cell inside |",
        "|---|---|---|---|---:|---|---:|---:|---:|",
    ]
    for row in feedback.get("rows", []):
        active = row.get("activeLearning") or {}
        model = row.get("currentModel") or {}
        debug = model.get("debug") or {}
        lines.append(
            f"| `{row.get('key')}` | `{active.get('priority')}` | "
            f"`{active.get('evaluationTier')}` | `{active.get('sourceModelStatus')}` | "
            f"{float(active.get('score', 0.0)):.2f} | "
            f"{', '.join(active.get('reasons', []))} | "
            f"{_fmt(model.get('fitQuality'))} | "
            f"{_fmt(debug.get('silhouetteIoU'))} | "
            f"{_fmt(debug.get('cellInsideRatio'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is step #1 for the next learned-localizer loop: expand the human vertex+axis labels without relabeling the completed 28-row fixture.",
            "- Step #2 should train/evaluate the richer localizer only after a meaningful portion of this queue is labeled.",
            "- Until then, KNN V0 remains the best diagnostics baseline and should not be production-wired.",
            "",
        ]
    )
    return "\n".join(lines)


def write_active_learning_outputs(
    *,
    feedback_path: Path = DEFAULT_ACTIVE_FEEDBACK,
    report_path: Path = DEFAULT_REPORT,
    label_report_path: Path = DEFAULT_LABEL_REPORT,
    model_summary_path: Path = DEFAULT_MODEL_SUMMARY,
    existing_feedback_path: Path = DEFAULT_FEEDBACK,
) -> Dict[str, Any]:
    feedback = build_active_learning_feedback(
        model_summary_path=model_summary_path,
        existing_feedback_path=existing_feedback_path,
    )
    existing_feedback = _read_json(existing_feedback_path)
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    feedback_path.write_text(json.dumps(feedback, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_queue_report(feedback, existing_feedback), encoding="utf-8")
    label_report_path.parent.mkdir(parents=True, exist_ok=True)
    label_report_path.write_text(render_active_feedback_report(feedback), encoding="utf-8")
    return feedback


def render_active_feedback_report(feedback: Dict[str, Any]) -> str:
    report = render_feedback_report(feedback, evaluate_feedback(feedback))
    default_command = ".venv/bin/python tools/vertex_axis_label_server.py --port 8778"
    active_command = str(feedback.get("labelerCommand") or default_command)
    return (
        report.replace(
            "# Vertex + Axis Human Feedback V0",
            "# Vertex + Axis Active-Learning Feedback V0",
            1,
        )
        .replace(
            "This fixture upgrades the vertex-only labels into a scaffold for complete visible-trihedral labels: vertex point plus three outgoing cube-edge rays.",
            "This fixture is the active-learning queue for additional visible-trihedral labels: vertex point plus three outgoing cube-edge rays.",
            1,
        )
        .replace(default_command, active_command, 1)
        .replace(
            "- The scaffold is ready for human axis labels, but the committed durable labels are still vertex-only.",
            "- The active-learning scaffold is ready for human vertex+axis labels; committed rows remain unlabeled until the labeler writes them.",
            1,
        )
    )


def _current_model_from_global_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    model = row.get("model") or {}
    vertex = _point_or_none(model.get("cubeCenter"))
    axes = model.get("axes") or []
    if vertex is None or len(axes) != 3:
        return None
    axis_items = []
    for idx, axis in enumerate(axes):
        vector = _point_or_none(axis)
        if vector is None:
            return None
        axis_items.append(_axis_item(f"axis_{idx}", vector, vertex))
    components = model.get("scoreComponents") or {}
    fit_diagnostics = row.get("fitDiagnostics") or {}
    return {
        "status": "ok",
        "modelSource": "global_cube_model_v01_manifest_pool",
        "vertexPoint": _round_point(vertex),
        "axes": axis_items,
        "fitQuality": model.get("score"),
        "debug": {
            "fitVersion": fit_diagnostics.get("fitVersion") or "v0.1-center-refine",
            "sourceStatus": row.get("status"),
            "diagnosticDisposition": row.get("diagnosticDisposition"),
            "silhouetteIoU": components.get("silhouetteIoU"),
            "insideRatio": components.get("insideRatio"),
            "cellInsideRatio": components.get("cellInsideRatio"),
            "maskCoverage": components.get("maskCoverage"),
            "detectorSignalQuality": components.get("detectorSignalQuality"),
            "edgeLength": model.get("edgeLength"),
            "signChoice": model.get("signChoice"),
        },
    }


def _axis_item(name: str, vector: Point, vertex: Point) -> Dict[str, Any]:
    endpoint = (vertex[0] + vector[0], vertex[1] + vector[1])
    angle = math.degrees(math.atan2(vector[1], vector[0]))
    length = math.hypot(vector[0], vector[1])
    return {
        "name": name,
        "status": "ok",
        "vector": _round_point(vector),
        "endpoint": _round_point(endpoint),
        "angleDeg": round(angle, 2),
        "lengthPx": round(length, 2),
    }


def _active_reasons(row: Dict[str, Any], key: str, existing_rows: Dict[str, Dict[str, Any]]) -> List[str]:
    reasons: List[str] = []
    tier = row.get("evaluationTier")
    status = row.get("status")
    if tier == "easy_corpus":
        reasons.append("easy_corpus_unlabeled")
    elif tier == "corpus_stress":
        reasons.append("corpus_stress_unlabeled")
    else:
        reasons.append("hard_case_unlabeled")
    if status != "ok":
        reasons.append("weak_geometry_or_retake_boundary")
    components = ((row.get("model") or {}).get("scoreComponents") or {})
    if float(components.get("silhouetteIoU") or 0.0) < 0.70:
        reasons.append("low_silhouette_iou")
    if float(components.get("cellInsideRatio") or 0.0) < 0.95:
        reasons.append("low_cell_inside")
    paired_key = _paired_key(key)
    if paired_key in existing_rows:
        reasons.append("paired_with_existing_label")
    return reasons


def _priority_bucket(row: Dict[str, Any], reasons: Sequence[str]) -> str:
    if row.get("evaluationTier") == "easy_corpus" and row.get("status") == "ok":
        return "tier1_easy_unlabeled"
    if "paired_with_existing_label" in reasons and row.get("status") == "ok":
        return "tier2_pair_completion"
    if row.get("status") != "ok":
        return "tier4_retake_or_model_boundary"
    return "tier3_manifest_stress"


def _active_score(row: Dict[str, Any], reasons: Sequence[str]) -> float:
    score = 0.0
    tier = row.get("evaluationTier")
    if tier == "easy_corpus":
        score += 100.0
    elif tier == "corpus_stress":
        score += 60.0
    else:
        score += 40.0
    if "paired_with_existing_label" in reasons:
        score += 20.0
    if row.get("status") != "ok":
        score -= 25.0
    components = ((row.get("model") or {}).get("scoreComponents") or {})
    score += float(components.get("silhouetteIoU") or 0.0) * 10.0
    score += float(components.get("cellInsideRatio") or 0.0) * 5.0
    return round(score, 4)


def _sort_key(row: Dict[str, Any]) -> Tuple[int, float, str]:
    active = row.get("activeLearning") or {}
    order = {
        "tier1_easy_unlabeled": 0,
        "tier2_pair_completion": 1,
        "tier3_manifest_stress": 2,
        "tier4_retake_or_model_boundary": 3,
    }
    return (
        order.get(str(active.get("priority")), 9),
        -float(active.get("score") or 0.0),
        str(row.get("key")),
    )


def _row_key(row: Dict[str, Any]) -> Optional[str]:
    set_id = row.get("setId")
    side = row.get("side")
    if set_id is None or side is None:
        return None
    return f"{set_id}_{side}"


def _paired_key(key: str) -> str:
    if key.endswith("_A"):
        return f"{key[:-2]}_B"
    if key.endswith("_B"):
        return f"{key[:-2]}_A"
    return key


def _point_or_none(value: Any) -> Optional[Point]:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return float(value[0]), float(value[1])
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return float(value[0]), float(value[1])
    return None


def _round_point(point: Point) -> List[float]:
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _format_counts(counts: Dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"`{key}` {value}" for key, value in sorted(counts.items()))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-summary", type=Path, default=DEFAULT_MODEL_SUMMARY)
    parser.add_argument("--existing-feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--feedback-out", type=Path, default=DEFAULT_ACTIVE_FEEDBACK)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--label-report-out", type=Path, default=DEFAULT_LABEL_REPORT)
    args = parser.parse_args(argv)
    feedback = write_active_learning_outputs(
        feedback_path=args.feedback_out,
        report_path=args.report_out,
        label_report_path=args.label_report_out,
        model_summary_path=args.model_summary,
        existing_feedback_path=args.existing_feedback,
    )
    print(f"wrote {args.feedback_out}")
    print(f"wrote {args.report_out}")
    print(f"wrote {args.label_report_out}")
    print(f"queue rows: {len(feedback.get('rows', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
