#!/usr/bin/env python3
"""Human-feedback scaffold for vertex-point candidate diagnostics.

Diagnostics-only. This module does not alter recognizer behavior.

The visible trihedral corner ("vertex point") is the load-bearing geometry
primitive for the first-principles recognizer path. The candidate generator can
rank plausible points, but this feedback artifact records whether those points
match a human-marked truth point.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CANDIDATE_SUMMARY = (
    ROOT / "tests" / "fixtures" / "vertex_point_candidates_easy_corpus_summary.json"
)
DEFAULT_FEEDBACK = ROOT / "tests" / "fixtures" / "vertex_point_human_feedback.json"
DEFAULT_REPORT = ROOT / "tools" / "VERTEX_POINT_HUMAN_FEEDBACK_REPORT.md"
DEFAULT_DISTANCE_THRESHOLD_PX = 10.0

Point = Tuple[float, float]


def build_feedback_scaffold(
    candidate_document: Dict[str, Any],
    *,
    distance_threshold_px: float = DEFAULT_DISTANCE_THRESHOLD_PX,
    source_candidate_summary: str = str(DEFAULT_CANDIDATE_SUMMARY),
) -> Dict[str, Any]:
    """Create an unlabeled human-feedback fixture from candidate rows."""
    rows: List[Dict[str, Any]] = []
    for row in candidate_document.get("rows", []):
        candidates = [
            {
                "rank": candidate.get("rank"),
                "source": candidate.get("source"),
                "vertexPoint": candidate.get("vertexPoint"),
                "modelStatus": candidate.get("modelStatus"),
                "modelScore": candidate.get("modelScore"),
                "scoreComponents": candidate.get("scoreComponents", {}),
            }
            for candidate in row.get("topCandidates", [])
        ]
        rows.append({
            "setId": str(row.get("setId")),
            "side": row.get("side"),
            "evaluationTier": row.get("evaluationTier"),
            "imagePath": row.get("imagePath"),
            "overlayPath": row.get("overlayPath"),
            "candidateStatus": row.get("status"),
            "candidateDiagnostics": row.get("candidateDiagnostics", {}),
            "status": "unlabeled",
            "humanVertexPoint": None,
            "labelQuality": None,
            "notes": "",
            "target": (
                "Mark the visible trihedral vertex point. A top-1 hit is within "
                f"{distance_threshold_px:g}px; top-3 recall means any of ranks "
                "1-3 is within that threshold."
            ),
            "topCandidates": candidates,
        })

    return {
        "schemaVersion": 1,
        "probe": "vertex_point_human_feedback_v0",
        "description": (
            "Human labels for visible trihedral-corner candidate diagnostics. "
            "Rows start as unlabeled scaffolds; future labels should set "
            "status='labeled' and humanVertexPoint=[x,y]."
        ),
        "sourceCandidateSummary": source_candidate_summary,
        "distanceThresholdPx": float(distance_threshold_px),
        "allowedStatuses": ["unlabeled", "labeled", "ambiguous", "not_visible"],
        "rows": rows,
    }


def evaluate_feedback(feedback_document: Dict[str, Any]) -> Dict[str, Any]:
    """Return row-level and aggregate top-1/top-3 metrics."""
    threshold = float(
        feedback_document.get("distanceThresholdPx", DEFAULT_DISTANCE_THRESHOLD_PX)
    )
    evaluated_rows: List[Dict[str, Any]] = []
    for row in feedback_document.get("rows", []):
        evaluated_rows.append(evaluate_feedback_row(row, threshold))

    labeled = [row for row in evaluated_rows if row["evaluationStatus"] == "labeled"]
    top1_hits = [row for row in labeled if row.get("top1WithinThreshold")]
    top3_hits = [row for row in labeled if row.get("top3ContainsTruth")]
    missed = [row for row in labeled if not row.get("top3ContainsTruth")]
    unlabeled = [row for row in evaluated_rows if row["evaluationStatus"] == "unlabeled"]
    ambiguous = [row for row in evaluated_rows if row["evaluationStatus"] == "ambiguous"]
    not_visible = [row for row in evaluated_rows if row["evaluationStatus"] == "not_visible"]
    invalid = [row for row in evaluated_rows if row["evaluationStatus"] == "invalid_label"]

    return {
        "distanceThresholdPx": threshold,
        "summary": {
            "rowCount": len(evaluated_rows),
            "labeledRowCount": len(labeled),
            "unlabeledRowCount": len(unlabeled),
            "ambiguousRowCount": len(ambiguous),
            "notVisibleRowCount": len(not_visible),
            "invalidLabelRowCount": len(invalid),
            "top1HitCount": len(top1_hits),
            "top3HitCount": len(top3_hits),
            "top3MissCount": len(missed),
            "top1HitRate": _rate(len(top1_hits), len(labeled)),
            "top3HitRate": _rate(len(top3_hits), len(labeled)),
        },
        "rows": evaluated_rows,
    }


def evaluate_feedback_row(row: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    status = str(row.get("status") or "unlabeled")
    base = {
        "setId": str(row.get("setId")),
        "side": row.get("side"),
        "evaluationTier": row.get("evaluationTier"),
        "candidateStatus": row.get("candidateStatus"),
        "overlayPath": row.get("overlayPath"),
        "labelStatus": status,
    }
    if status in {"unlabeled", "ambiguous", "not_visible"}:
        return {**base, "evaluationStatus": status}
    if status != "labeled":
        return {**base, "evaluationStatus": "invalid_label", "error": f"unknown status {status!r}"}

    human_point = _parse_point(row.get("humanVertexPoint"))
    if human_point is None:
        return {
            **base,
            "evaluationStatus": "invalid_label",
            "error": "labeled row is missing humanVertexPoint=[x,y]",
        }

    distances: List[Dict[str, Any]] = []
    for candidate in row.get("topCandidates", []):
        candidate_point = _parse_point(candidate.get("vertexPoint"))
        if candidate_point is None:
            continue
        distance = _distance_px(human_point, candidate_point)
        distances.append({
            "rank": candidate.get("rank"),
            "source": candidate.get("source"),
            "distancePx": round(distance, 2),
            "withinThreshold": distance <= threshold,
            "vertexPoint": [round(candidate_point[0], 2), round(candidate_point[1], 2)],
        })

    if not distances:
        return {
            **base,
            "evaluationStatus": "invalid_label",
            "error": "labeled row has no candidate points to score",
        }

    distances.sort(key=lambda item: int(item["rank"] or 999))
    top1 = next((item for item in distances if item.get("rank") == 1), distances[0])
    top3 = [item for item in distances if int(item.get("rank") or 999) <= 3]
    best = min(distances, key=lambda item: float(item["distancePx"]))
    return {
        **base,
        "evaluationStatus": "labeled",
        "humanVertexPoint": [round(human_point[0], 2), round(human_point[1], 2)],
        "candidateDistances": distances,
        "top1DistancePx": top1["distancePx"],
        "top1WithinThreshold": bool(top1["withinThreshold"]),
        "top3ContainsTruth": any(bool(item["withinThreshold"]) for item in top3),
        "bestRank": best.get("rank"),
        "bestDistancePx": best["distancePx"],
    }


def render_report(feedback_document: Dict[str, Any], evaluation: Dict[str, Any]) -> str:
    summary = evaluation["summary"]
    threshold = evaluation["distanceThresholdPx"]
    lines = [
        "# Vertex Point Human Feedback",
        "",
        "Diagnostics/data-only artifact. This does not alter recognition behavior.",
        "",
        "The purpose is to validate whether the ranked vertex-point candidates actually hit the human-visible trihedral corner.",
        "",
        "## Summary",
        "",
        f"- Distance threshold: {threshold:g}px",
        f"- Rows: {summary['rowCount']}",
        f"- Labeled rows: {summary['labeledRowCount']}",
        f"- Unlabeled rows: {summary['unlabeledRowCount']}",
        f"- Ambiguous rows: {summary['ambiguousRowCount']}",
        f"- Not-visible rows: {summary['notVisibleRowCount']}",
        f"- Invalid-label rows: {summary['invalidLabelRowCount']}",
        f"- Top-1 hits: {summary['top1HitCount']}",
        f"- Top-3 hits: {summary['top3HitCount']}",
        f"- Top-3 misses: {summary['top3MissCount']}",
        f"- Top-1 hit rate: {_format_rate(summary['top1HitRate'])}",
        f"- Top-3 hit rate: {_format_rate(summary['top3HitRate'])}",
        "",
        "## Readout",
        "",
        "| Set | Side | Tier | Label | Candidate status | Top-1 dist | Top-1 hit | Top-3 hit | Best rank | Best dist | Overlay |",
        "|---:|---|---|---|---|---:|---|---|---:|---:|---|",
    ]
    for row in evaluation["rows"]:
        overlay = row.get("overlayPath") or ""
        overlay_text = f"`{overlay}`" if overlay else ""
        lines.append(
            f"| {row.get('setId')} | {row.get('side', '')} | `{row.get('evaluationTier', '')}` | "
            f"`{row.get('evaluationStatus')}` | `{row.get('candidateStatus', '')}` | "
            f"{row.get('top1DistancePx', '')} | {_bool_text(row.get('top1WithinThreshold'))} | "
            f"{_bool_text(row.get('top3ContainsTruth'))} | {row.get('bestRank', '')} | "
            f"{row.get('bestDistancePx', '')} | {overlay_text} |"
        )

    source = feedback_document.get("sourceCandidateSummary", "")
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Source candidate summary: `{source}`",
        "- Rows begin unlabeled by design. Human feedback should only add label fields; candidate coordinates should remain generated data.",
        "- Top-1 precision tells us whether the automatic first choice is trustworthy enough to seed downstream geometry.",
        "- Top-3 recall tells us whether the correct vertex is at least present in the candidate set for a later fitter or manual review.",
        "- A top-3 miss on an easy-corpus row is a geometry-init failure, not a color-classification problem.",
        "",
    ])
    return "\n".join(lines)


def _parse_point(value: Any) -> Optional[Point]:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return (float(value[0]), float(value[1]))
    return None


def _distance_px(left: Point, right: Point) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _format_rate(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _bool_text(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", type=Path, default=DEFAULT_CANDIDATE_SUMMARY)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--distance-threshold-px", type=float, default=DEFAULT_DISTANCE_THRESHOLD_PX)
    parser.add_argument(
        "--write-scaffold",
        action="store_true",
        help="Create the feedback fixture from the candidate summary.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --write-scaffold to replace an existing feedback file.",
    )
    args = parser.parse_args(argv)

    if args.write_scaffold:
        if args.feedback.exists() and not args.force:
            print(
                f"error: {args.feedback} already exists; use --force only when you "
                "intend to replace human feedback labels.",
                file=sys.stderr,
            )
            return 2
        candidate_document = _read_json(args.candidate_summary)
        scaffold = build_feedback_scaffold(
            candidate_document,
            distance_threshold_px=args.distance_threshold_px,
            source_candidate_summary=str(args.candidate_summary),
        )
        _write_json(args.feedback, scaffold)
        print(f"wrote {args.feedback}")

    feedback = _read_json(args.feedback)
    evaluation = evaluate_feedback(feedback)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(feedback, evaluation), encoding="utf-8")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
