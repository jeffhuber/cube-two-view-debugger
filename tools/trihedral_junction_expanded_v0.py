#!/usr/bin/env python3
"""Diagnostics-only expanded-label explicit trihedral junction benchmark.

The first explicit line/junction probe only covered the original 28 canonical
vertex+axis labels. This wrapper reruns the same structural detector over the
current 58-row expanded label set, including active-learning rows whose model
coordinates have been scaled into EXIF-correct source-image coordinates.

Nothing here alters recognizer behavior.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from tools.axis_ray_vertex_refinement_v0 import _fmt, _write_json
from tools.expanded_vertex_localizer_v0 import (
    DEFAULT_ACTIVE_FEEDBACK,
    DEFAULT_CANONICAL_FEEDBACK,
    build_expanded_feedback,
)
from tools.trihedral_junction_extraction_v0 import (
    JunctionConfig,
    generate_trihedral_junction_extraction_summary,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "trihedral_junction_expanded_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "TRIHEDRAL_JUNCTION_EXPANDED_V0_REPORT.md"


def generate_trihedral_junction_expanded_summary(
    *,
    canonical_feedback_path: Path = DEFAULT_CANONICAL_FEEDBACK,
    active_feedback_path: Path = DEFAULT_ACTIVE_FEEDBACK,
    config: JunctionConfig = JunctionConfig(),
) -> Dict[str, Any]:
    feedback = build_expanded_feedback(
        canonical_feedback_path=canonical_feedback_path,
        active_feedback_path=active_feedback_path,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump({"rows": feedback["rows"]}, handle, indent=2, sort_keys=True)
        handle.write("\n")
        combined_path = Path(handle.name)
    try:
        extraction = generate_trihedral_junction_extraction_summary(
            feedback_path=combined_path,
            config=config,
        )
    finally:
        combined_path.unlink(missing_ok=True)
    rows = [_compact_row(row) for row in extraction["rows"]]
    summary = extraction["summary"]
    return {
        "schemaVersion": 1,
        "probe": "trihedral_junction_expanded_v0",
        "description": (
            "Diagnostics-only expanded-label rerun of explicit dark-line "
            "trihedral junction extraction."
        ),
        "sourceFeedback": feedback["sources"],
        "feedbackSummary": _feedback_summary(feedback),
        "config": extraction["config"],
        "summary": summary,
        "conclusion": _conclusion(summary),
        "rows": rows,
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    feedback = document["feedbackSummary"]
    conclusion = document["conclusion"]
    lines = [
        "# Trihedral Junction Expanded V0",
        "",
        "Diagnostics-only artifact. This does not alter recognition behavior.",
        "",
        "This report reruns the explicit dark-line trihedral junction extractor on the current 58-row vertex+axis label set. It answers whether the old structural detector becomes useful once evaluated against the expanded labels and fixed active-label coordinate space.",
        "",
        "## Label Set",
        "",
        f"- Labeled rows: {feedback['rowCount']}",
        f"- Canonical rows: {feedback['laneCounts'].get('canonical', 0)}",
        f"- Active-learning rows: {feedback['laneCounts'].get('active', 0)}",
        "",
        "## Summary",
        "",
        f"- Evaluated rows: {summary['evaluatedRowCount']}",
        f"- Axis-good rows: {summary['axisGoodRowCount']}",
        f"- Axis-blocked rows: {summary['axisBlockedRowCount']}",
        f"- Baseline strict/plausible: {summary['baselineStrictCount']} / {summary['baselinePlausibleCount']}",
        f"- Model junction strict/plausible: {summary['modelJunctionStrictCount']} / {summary['modelJunctionPlausibleCount']}",
        f"- Model junction gated strict/plausible: {summary['modelJunctionGatedStrictCount']} / {summary['modelJunctionGatedPlausibleCount']}",
        f"- Human-axis oracle strict/plausible: {summary['humanAxisOracleStrictCount']} / {summary['humanAxisOraclePlausibleCount']}",
        f"- Human-axis oracle gated strict/plausible: {summary['humanAxisOracleGatedStrictCount']} / {summary['humanAxisOracleGatedPlausibleCount']}",
        f"- Axis-good strict baseline/model-gated: {summary['axisGoodBaselineStrictCount']} / {summary['axisGoodModelGatedStrictCount']}",
        f"- Model junction gated accepted rows: {summary['modelJunctionGatedAcceptedCount']}",
        f"- Model junction gated improved/worsened rows by >5px: {summary['modelJunctionGatedImprovedRowCount']} / {summary['modelJunctionGatedWorsenedRowCount']}",
        f"- Human-axis oracle gated improved/worsened rows by >5px: {summary['humanAxisOracleGatedImprovedRowCount']} / {summary['humanAxisOracleGatedWorsenedRowCount']}",
        f"- Mean vertex error baseline/model-gated/oracle-gated: {_fmt(summary['meanBaselineVertexErrorPx'], 'px')} / {_fmt(summary['meanModelJunctionGatedVertexErrorPx'], 'px')} / {_fmt(summary['meanHumanAxisOracleGatedVertexErrorPx'], 'px')}",
        f"- Median vertex error baseline/model-gated/oracle-gated: {_fmt(summary['medianBaselineVertexErrorPx'], 'px')} / {_fmt(summary['medianModelJunctionGatedVertexErrorPx'], 'px')} / {_fmt(summary['medianHumanAxisOracleGatedVertexErrorPx'], 'px')}",
        f"- Best non-empty low-worsen model gate: {_format_sweep(summary['modelJunctionThresholdSweep'].get('bestNonEmptyLowWorsen'))}",
        f"- Best non-empty model gate with <=2 worsens: {_format_sweep(summary['modelJunctionThresholdSweep'].get('bestNonEmptyAtMostTwoWorsens'))}",
        f"- Production wiring recommendation: `{conclusion['productionWiringRecommendation']}`.",
        f"- Reason: {conclusion['reason']}",
        "",
        "## Rows",
        "",
        "| Row | Axis | Base | Model best | Model gated | Accepted | Delta | Spread | Min score | Oracle gated | Oracle delta |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in document["rows"]:
        if row.get("evaluationStatus") != "ok":
            lines.append(f"| `{row.get('key')}` | `{row.get('evaluationStatus')}` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| `{row.get('key')}` | `{row.get('axisCategory')}` | "
            f"{_fmt(row.get('baselineVertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('modelJunctionVertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('modelJunctionGatedVertexErrorPx'), 'px')} | "
            f"{'yes' if row.get('modelJunctionGatedAccepted') else 'no'} | "
            f"{_fmt(row.get('modelJunctionGatedImprovementPx'), 'px')} | "
            f"{_fmt(row.get('intersectionSpreadPx'), 'px')} | "
            f"{_fmt_plain(row.get('minLineScore'))} | "
            f"{_fmt(row.get('humanAxisOracleGatedVertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('humanAxisOracleGatedImprovementPx'), 'px')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is an expanded-label rerun of explicit line/junction extraction, not production wiring.",
            "- The result is a strong negative for this V0 structural detector: model-gated output underperforms baseline strict/plausible counts and accepts more worsened rows than improved rows.",
            "- The human-axis oracle also underperforms baseline, which means the failure is not only model-axis choice; the extracted line intersections themselves are not stable enough.",
            "- The useful conclusion is to stop revisiting this particular dark-line intersection objective. A future line path would need a materially different detector, such as segment grouping with actual face-boundary topology or a trained patch model.",
            "",
        ]
    )
    return "\n".join(lines)


def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "key": row.get("key"),
        "lane": row.get("sourceFeedbackLane"),
        "sourceRowKey": row.get("sourceRowKey"),
        "evaluationStatus": row.get("evaluationStatus"),
    }
    if row.get("evaluationStatus") != "ok":
        return result
    diagnostics = row.get("modelJunction") or {}
    result.update(
        {
            "axisCategory": row.get("axisCategory"),
            "baselineVertexErrorPx": row.get("baselineVertexErrorPx"),
            "modelJunctionVertexErrorPx": row.get("modelJunctionVertexErrorPx"),
            "modelJunctionGatedVertexErrorPx": row.get("modelJunctionGatedVertexErrorPx"),
            "modelJunctionGatedAccepted": row.get("modelJunctionGatedAccepted"),
            "modelJunctionGatedImprovementPx": row.get("modelJunctionGatedImprovementPx"),
            "intersectionSpreadPx": diagnostics.get("intersectionSpreadPx"),
            "minLineScore": diagnostics.get("minLineScore"),
            "minLineContrast": diagnostics.get("minLineContrast"),
            "movePx": diagnostics.get("movePx"),
            "humanAxisOracleVertexErrorPx": row.get("humanAxisOracleVertexErrorPx"),
            "humanAxisOracleGatedVertexErrorPx": row.get("humanAxisOracleGatedVertexErrorPx"),
            "humanAxisOracleGatedAccepted": row.get("humanAxisOracleGatedAccepted"),
            "humanAxisOracleGatedImprovementPx": row.get("humanAxisOracleGatedImprovementPx"),
        }
    )
    return result


def _feedback_summary(feedback: Dict[str, Any]) -> Dict[str, Any]:
    lane_counts: Dict[str, int] = {}
    for row in feedback.get("rows", []):
        lane = str(row.get("sourceFeedbackLane"))
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
    return {
        "rowCount": len(feedback.get("rows", [])),
        "laneCounts": lane_counts,
    }


def _conclusion(summary: Dict[str, Any]) -> Dict[str, Any]:
    baseline_strict = int(summary.get("baselineStrictCount") or 0)
    gated_strict = int(summary.get("modelJunctionGatedStrictCount") or 0)
    gated_worsened = int(summary.get("modelJunctionGatedWorsenedRowCount") or 0)
    gated_improved = int(summary.get("modelJunctionGatedImprovedRowCount") or 0)
    if gated_strict > baseline_strict and gated_worsened == 0:
        return {
            "productionWiringRecommendation": "diagnostics_only_needs_more_validation",
            "reason": "Model-junction gating improved strict count without worsened accepted rows on this label set.",
        }
    return {
        "productionWiringRecommendation": "do_not_wire",
        "reason": (
            "Expanded-label explicit junction extraction underperforms baseline "
            f"and accepts {gated_worsened} worsened rows versus {gated_improved} improved rows."
        ),
    }


def _format_sweep(item: Optional[Dict[str, Any]]) -> str:
    if not item:
        return "none"
    thresholds = item.get("thresholds") or {}
    return (
        f"accepted {item['acceptedCount']}, improved {item['improvedCount']}, "
        f"worsened {item['worsenedCount']}, strict/plausible "
        f"{item['strictCount']} / {item['plausibleCount']}, "
        f"mean {_fmt(item['meanVertexErrorPx'], 'px')} "
        f"(spread<={thresholds.get('maxSpreadPx')}, "
        f"score>={thresholds.get('minLineScore')}, "
        f"contrast>={thresholds.get('minLineContrast')}, "
        f"move<={thresholds.get('maxMovePx')})"
    )


def _fmt_plain(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-feedback", type=Path, default=DEFAULT_CANONICAL_FEEDBACK)
    parser.add_argument("--active-feedback", type=Path, default=DEFAULT_ACTIVE_FEEDBACK)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    document = generate_trihedral_junction_expanded_summary(
        canonical_feedback_path=args.canonical_feedback,
        active_feedback_path=args.active_feedback,
    )
    _write_json(args.summary_out, document)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(document), encoding="utf-8")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.report_out}")
    print(
        f"expanded junction gated strict rows: "
        f"{document['summary']['modelJunctionGatedStrictCount']} / "
        f"{document['summary']['evaluatedRowCount']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
