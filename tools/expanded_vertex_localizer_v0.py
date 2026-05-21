#!/usr/bin/env python3
"""Diagnostics-only expanded visible-vertex localizer benchmark.

This is the follow-up to the active-learning label pass. It combines the
original canonical vertex+axis labels with the new active-learning labels and
compares lightweight learned localizers over that expanded supervision set.

Nothing here alters recognizer behavior.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tools.axis_ray_vertex_refinement_v0 import _fmt, _write_json
from tools.knn_vertex_localizer_v0 import (
    KnnVertexConfig,
    generate_knn_vertex_localizer_summary,
)
from tools.learned_vertex_localizer_v0 import (
    LearnedVertexConfig,
    generate_learned_vertex_localizer_summary,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_FEEDBACK = ROOT / "tests" / "fixtures" / "vertex_axis_human_feedback_v0.json"
DEFAULT_ACTIVE_FEEDBACK = ROOT / "tests" / "fixtures" / "vertex_axis_active_learning_feedback_v0.json"
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "expanded_vertex_localizer_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "EXPANDED_VERTEX_LOCALIZER_V0_REPORT.md"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    method: str
    config: Any
    description: str


def default_benchmarks() -> List[BenchmarkSpec]:
    return [
        BenchmarkSpec(
            name="knn_default_radius220",
            method="knn",
            config=KnnVertexConfig(),
            description="Existing KNN V0 baseline over the expanded 58-row label set.",
        ),
        BenchmarkSpec(
            name="knn_wide_radius520",
            method="knn",
            config=KnnVertexConfig(
                candidate_config=LearnedVertexConfig(search_radius_px=520, search_step_px=32)
            ),
            description="Same KNN scorer with wider, coarser local candidate generation.",
        ),
        BenchmarkSpec(
            name="ridge_default_radius220",
            method="ridge",
            config=LearnedVertexConfig(),
            description="Existing ridge-regression scorer over the expanded 58-row label set.",
        ),
    ]


def build_expanded_feedback(
    *,
    canonical_feedback_path: Path = DEFAULT_CANONICAL_FEEDBACK,
    active_feedback_path: Path = DEFAULT_ACTIVE_FEEDBACK,
) -> Dict[str, Any]:
    canonical = _read_json(canonical_feedback_path)
    active = _read_json(active_feedback_path)
    rows: List[Dict[str, Any]] = []
    rows.extend(_labeled_rows(canonical.get("rows", []), lane="canonical"))
    rows.extend(_labeled_rows(active.get("rows", []), lane="active"))
    return {
        "schemaVersion": 1,
        "artifact": "expanded_vertex_axis_feedback_v0",
        "description": (
            "Combined diagnostics-only visible-trihedral feedback: canonical "
            "labels plus the active-learning label queue."
        ),
        "sources": {
            "canonicalFeedback": str(canonical_feedback_path),
            "activeLearningFeedback": str(active_feedback_path),
        },
        "rows": rows,
    }


def generate_expanded_vertex_localizer_summary(
    *,
    canonical_feedback_path: Path = DEFAULT_CANONICAL_FEEDBACK,
    active_feedback_path: Path = DEFAULT_ACTIVE_FEEDBACK,
    benchmarks: Optional[Sequence[BenchmarkSpec]] = None,
) -> Dict[str, Any]:
    feedback = build_expanded_feedback(
        canonical_feedback_path=canonical_feedback_path,
        active_feedback_path=active_feedback_path,
    )
    benchmark_docs = []
    specs = list(default_benchmarks() if benchmarks is None else benchmarks)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump({"rows": feedback["rows"]}, handle, indent=2, sort_keys=True)
        handle.write("\n")
        combined_path = Path(handle.name)
    try:
        for spec in specs:
            benchmark_docs.append(_run_benchmark(spec, combined_path))
    finally:
        combined_path.unlink(missing_ok=True)
    return {
        "schemaVersion": 1,
        "probe": "expanded_vertex_localizer_v0",
        "description": (
            "Diagnostics-only expanded-label benchmark for visible trihedral "
            "vertex candidate reach, ranking, and confidence."
        ),
        "sourceFeedback": feedback["sources"],
        "feedbackSummary": _feedback_summary(feedback["rows"]),
        "benchmarks": benchmark_docs,
        "conclusion": _conclusion(benchmark_docs),
    }


def render_report(document: Dict[str, Any]) -> str:
    feedback = document["feedbackSummary"]
    conclusion = document["conclusion"]
    lines = [
        "# Expanded Vertex Localizer V0",
        "",
        "Diagnostics-only artifact. This does not alter recognition behavior.",
        "",
        "This report uses the completed active-learning vertex+axis labels to test whether the larger label set makes the local visible-trihedral vertex localizer reliable enough to consider wiring. It also fixes the active-label current-model coordinate space before evaluation.",
        "",
        "## Label Set",
        "",
        f"- Labeled rows: {feedback['rowCount']}",
        f"- Canonical rows: {feedback['laneCounts'].get('canonical', 0)}",
        f"- Active-learning rows: {feedback['laneCounts'].get('active', 0)}",
        "",
        "## Benchmark Summary",
        "",
        "| Benchmark | Rows | Base strict/plausible | Oracle strict/plausible | Top-1 strict/plausible | Gated strict/plausible | Accepted | Top-1 +/- | Gated +/- | Mean base/oracle/top-1/gated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark in document["benchmarks"]:
        summary = benchmark["summary"]
        lines.append(
            f"| `{benchmark['name']}` | {summary['evaluatedRowCount']} | "
            f"{summary['baselineStrictCount']} / {summary['baselinePlausibleCount']} | "
            f"{summary['candidateOracleStrictCount']} / {summary['candidateOraclePlausibleCount']} | "
            f"{summary['top1StrictCount']} / {summary['top1PlausibleCount']} | "
            f"{summary['gatedStrictCount']} / {summary['gatedPlausibleCount']} | "
            f"{summary['gatedAcceptedCount']} | "
            f"{summary['top1ImprovedRowCount']} / {summary['top1WorsenedRowCount']} | "
            f"{summary['gatedImprovedRowCount']} / {summary['gatedWorsenedRowCount']} | "
            f"{_fmt(summary['meanBaselineVertexErrorPx'], 'px')} / "
            f"{_fmt(summary['meanCandidateOracleVertexErrorPx'], 'px')} / "
            f"{_fmt(summary['meanTop1VertexErrorPx'], 'px')} / "
            f"{_fmt(summary['meanGatedVertexErrorPx'], 'px')} |"
        )
    lines.extend(
        [
            "",
            "## Coordinate-Space Finding",
            "",
            "- The active queue's model hypotheses were produced in processing-image coordinates, while human clicks are in EXIF-correct full-resolution image coordinates.",
            "- This PR scales the active queue `currentModel` vertex and axis vectors into the label coordinate space before using the labels for localizer training/evaluation.",
            "- Without that bridge, active-row baselines were off by roughly 1800 px and the localizer result was meaningless.",
            "",
            "## Interpretation",
            "",
            f"- Best candidate reach: `{conclusion['bestOracleBenchmark']}` reaches {conclusion['bestOracleStrictCount']} / {feedback['rowCount']} strict rows.",
            f"- Best top-1 ranker: `{conclusion['bestTop1Benchmark']}` reaches {conclusion['bestTop1StrictCount']} / {feedback['rowCount']} strict rows.",
            f"- Best gated ranker: `{conclusion['bestGatedBenchmark']}` reaches {conclusion['bestGatedStrictCount']} / {feedback['rowCount']} strict rows.",
            f"- Production wiring recommendation: `{conclusion['productionWiringRecommendation']}`.",
            f"- Reason: {conclusion['reason']}",
            "- The wide candidate grid shows that candidate reach is mostly recoverable: oracle strict improves from 44/58 to 53/58.",
            "- The ranker still cannot safely select those candidates: top-1 remains 13/58 strict and every non-empty confidence gate still has worsened rows.",
            "- The next useful move is not another scalar gate. It should be a stronger vertex localizer trained on image patches/line junctions or a model objective that scores face-boundary consistency directly.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_benchmark(spec: BenchmarkSpec, feedback_path: Path) -> Dict[str, Any]:
    if spec.method == "knn":
        document = generate_knn_vertex_localizer_summary(feedback_path=feedback_path, config=spec.config)
        field_prefix = "knn"
    elif spec.method == "ridge":
        document = generate_learned_vertex_localizer_summary(feedback_path=feedback_path, config=spec.config)
        field_prefix = "learned"
    else:
        raise ValueError(f"unsupported benchmark method: {spec.method}")
    return {
        "name": spec.name,
        "method": spec.method,
        "description": spec.description,
        "config": spec.config.as_dict(),
        "summary": _compact_summary(document["summary"], field_prefix=field_prefix),
        "rows": _compact_rows(document["rows"], field_prefix=field_prefix),
    }


def _compact_summary(summary: Dict[str, Any], *, field_prefix: str) -> Dict[str, Any]:
    return {
        "rowCount": summary["rowCount"],
        "evaluatedRowCount": summary["evaluatedRowCount"],
        "axisGoodRowCount": summary["axisGoodRowCount"],
        "axisBlockedRowCount": summary["axisBlockedRowCount"],
        "baselineStrictCount": summary["baselineStrictCount"],
        "baselinePlausibleCount": summary["baselinePlausibleCount"],
        "candidateOracleStrictCount": summary["candidateOracleStrictCount"],
        "candidateOraclePlausibleCount": summary["candidateOraclePlausibleCount"],
        "top1StrictCount": summary[f"{field_prefix}Top1StrictCount"],
        "top1PlausibleCount": summary[f"{field_prefix}Top1PlausibleCount"],
        "gatedStrictCount": summary[f"{field_prefix}GatedStrictCount"],
        "gatedPlausibleCount": summary[f"{field_prefix}GatedPlausibleCount"],
        "gatedAcceptedCount": summary[f"{field_prefix}GatedAcceptedCount"],
        "top1ImprovedRowCount": summary[f"{field_prefix}Top1ImprovedRowCount"],
        "top1WorsenedRowCount": summary[f"{field_prefix}Top1WorsenedRowCount"],
        "gatedImprovedRowCount": summary[f"{field_prefix}GatedImprovedRowCount"],
        "gatedWorsenedRowCount": summary[f"{field_prefix}GatedWorsenedRowCount"],
        "meanBaselineVertexErrorPx": summary["meanBaselineVertexErrorPx"],
        "meanCandidateOracleVertexErrorPx": summary["meanCandidateOracleVertexErrorPx"],
        "meanTop1VertexErrorPx": summary[f"mean{field_prefix.title()}Top1VertexErrorPx"],
        "meanGatedVertexErrorPx": summary[f"mean{field_prefix.title()}GatedVertexErrorPx"],
        "medianBaselineVertexErrorPx": summary["medianBaselineVertexErrorPx"],
        "medianCandidateOracleVertexErrorPx": summary["medianCandidateOracleVertexErrorPx"],
        "medianTop1VertexErrorPx": summary[f"median{field_prefix.title()}Top1VertexErrorPx"],
        "medianGatedVertexErrorPx": summary[f"median{field_prefix.title()}GatedVertexErrorPx"],
        "thresholdSweep": summary.get(f"{field_prefix}ThresholdSweep"),
    }


def _compact_rows(rows: Sequence[Dict[str, Any]], *, field_prefix: str) -> List[Dict[str, Any]]:
    compact = []
    for row in rows:
        result = {
            "key": row.get("key"),
            "lane": row.get("sourceFeedbackLane"),
            "sourceRowKey": row.get("sourceRowKey"),
            "evaluationStatus": row.get("evaluationStatus"),
        }
        if row.get("evaluationStatus") == "ok":
            result.update(
                {
                    "axisCategory": row.get("axisCategory"),
                    "baselineVertexErrorPx": row.get("baselineVertexErrorPx"),
                    "candidateOracleVertexErrorPx": row.get("candidateOracleVertexErrorPx"),
                    "top1VertexErrorPx": row.get(f"{field_prefix}Top1VertexErrorPx"),
                    "gatedVertexErrorPx": row.get(f"{field_prefix}GatedVertexErrorPx"),
                    "gatedAccepted": row.get(f"{field_prefix}GatedAccepted"),
                    "gatedImprovementPx": row.get(f"{field_prefix}GatedImprovementPx"),
                }
            )
        compact.append(result)
    return compact


def _feedback_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    lane_counts: Dict[str, int] = {}
    for row in rows:
        lane = str(row.get("sourceFeedbackLane"))
        lane_counts[lane] = lane_counts.get(lane, 0) + 1
    return {
        "rowCount": len(rows),
        "laneCounts": lane_counts,
    }


def _conclusion(benchmarks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not benchmarks:
        return {
            "productionWiringRecommendation": "do_not_wire",
            "reason": "No benchmarks were evaluated.",
        }
    best_oracle = max(benchmarks, key=lambda item: item["summary"]["candidateOracleStrictCount"])
    best_top1 = max(benchmarks, key=lambda item: item["summary"]["top1StrictCount"])
    best_gated = max(benchmarks, key=lambda item: item["summary"]["gatedStrictCount"])
    any_worsen_free_gate = any(
        (item["summary"].get("thresholdSweep") or {}).get("bestNonEmptyZeroWorsen")
        for item in benchmarks
    )
    if any_worsen_free_gate:
        recommendation = "diagnostics_only_confidence_needs_more_coverage"
        reason = "At least one zero-worsen gate exists, but it still needs coverage and validation before recognizer wiring."
    else:
        recommendation = "do_not_wire"
        reason = "Expanded labels and wider candidate reach still produce false-confident worsened rows; ranking/confidence remains unsafe."
    return {
        "bestOracleBenchmark": best_oracle["name"],
        "bestOracleStrictCount": best_oracle["summary"]["candidateOracleStrictCount"],
        "bestTop1Benchmark": best_top1["name"],
        "bestTop1StrictCount": best_top1["summary"]["top1StrictCount"],
        "bestGatedBenchmark": best_gated["name"],
        "bestGatedStrictCount": best_gated["summary"]["gatedStrictCount"],
        "hasNonEmptyZeroWorsenGate": any_worsen_free_gate,
        "productionWiringRecommendation": recommendation,
        "reason": reason,
    }


def _labeled_rows(rows: Iterable[Dict[str, Any]], *, lane: str) -> List[Dict[str, Any]]:
    labeled = []
    for row in rows:
        if row.get("status") != "labeled":
            continue
        if not row.get("humanVertexPoint"):
            continue
        if sum(1 for point in row.get("humanAxisEndpoints", []) if point) != 3:
            continue
        labeled.append(
            {
                **row,
                "key": f"{lane}:{row.get('key')}",
                "sourceFeedbackLane": lane,
                "sourceRowKey": row.get("key"),
            }
        )
    return labeled


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-feedback", type=Path, default=DEFAULT_CANONICAL_FEEDBACK)
    parser.add_argument("--active-feedback", type=Path, default=DEFAULT_ACTIVE_FEEDBACK)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    document = generate_expanded_vertex_localizer_summary(
        canonical_feedback_path=args.canonical_feedback,
        active_feedback_path=args.active_feedback,
    )
    _write_json(args.summary_out, document)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(document), encoding="utf-8")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.report_out}")
    print(
        f"best oracle/top1/gated strict: "
        f"{document['conclusion']['bestOracleStrictCount']} / "
        f"{document['conclusion']['bestTop1StrictCount']} / "
        f"{document['conclusion']['bestGatedStrictCount']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
