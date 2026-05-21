#!/usr/bin/env python3
"""Evaluate canonical vertex/axis feedback and hypothesis agreement.

Diagnostics/data-only. This module does not alter recognizer behavior.

The current first-principles recognizer direction depends on a trusted visible
trihedral vertex plus three projected axes. Prior probes produced two useful
but separate labeled lanes:

* full-resolution global cube model rows comparing rembg and SAM3 whole-cube
  masks against human vertex labels
* easy-corpus processing-image rows with multiple candidate-source families
  compared against human vertex labels

This probe normalizes both lanes into one canonical feedback shape, expands the
deployable hypothesis pool, and tests whether source agreement can be used as a
safe confidence signal. Label/oracle fields are emitted for evaluation only and
are never used by the deployable policies.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEOMETRY_FIXTURE = ROOT / "tests" / "fixtures" / "geometry_first_face_split_v0_summary.json"
DEFAULT_SOURCE_PROBE_FIXTURE = (
    ROOT / "tests" / "fixtures" / "vertex_candidate_source_probe_easy_corpus_summary.json"
)
DEFAULT_FEEDBACK = ROOT / "tests" / "fixtures" / "vertex_axis_feedback_v0.json"
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "vertex_hypothesis_ensemble_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "VERTEX_HYPOTHESIS_ENSEMBLE_V0_REPORT.md"

LANE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gcm_fullres": {
        "strictPx": 30.0,
        "plausiblePx": 50.0,
        "agreementClusterV0": {"radiusPx": 50.0, "minDistinctSources": 2},
        "strictAgreementClusterV0": {"radiusPx": 20.0, "minDistinctSources": 2},
        "sourcePriority": [
            "sam3",
            "rembg",
            "inverse_residual_weighted",
            "fit_quality_weighted",
            "midpoint",
        ],
    },
    "easy_processing": {
        "strictPx": 10.0,
        "plausiblePx": 20.0,
        "agreementClusterV0": {"radiusPx": 25.0, "minDistinctSources": 4},
        "strictAgreementClusterV0": {"radiusPx": 20.0, "minDistinctSources": 4},
        "sourcePriority": [
            "model_ranked",
            "model_local_grid",
            "bezel_axis_ray",
            "bezel_line_intersection",
            "dark_junction_grid",
        ],
    },
}

POLICY_ORDER = [
    "source_priority_top1_v0",
    "agreement_cluster_v0",
    "strict_agreement_cluster_v0",
    "oracle_best_candidate",
]

Point = Tuple[float, float]


def generate_vertex_hypothesis_ensemble(
    *,
    geometry_fixture_path: Path = DEFAULT_GEOMETRY_FIXTURE,
    source_probe_fixture_path: Path = DEFAULT_SOURCE_PROBE_FIXTURE,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(feedback_document, summary_document)`` for the ensemble probe."""
    geometry_document = _read_json(geometry_fixture_path)
    source_probe_document = _read_json(source_probe_fixture_path)
    feedback = build_canonical_feedback(
        geometry_document=geometry_document,
        source_probe_document=source_probe_document,
        geometry_fixture_path=geometry_fixture_path,
        source_probe_fixture_path=source_probe_fixture_path,
    )
    summary = evaluate_feedback(feedback)
    return feedback, summary


def build_canonical_feedback(
    *,
    geometry_document: Dict[str, Any],
    source_probe_document: Dict[str, Any],
    geometry_fixture_path: Path = DEFAULT_GEOMETRY_FIXTURE,
    source_probe_fixture_path: Path = DEFAULT_SOURCE_PROBE_FIXTURE,
) -> Dict[str, Any]:
    """Normalize current labeled vertex data into a reusable feedback fixture."""
    lanes = {
        "gcm_fullres": _build_gcm_lane(geometry_document),
        "easy_processing": _build_easy_lane(source_probe_document),
    }
    return {
        "schemaVersion": 1,
        "artifact": "vertex_axis_feedback_v0",
        "description": (
            "Canonical diagnostics/data-only feedback fixture for visible "
            "trihedral vertex and axis hypothesis selection. Truth/oracle "
            "fields are for evaluation only."
        ),
        "sources": {
            "geometryFirstFaceSplitV0": str(geometry_fixture_path),
            "vertexCandidateSourceProbeEasyCorpus": str(source_probe_fixture_path),
        },
        "laneConfigs": LANE_CONFIGS,
        "lanes": lanes,
    }


def evaluate_feedback(feedback: Dict[str, Any]) -> Dict[str, Any]:
    lane_results: Dict[str, Any] = {}
    for lane_name, lane in feedback["lanes"].items():
        lane_results[lane_name] = _evaluate_lane(lane_name, lane["rows"])

    return {
        "schemaVersion": 1,
        "probe": "vertex_hypothesis_ensemble_v0",
        "description": (
            "Diagnostics/data-only ensemble probe for canonical vertex/axis "
            "feedback. Evaluates expanded candidate pools and agreement-based "
            "confidence policies."
        ),
        "sourceFeedback": str(DEFAULT_FEEDBACK),
        "policyOrder": POLICY_ORDER,
        "laneResults": lane_results,
        "conclusion": _overall_conclusion(lane_results),
    }


def render_report(document: Dict[str, Any]) -> str:
    lines = [
        "# Vertex Hypothesis Ensemble V0",
        "",
        "Diagnostics/data-only artifact. This does not alter recognition behavior.",
        "",
        "This report drives the current three-step path to a conclusion: canonicalize vertex/axis feedback, expand the hypothesis pool, then test whether agreement can become a safe confidence signal.",
        "",
        "## Summary",
        "",
    ]

    for lane_name, lane in document["laneResults"].items():
        config = lane["config"]
        lines.extend(
            [
                f"### `{lane_name}`",
                "",
                f"- Rows: {lane['rowCount']}",
                f"- Strict threshold: {config['strictPx']:.0f} px",
                f"- Plausible threshold: {config['plausiblePx']:.0f} px",
                f"- Candidate count: mean {lane['candidateSummary']['meanCandidateCount']:.1f}, max {lane['candidateSummary']['maxCandidateCount']}",
                f"- Oracle-best candidate: strict {lane['oracleSummary']['strictReachableCount']} / {lane['rowCount']}, plausible {lane['oracleSummary']['plausibleReachableCount']} / {lane['rowCount']}",
                "",
                "| Policy | Selected | Abstained | Strict-ready | Plausible | False-confident | Mean selected error |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for policy in POLICY_ORDER:
            summary = lane["policySummaries"][policy]
            mean = summary["meanSelectedErrorPx"]
            mean_text = "" if mean is None else f"{mean:.1f} px"
            lines.append(
                f"| `{policy}` | {summary['selectedCount']} | {summary['abstainCount']} | "
                f"{summary['strictReadyCount']} | {summary['plausibleCount']} | "
                f"{summary['falseConfidentCount']} | {mean_text} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Agreement Policy Readout",
            "",
            "| Lane | Row | Agreement | Strict agreement | Oracle-best |",
            "|---|---|---|---|---|",
        ]
    )
    for lane_name, lane in document["laneResults"].items():
        for row in lane["rows"]:
            agreement = row["policyResults"]["agreement_cluster_v0"]
            strict = row["policyResults"]["strict_agreement_cluster_v0"]
            oracle = row["policyResults"]["oracle_best_candidate"]
            lines.append(
                f"| `{lane_name}` | `{row['key']}` | "
                f"{_selection_report_text(agreement)} | "
                f"{_selection_report_text(strict)} | "
                f"{_selection_report_text(oracle)} |"
            )

    conclusion = document["conclusion"]
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- Production wiring recommendation: `{conclusion['productionWiringRecommendation']}`.",
            f"- Reason: {conclusion['reason']}",
            "- The canonical fixture is now reusable for later probes, but the current deployable agreement policies still emit false-confident selections.",
            "- The expanded hypothesis pool contains more signal than the current rankers can safely select, especially in the easy-corpus lane. That makes ranking/confidence the blocker, not face splitting.",
            "- The next useful input is richer labels/features around the visible vertex and axes, or a model objective that scores face-boundary consistency directly. More single-score source selection is unlikely to close the gap by itself.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_gcm_lane(geometry_document: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for row in geometry_document.get("rows", []):
        truth = _point_or_none(row.get("trueVertex"))
        if truth is None:
            continue
        candidates = _gcm_candidates(row)
        rows.append(
            _feedback_row(
                lane="gcm_fullres",
                key=str(row.get("key")),
                set_id=str(row.get("setId")),
                side=str(row.get("side")),
                truth=truth,
                candidates=candidates,
                metadata={
                    "bestSourceByVertexError": row.get("bestSourceByVertexError"),
                    "bestSourceStatus": row.get("bestSourceStatus"),
                },
            )
        )
    return {
        "description": "Full-resolution global cube model rows with rembg/SAM3 whole-cube hypotheses.",
        "coordinateSpace": "source_image_full_resolution",
        "rows": rows,
    }


def _build_easy_lane(source_probe_document: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for row in source_probe_document.get("rows", []):
        if row.get("evaluationStatus") != "labeled":
            continue
        truth = _point_or_none(row.get("humanVertexPoint"))
        if truth is None:
            continue
        candidates = _easy_candidates(row)
        rows.append(
            _feedback_row(
                lane="easy_processing",
                key=f"{row.get('setId')}_{row.get('side')}",
                set_id=str(row.get("setId")),
                side=str(row.get("side")),
                truth=truth,
                candidates=candidates,
                metadata={
                    "imagePath": row.get("imagePath"),
                    "evaluationTier": row.get("evaluationTier"),
                    "notes": row.get("notes") or "",
                },
            )
        )
    return {
        "description": "Processing-image easy-corpus rows with human vertex labels and source-probe candidates.",
        "coordinateSpace": "recognizer_processing_image",
        "rows": rows,
    }


def _feedback_row(
    *,
    lane: str,
    key: str,
    set_id: str,
    side: str,
    truth: Point,
    candidates: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    strict_px = float(LANE_CONFIGS[lane]["strictPx"])
    plausible_px = float(LANE_CONFIGS[lane]["plausiblePx"])
    evaluated_candidates = []
    for candidate in candidates:
        point = _point_or_none(candidate.get("point"))
        if point is None:
            continue
        error = _distance(point, truth)
        evaluated_candidates.append(
            {
                **candidate,
                "point": _round_point(point),
                "errorPx": round(error, 2),
                "status": _status_from_error(error, strict_px, plausible_px),
            }
        )
    oracle = _nearest_candidate(evaluated_candidates, truth)
    return {
        "key": key,
        "setId": set_id,
        "side": side,
        "truth": _round_point(truth),
        "metadata": metadata,
        "candidateCount": len(evaluated_candidates),
        "candidates": evaluated_candidates,
        "oracleBestCandidate": oracle,
    }


def _gcm_candidates(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources = row.get("sources") or {}
    candidates: List[Dict[str, Any]] = []
    source_candidates: Dict[str, Dict[str, Any]] = {}
    for source_name in ("rembg", "sam3"):
        source = sources.get(source_name) or {}
        point = _point_or_none(source.get("vertex"))
        if point is None:
            continue
        candidate = {
            "source": source_name,
            "family": "whole_cube_fit",
            "rank": 1,
            "score": _safe_float(source.get("fitQuality")),
            "point": point,
            "details": {
                "fitQuality": source.get("fitQuality"),
                "fitResidualRmsPx": source.get("fitResidualRmsPx"),
                "status": source.get("status"),
                "nondegenerate": source.get("nondegenerate"),
            },
        }
        candidates.append(candidate)
        source_candidates[source_name] = candidate

    rembg = source_candidates.get("rembg")
    sam3 = source_candidates.get("sam3")
    if rembg and sam3:
        rembg_point = _point_or_none(rembg["point"])
        sam3_point = _point_or_none(sam3["point"])
        if rembg_point and sam3_point:
            candidates.append(
                {
                    "source": "midpoint",
                    "family": "derived_pair",
                    "rank": 1,
                    "score": None,
                    "point": _weighted_point([(rembg_point, 1.0), (sam3_point, 1.0)]),
                    "details": {"derivedFrom": ["rembg", "sam3"]},
                }
            )
            residual_weighted = _weighted_by_metric(
                rembg_point,
                sam3_point,
                _safe_float(rembg["details"].get("fitResidualRmsPx")),
                _safe_float(sam3["details"].get("fitResidualRmsPx")),
                inverse=True,
            )
            if residual_weighted is not None:
                candidates.append(
                    {
                        "source": "inverse_residual_weighted",
                        "family": "derived_pair",
                        "rank": 1,
                        "score": None,
                        "point": residual_weighted,
                        "details": {"derivedFrom": ["rembg", "sam3"]},
                    }
                )
            quality_weighted = _weighted_by_metric(
                rembg_point,
                sam3_point,
                _safe_float(rembg["details"].get("fitQuality")),
                _safe_float(sam3["details"].get("fitQuality")),
                inverse=False,
            )
            if quality_weighted is not None:
                candidates.append(
                    {
                        "source": "fit_quality_weighted",
                        "family": "derived_pair",
                        "rank": 1,
                        "score": None,
                        "point": quality_weighted,
                        "details": {"derivedFrom": ["rembg", "sam3"]},
                    }
                )
    return candidates


def _easy_candidates(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for source in row.get("sources", []):
        source_name = str(source.get("source"))
        for item in source.get("topCandidates", []):
            point = _point_or_none(item.get("point"))
            if point is None:
                continue
            candidates.append(
                {
                    "source": source_name,
                    "family": "source_probe_top_candidates",
                    "rank": int(item.get("rank") or len(candidates) + 1),
                    "score": item.get("score"),
                    "point": point,
                    "details": item.get("details") or {},
                }
            )
    return candidates


def _evaluate_lane(lane_name: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    config = LANE_CONFIGS[lane_name]
    evaluated_rows: List[Dict[str, Any]] = []
    policy_entries: Dict[str, List[Dict[str, Any]]] = {policy: [] for policy in POLICY_ORDER}
    for row in rows:
        policy_results = {
            "source_priority_top1_v0": _evaluate_selection(
                row,
                _select_source_priority(row, config),
                config,
            ),
            "agreement_cluster_v0": _evaluate_selection(
                row,
                _select_agreement_cluster(row, config["agreementClusterV0"]),
                config,
            ),
            "strict_agreement_cluster_v0": _evaluate_selection(
                row,
                _select_agreement_cluster(row, config["strictAgreementClusterV0"]),
                config,
            ),
            "oracle_best_candidate": _evaluate_selection(
                row,
                _oracle_selection(row),
                config,
            ),
        }
        for policy, result in policy_results.items():
            policy_entries[policy].append(result)
        evaluated_rows.append(
            {
                "key": row["key"],
                "candidateCount": row["candidateCount"],
                "oracleBestCandidate": row.get("oracleBestCandidate"),
                "policyResults": policy_results,
            }
        )

    return {
        "config": config,
        "rowCount": len(rows),
        "candidateSummary": _candidate_summary(rows),
        "oracleSummary": _oracle_summary(rows, config),
        "policySummaries": {
            policy: _policy_summary(entries, config)
            for policy, entries in policy_entries.items()
        },
        "rows": evaluated_rows,
    }


def _select_source_priority(row: Dict[str, Any], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = list(row.get("candidates") or [])
    for source in config["sourcePriority"]:
        source_candidates = [c for c in candidates if c.get("source") == source]
        if source_candidates:
            candidate = sorted(
                source_candidates,
                key=lambda c: (
                    int(c.get("rank") or 9999),
                    -float(c.get("score") or 0.0),
                ),
            )[0]
            return {
                "selectionType": "candidate",
                "point": candidate["point"],
                "source": candidate["source"],
                "details": {"reason": "first_available_source_priority", "rank": candidate.get("rank")},
            }
    return None


def _select_agreement_cluster(row: Dict[str, Any], cluster_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = [
        candidate
        for candidate in row.get("candidates", [])
        if _point_or_none(candidate.get("point")) is not None
    ]
    if not candidates:
        return None
    radius = float(cluster_config["radiusPx"])
    min_distinct = int(cluster_config["minDistinctSources"])
    clusters = []
    for seed in candidates:
        seed_point = _point_or_none(seed.get("point"))
        if seed_point is None:
            continue
        members = []
        for candidate in candidates:
            point = _point_or_none(candidate.get("point"))
            if point is None:
                continue
            if _distance(seed_point, point) <= radius:
                members.append(candidate)
        if not members:
            continue
        distinct_sources = sorted({str(member.get("source")) for member in members})
        if len(distinct_sources) < min_distinct:
            continue
        points = [_point_or_none(member.get("point")) for member in members]
        center = _mean_point([point for point in points if point is not None])
        if center is None:
            continue
        avg_rank = _mean([float(member.get("rank") or 9999) for member in members])
        clusters.append(
            {
                "center": center,
                "members": members,
                "distinctSources": distinct_sources,
                "avgRank": avg_rank,
                "maxDistanceToCenterPx": max(
                    _distance(center, _point_or_none(member.get("point")) or center)
                    for member in members
                ),
            }
        )
    if not clusters:
        return None
    clusters.sort(
        key=lambda cluster: (
            -len(cluster["distinctSources"]),
            -len(cluster["members"]),
            cluster["avgRank"],
            cluster["maxDistanceToCenterPx"],
        )
    )
    best = clusters[0]
    return {
        "selectionType": "cluster_center",
        "point": _round_point(best["center"]),
        "source": "agreement_cluster",
        "details": {
            "radiusPx": radius,
            "minDistinctSources": min_distinct,
            "distinctSources": best["distinctSources"],
            "memberCount": len(best["members"]),
            "avgRank": round(float(best["avgRank"]), 4),
            "maxDistanceToCenterPx": round(float(best["maxDistanceToCenterPx"]), 2),
        },
    }


def _oracle_selection(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidate = row.get("oracleBestCandidate")
    if not candidate:
        return None
    return {
        "selectionType": "candidate_oracle",
        "point": candidate.get("point"),
        "source": candidate.get("source"),
        "details": {"reason": "label_oracle", "rank": candidate.get("rank")},
    }


def _evaluate_selection(
    row: Dict[str, Any],
    selection: Optional[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    if selection is None:
        return {"selection": "abstain", "status": "abstained"}
    point = _point_or_none(selection.get("point"))
    truth = _point_or_none(row.get("truth"))
    if point is None or truth is None:
        return {"selection": "abstain", "status": "missing_point"}
    error = _distance(point, truth)
    return {
        "selection": str(selection.get("source") or selection.get("selectionType") or "selected"),
        "selectionType": selection.get("selectionType"),
        "point": _round_point(point),
        "errorPx": round(error, 2),
        "status": _status_from_error(error, float(config["strictPx"]), float(config["plausiblePx"])),
        "details": selection.get("details") or {},
    }


def _candidate_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = [int(row.get("candidateCount") or 0) for row in rows]
    if not counts:
        return {"meanCandidateCount": 0.0, "maxCandidateCount": 0, "minCandidateCount": 0}
    return {
        "meanCandidateCount": round(float(_mean(counts)), 4),
        "maxCandidateCount": max(counts),
        "minCandidateCount": min(counts),
    }


def _oracle_summary(rows: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    strict_px = float(config["strictPx"])
    plausible_px = float(config["plausiblePx"])
    candidates = [row.get("oracleBestCandidate") for row in rows if row.get("oracleBestCandidate")]
    errors = [float(candidate["errorPx"]) for candidate in candidates]
    return {
        "strictReachableCount": sum(1 for error in errors if error <= strict_px),
        "plausibleReachableCount": sum(1 for error in errors if error <= plausible_px),
        "falseReachableCount": sum(1 for error in errors if error > plausible_px),
        "meanBestErrorPx": round(float(_mean(errors)), 4) if errors else None,
        "medianBestErrorPx": round(float(statistics.median(errors)), 4) if errors else None,
    }


def _policy_summary(entries: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    selected = [entry for entry in entries if entry.get("selection") != "abstain" and "errorPx" in entry]
    errors = [float(entry["errorPx"]) for entry in selected]
    strict_px = float(config["strictPx"])
    plausible_px = float(config["plausiblePx"])
    return {
        "selectedCount": len(selected),
        "abstainCount": len(entries) - len(selected),
        "strictReadyCount": sum(1 for error in errors if error <= strict_px),
        "plausibleCount": sum(1 for error in errors if error <= plausible_px),
        "falseConfidentCount": sum(1 for error in errors if error > plausible_px),
        "meanSelectedErrorPx": round(float(_mean(errors)), 4) if errors else None,
        "medianSelectedErrorPx": round(float(statistics.median(errors)), 4) if errors else None,
    }


def _overall_conclusion(lane_results: Dict[str, Any]) -> Dict[str, str]:
    false_confident_policies = []
    low_coverage_zero_fp_policies = []
    for lane_name, lane in lane_results.items():
        for policy in ("agreement_cluster_v0", "strict_agreement_cluster_v0"):
            summary = lane["policySummaries"][policy]
            if summary["falseConfidentCount"] > 0:
                false_confident_policies.append(f"{lane_name}/{policy}")
            elif summary["selectedCount"] < max(2, lane["rowCount"] // 4):
                low_coverage_zero_fp_policies.append(f"{lane_name}/{policy}")
    if false_confident_policies:
        return {
            "productionWiringRecommendation": "wait",
            "reason": (
                "Agreement-based deployable policies still make false-confident "
                "vertex selections: " + ", ".join(false_confident_policies) + "."
            ),
        }
    if low_coverage_zero_fp_policies:
        return {
            "productionWiringRecommendation": "wait",
            "reason": (
                "Zero-false-positive policies select too little data to be a "
                "usable recognizer path: " + ", ".join(low_coverage_zero_fp_policies) + "."
            ),
        }
    return {
        "productionWiringRecommendation": "investigate",
        "reason": "Agreement policies did not show an obvious false-confident failure in this fixture.",
    }


def _selection_report_text(selection: Dict[str, Any]) -> str:
    if selection.get("selection") == "abstain":
        return "`abstain`"
    return f"`{selection.get('selection')}` / {selection.get('errorPx')} px / `{selection.get('status')}`"


def _nearest_candidate(candidates: Sequence[Dict[str, Any]], truth: Point) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    candidate = min(candidates, key=lambda item: _distance(_point_or_none(item.get("point")) or truth, truth))
    return {
        "source": candidate.get("source"),
        "family": candidate.get("family"),
        "rank": candidate.get("rank"),
        "point": candidate.get("point"),
        "errorPx": candidate.get("errorPx"),
        "status": candidate.get("status"),
    }


def _weighted_by_metric(
    a: Point,
    b: Point,
    metric_a: Optional[float],
    metric_b: Optional[float],
    *,
    inverse: bool,
) -> Optional[Point]:
    if metric_a is None or metric_b is None:
        return None
    if inverse:
        if metric_a <= 0.0 or metric_b <= 0.0:
            return None
        weight_a = 1.0 / metric_a
        weight_b = 1.0 / metric_b
    else:
        weight_a = max(0.0, metric_a)
        weight_b = max(0.0, metric_b)
    if weight_a + weight_b <= 0.0:
        return None
    return _weighted_point([(a, weight_a), (b, weight_b)])


def _weighted_point(points: Sequence[Tuple[Point, float]]) -> Point:
    total = sum(weight for _, weight in points)
    if total <= 0.0:
        return _mean_point([point for point, _ in points]) or (0.0, 0.0)
    return (
        sum(point[0] * weight for point, weight in points) / total,
        sum(point[1] * weight for point, weight in points) / total,
    )


def _mean_point(points: Sequence[Point]) -> Optional[Point]:
    if not points:
        return None
    return (
        _mean([point[0] for point in points]),
        _mean([point[1] for point in points]),
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _status_from_error(error: float, strict_px: float, plausible_px: float) -> str:
    if error <= strict_px:
        return "strict_ready"
    if error <= plausible_px:
        return "plausible"
    return "false_confident"


def _point_or_none(value: Any) -> Optional[Point]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _round_point(point: Point) -> List[float]:
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def _distance(a: Point, b: Point) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-fixture", type=Path, default=DEFAULT_GEOMETRY_FIXTURE)
    parser.add_argument("--source-probe-fixture", type=Path, default=DEFAULT_SOURCE_PROBE_FIXTURE)
    parser.add_argument("--feedback-out", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    feedback, summary = generate_vertex_hypothesis_ensemble(
        geometry_fixture_path=args.geometry_fixture,
        source_probe_fixture_path=args.source_probe_fixture,
    )
    _write_json(args.feedback_out, feedback)
    _write_json(args.summary_out, summary)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(summary), encoding="utf-8")
    print(f"wrote {args.feedback_out}")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.report_out}")
    print(summary["conclusion"]["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
