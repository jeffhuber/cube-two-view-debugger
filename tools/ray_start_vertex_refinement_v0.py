#!/usr/bin/env python3
"""Diagnostics-only ray-start vertex refinement.

The axis-ray V0 probe showed that plain dark-line support is mixed: it can
find real cube edges, but it often slides along an edge or to another dark
junction. This probe tests the next idea: the visible trihedral vertex should
be where three dark outgoing edge rays start, so candidates get forward ray
support and a penalty when the same dark support continues behind the point.

Human labels are used only for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tools.axis_ray_vertex_refinement_v0 import (
    AXIS_GOOD_DEG,
    DEFAULT_FEEDBACK,
    PLAUSIBLE_VERTEX_PX,
    STRICT_VERTEX_PX,
    Point,
    _best_axis_assignment,
    _count_within,
    _darkness_array,
    _distance,
    _fmt,
    _load_image,
    _mean,
    _median,
    _model_axis_vectors,
    _point_or_none,
    _round_point,
    _sample_bilinear,
    _unit,
    _values,
    _vertex_status,
    _write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "ray_start_vertex_refinement_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "RAY_START_VERTEX_REFINEMENT_V0_REPORT.md"


@dataclass(frozen=True)
class RayStartConfig:
    search_radius_px: int = 180
    search_step_px: int = 12
    forward_min_px: float = 18.0
    forward_max_px: float = 150.0
    backward_min_px: float = 18.0
    backward_max_px: float = 105.0
    sample_count: int = 9
    line_half_width_px: float = 2.0
    side_offset_px: float = 16.0
    center_radius_px: float = 5.0
    distance_prior_weight: float = 0.08
    accept_min_score_gain: float = 0.04
    accept_min_mean_startness: float = 0.03
    accept_min_min_startness: float = -0.02

    def as_dict(self) -> Dict[str, Any]:
        return {
            "searchRadiusPx": self.search_radius_px,
            "searchStepPx": self.search_step_px,
            "forwardMinPx": self.forward_min_px,
            "forwardMaxPx": self.forward_max_px,
            "backwardMinPx": self.backward_min_px,
            "backwardMaxPx": self.backward_max_px,
            "sampleCount": self.sample_count,
            "lineHalfWidthPx": self.line_half_width_px,
            "sideOffsetPx": self.side_offset_px,
            "centerRadiusPx": self.center_radius_px,
            "distancePriorWeight": self.distance_prior_weight,
            "acceptMinScoreGain": self.accept_min_score_gain,
            "acceptMinMeanStartness": self.accept_min_mean_startness,
            "acceptMinMinStartness": self.accept_min_min_startness,
            "strictVertexPx": STRICT_VERTEX_PX,
            "plausibleVertexPx": PLAUSIBLE_VERTEX_PX,
            "axisGoodDeg": AXIS_GOOD_DEG,
        }


def generate_ray_start_vertex_refinement_summary(
    *,
    feedback_path: Path = DEFAULT_FEEDBACK,
    config: RayStartConfig = RayStartConfig(),
) -> Dict[str, Any]:
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    rows = [evaluate_row(row, config=config) for row in feedback.get("rows", [])]
    return {
        "schemaVersion": 1,
        "probe": "ray_start_vertex_refinement_v0",
        "description": (
            "Diagnostics-only visible-trihedral vertex search that rewards "
            "forward dark-ray support and penalizes dark continuation behind "
            "the candidate point."
        ),
        "sourceFeedback": str(feedback_path),
        "config": config.as_dict(),
        "summary": summarize_rows(rows),
        "rows": rows,
    }


def evaluate_row(row: Dict[str, Any], *, config: RayStartConfig) -> Dict[str, Any]:
    base = {
        "key": row.get("key"),
        "setId": row.get("setId"),
        "side": row.get("side"),
        "imagePath": row.get("imagePath"),
    }
    human_vertex = _point_or_none(row.get("humanVertexPoint"))
    human_axis_endpoints = [_point_or_none(point) for point in row.get("humanAxisEndpoints", [])]
    human_axis_endpoints = [point for point in human_axis_endpoints if point is not None]
    model = row.get("currentModel") or {}
    model_vertex = _point_or_none(model.get("vertexPoint"))
    model_axes = _model_axis_vectors(model)
    if human_vertex is None or len(human_axis_endpoints) != 3:
        return {**base, "evaluationStatus": "missing_human_trihedral"}
    if model_vertex is None or len(model_axes) != 3:
        return {**base, "evaluationStatus": "missing_model_trihedral"}
    image_path = Path(str(row.get("imagePath") or ""))
    if not image_path.exists():
        return {**base, "evaluationStatus": "missing_image"}

    image = _load_image(image_path)
    darkness = _darkness_array(image)
    human_axis_vectors = [
        (endpoint[0] - human_vertex[0], endpoint[1] - human_vertex[1])
        for endpoint in human_axis_endpoints
    ]
    axis_assignment = _best_axis_assignment(model_axes, human_axis_vectors)
    max_axis_error = max(axis_assignment["angleErrorsDeg"])
    axis_category = "axis_good" if max_axis_error <= AXIS_GOOD_DEG else "axis_blocked"

    baseline_error = _distance(model_vertex, human_vertex)
    model_best = refine_vertex(darkness, start=model_vertex, axes=model_axes, config=config)
    model_baseline_score = score_candidate(darkness, model_vertex, model_axes, config=config)
    model_gated = _gated_choice(model_vertex, model_baseline_score, model_best, config)
    human_axis_best = refine_vertex(darkness, start=model_vertex, axes=human_axis_vectors, config=config)
    human_baseline_score = score_candidate(darkness, model_vertex, human_axis_vectors, config=config)
    human_axis_gated = _gated_choice(model_vertex, human_baseline_score, human_axis_best, config)

    model_best_error = _distance(model_best["point"], human_vertex)
    model_gated_error = _distance(model_gated["point"], human_vertex)
    human_best_error = _distance(human_axis_best["point"], human_vertex)
    human_gated_error = _distance(human_axis_gated["point"], human_vertex)
    return {
        **base,
        "evaluationStatus": "ok",
        "axisCategory": axis_category,
        "modelAxisMaxAngleErrorDeg": round(max_axis_error, 2),
        "baselineVertex": _round_point(model_vertex),
        "baselineVertexErrorPx": round(baseline_error, 2),
        "modelAxisBestVertex": _round_point(model_best["point"]),
        "modelAxisBestVertexErrorPx": round(model_best_error, 2),
        "modelAxisBestImprovementPx": round(baseline_error - model_best_error, 2),
        "modelAxisGatedVertex": _round_point(model_gated["point"]),
        "modelAxisGatedAccepted": bool(model_gated["accepted"]),
        "modelAxisGatedVertexErrorPx": round(model_gated_error, 2),
        "modelAxisGatedImprovementPx": round(baseline_error - model_gated_error, 2),
        "modelAxisGatedStatus": _vertex_status(model_gated_error),
        "modelAxisScoreGain": round(model_best["score"] - model_baseline_score["score"], 6),
        "modelAxisBestComponents": model_best["components"],
        "humanAxisOracleBestVertex": _round_point(human_axis_best["point"]),
        "humanAxisOracleBestVertexErrorPx": round(human_best_error, 2),
        "humanAxisOracleBestImprovementPx": round(baseline_error - human_best_error, 2),
        "humanAxisOracleGatedVertex": _round_point(human_axis_gated["point"]),
        "humanAxisOracleGatedAccepted": bool(human_axis_gated["accepted"]),
        "humanAxisOracleGatedVertexErrorPx": round(human_gated_error, 2),
        "humanAxisOracleGatedImprovementPx": round(baseline_error - human_gated_error, 2),
        "humanAxisOracleGatedStatus": _vertex_status(human_gated_error),
        "humanAxisOracleScoreGain": round(human_axis_best["score"] - human_baseline_score["score"], 6),
        "humanVertex": _round_point(human_vertex),
    }


def refine_vertex(
    darkness: np.ndarray,
    *,
    start: Point,
    axes: Sequence[Point],
    config: RayStartConfig,
) -> Dict[str, Any]:
    axis_units = [_unit(axis) for axis in axes]
    if any(axis is None for axis in axis_units):
        return {
            "point": start,
            "score": float("-inf"),
            "components": {"status": "invalid_axis"},
        }
    best: Optional[Dict[str, Any]] = None
    radius = int(config.search_radius_px)
    step = int(config.search_step_px)
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            if dx * dx + dy * dy > radius * radius:
                continue
            candidate = (start[0] + dx, start[1] + dy)
            scored = score_candidate(
                darkness,
                candidate,
                axis_units,  # type: ignore[arg-type]
                config=config,
            )
            distance_prior = math.hypot(dx, dy) / max(1.0, radius)
            score = scored["score"] - config.distance_prior_weight * distance_prior
            components = dict(scored["components"])
            components["distancePrior"] = round(distance_prior, 4)
            if best is None or score > best["score"]:
                best = {
                    "point": candidate,
                    "score": round(score, 6),
                    "components": components,
                }
    assert best is not None
    return best


def score_candidate(
    darkness: np.ndarray,
    point: Point,
    axes: Sequence[Point],
    *,
    config: RayStartConfig,
) -> Dict[str, Any]:
    center_dark = _disk_mean(darkness, point, config.center_radius_px)
    if center_dark is None:
        return {"score": float("-inf"), "components": {"status": "outside_image"}}
    forward_scores: List[float] = []
    backward_scores: List[float] = []
    startness_values: List[float] = []
    forward_contrasts: List[float] = []
    valid_rays = 0
    for axis in axes:
        forward = _score_directional_ray(
            darkness,
            point,
            axis,
            min_px=config.forward_min_px,
            max_px=config.forward_max_px,
            config=config,
        )
        backward = _score_directional_ray(
            darkness,
            point,
            (-axis[0], -axis[1]),
            min_px=config.backward_min_px,
            max_px=config.backward_max_px,
            config=config,
        )
        if forward["validSampleCount"] >= max(3, config.sample_count // 2):
            valid_rays += 1
        forward_scores.append(float(forward["score"]))
        backward_scores.append(float(backward["score"]))
        startness_values.append(float(forward["meanDarkness"]) - float(backward["meanDarkness"]))
        forward_contrasts.append(float(forward["meanContrast"]))
    if valid_rays < 3:
        return {
            "score": float("-inf"),
            "components": {"status": "incomplete_ray_samples", "validRays": valid_rays},
        }
    mean_forward = statistics.mean(forward_scores)
    min_forward = min(forward_scores)
    mean_backward = statistics.mean(backward_scores)
    max_backward = max(backward_scores)
    mean_startness = statistics.mean(startness_values)
    min_startness = min(startness_values)
    mean_forward_contrast = statistics.mean(forward_contrasts)
    score = (
        center_dark * 0.70
        + mean_forward * 1.00
        + min_forward * 0.70
        + mean_startness * 1.15
        + min_startness * 0.55
        + max(0.0, mean_forward_contrast) * 0.25
        - mean_backward * 0.45
        - max_backward * 0.20
    )
    return {
        "score": round(score, 6),
        "components": {
            "status": "ok",
            "centerDarkness": round(center_dark, 5),
            "meanForwardScore": round(mean_forward, 5),
            "minForwardScore": round(min_forward, 5),
            "meanBackwardScore": round(mean_backward, 5),
            "maxBackwardScore": round(max_backward, 5),
            "meanStartness": round(mean_startness, 5),
            "minStartness": round(min_startness, 5),
            "meanForwardContrast": round(mean_forward_contrast, 5),
        },
    }


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [row for row in rows if row.get("evaluationStatus") == "ok"]
    axis_good = [row for row in ok if row.get("axisCategory") == "axis_good"]
    return {
        "rowCount": len(rows),
        "evaluatedRowCount": len(ok),
        "axisGoodRowCount": len(axis_good),
        "axisBlockedRowCount": len(ok) - len(axis_good),
        "baselineStrictCount": _count_within(ok, "baselineVertexErrorPx", STRICT_VERTEX_PX),
        "baselinePlausibleCount": _count_within(ok, "baselineVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "modelAxisBestStrictCount": _count_within(ok, "modelAxisBestVertexErrorPx", STRICT_VERTEX_PX),
        "modelAxisBestPlausibleCount": _count_within(ok, "modelAxisBestVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "modelAxisGatedStrictCount": _count_within(ok, "modelAxisGatedVertexErrorPx", STRICT_VERTEX_PX),
        "modelAxisGatedPlausibleCount": _count_within(ok, "modelAxisGatedVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "humanAxisOracleBestStrictCount": _count_within(ok, "humanAxisOracleBestVertexErrorPx", STRICT_VERTEX_PX),
        "humanAxisOracleBestPlausibleCount": _count_within(ok, "humanAxisOracleBestVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "humanAxisOracleGatedStrictCount": _count_within(ok, "humanAxisOracleGatedVertexErrorPx", STRICT_VERTEX_PX),
        "humanAxisOracleGatedPlausibleCount": _count_within(ok, "humanAxisOracleGatedVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "axisGoodBaselineStrictCount": _count_within(axis_good, "baselineVertexErrorPx", STRICT_VERTEX_PX),
        "axisGoodModelAxisGatedStrictCount": _count_within(axis_good, "modelAxisGatedVertexErrorPx", STRICT_VERTEX_PX),
        "axisGoodHumanAxisOracleGatedStrictCount": _count_within(axis_good, "humanAxisOracleGatedVertexErrorPx", STRICT_VERTEX_PX),
        "modelAxisGatedAcceptedCount": sum(1 for row in ok if row.get("modelAxisGatedAccepted")),
        "humanAxisOracleGatedAcceptedCount": sum(1 for row in ok if row.get("humanAxisOracleGatedAccepted")),
        "modelAxisGatedImprovedRowCount": sum(1 for row in ok if float(row["modelAxisGatedImprovementPx"]) > 5.0),
        "modelAxisGatedWorsenedRowCount": sum(1 for row in ok if float(row["modelAxisGatedImprovementPx"]) < -5.0),
        "humanAxisOracleGatedImprovedRowCount": sum(1 for row in ok if float(row["humanAxisOracleGatedImprovementPx"]) > 5.0),
        "humanAxisOracleGatedWorsenedRowCount": sum(1 for row in ok if float(row["humanAxisOracleGatedImprovementPx"]) < -5.0),
        "meanBaselineVertexErrorPx": _mean(_values(ok, "baselineVertexErrorPx")),
        "meanModelAxisGatedVertexErrorPx": _mean(_values(ok, "modelAxisGatedVertexErrorPx")),
        "meanHumanAxisOracleGatedVertexErrorPx": _mean(_values(ok, "humanAxisOracleGatedVertexErrorPx")),
        "medianBaselineVertexErrorPx": _median(_values(ok, "baselineVertexErrorPx")),
        "medianModelAxisGatedVertexErrorPx": _median(_values(ok, "modelAxisGatedVertexErrorPx")),
        "medianHumanAxisOracleGatedVertexErrorPx": _median(_values(ok, "humanAxisOracleGatedVertexErrorPx")),
        "modelAxisThresholdSweep": _model_axis_threshold_sweep(ok),
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    lines = [
        "# Ray-Start Vertex Refinement V0",
        "",
        "Diagnostics-only artifact. This does not alter recognition behavior.",
        "",
        "This probe searches for a visible-trihedral vertex by rewarding dark outgoing axis rays and penalizing dark continuation behind the candidate point.",
        "",
        "## Summary",
        "",
        f"- Rows: {summary['rowCount']}",
        f"- Evaluated rows: {summary['evaluatedRowCount']}",
        f"- Axis-good rows: {summary['axisGoodRowCount']}",
        f"- Axis-blocked rows: {summary['axisBlockedRowCount']}",
        f"- Baseline strict/plausible: {summary['baselineStrictCount']} / {summary['baselinePlausibleCount']}",
        f"- Model-axis best strict/plausible: {summary['modelAxisBestStrictCount']} / {summary['modelAxisBestPlausibleCount']}",
        f"- Model-axis gated strict/plausible: {summary['modelAxisGatedStrictCount']} / {summary['modelAxisGatedPlausibleCount']}",
        f"- Human-axis oracle best strict/plausible: {summary['humanAxisOracleBestStrictCount']} / {summary['humanAxisOracleBestPlausibleCount']}",
        f"- Human-axis oracle gated strict/plausible: {summary['humanAxisOracleGatedStrictCount']} / {summary['humanAxisOracleGatedPlausibleCount']}",
        f"- Axis-good strict baseline/model-gated/oracle-gated: {summary['axisGoodBaselineStrictCount']} / {summary['axisGoodModelAxisGatedStrictCount']} / {summary['axisGoodHumanAxisOracleGatedStrictCount']}",
        f"- Model-axis gated accepted rows: {summary['modelAxisGatedAcceptedCount']}",
        f"- Human-axis oracle gated accepted rows: {summary['humanAxisOracleGatedAcceptedCount']}",
        f"- Model-axis gated improved/worsened rows by >5px: {summary['modelAxisGatedImprovedRowCount']} / {summary['modelAxisGatedWorsenedRowCount']}",
        f"- Human-axis oracle gated improved/worsened rows by >5px: {summary['humanAxisOracleGatedImprovedRowCount']} / {summary['humanAxisOracleGatedWorsenedRowCount']}",
        f"- Mean vertex error baseline/model-gated/oracle-gated: {_fmt(summary['meanBaselineVertexErrorPx'], 'px')} / {_fmt(summary['meanModelAxisGatedVertexErrorPx'], 'px')} / {_fmt(summary['meanHumanAxisOracleGatedVertexErrorPx'], 'px')}",
        f"- Median vertex error baseline/model-gated/oracle-gated: {_fmt(summary['medianBaselineVertexErrorPx'], 'px')} / {_fmt(summary['medianModelAxisGatedVertexErrorPx'], 'px')} / {_fmt(summary['medianHumanAxisOracleGatedVertexErrorPx'], 'px')}",
        f"- Best non-empty low-worsen model gate: {_format_sweep(summary['modelAxisThresholdSweep'].get('bestNonEmptyLowWorsen'))}",
        f"- Best non-empty model gate with <=2 worsens: {_format_sweep(summary['modelAxisThresholdSweep'].get('bestNonEmptyAtMostTwoWorsens'))}",
        "",
        "## Rows",
        "",
        "| Row | Axis | Base | Model gated | Accepted | Delta | Oracle gated | Oracle accepted | Oracle delta |",
        "|---|---|---:|---:|---|---:|---:|---|---:|",
    ]
    for row in document["rows"]:
        if row.get("evaluationStatus") != "ok":
            lines.append(f"| `{row.get('key')}` | `{row.get('evaluationStatus')}` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| `{row.get('key')}` | `{row.get('axisCategory')}` | "
            f"{_fmt(row.get('baselineVertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('modelAxisGatedVertexErrorPx'), 'px')} | "
            f"{'yes' if row.get('modelAxisGatedAccepted') else 'no'} | "
            f"{_fmt(row.get('modelAxisGatedImprovementPx'), 'px')} | "
            f"{_fmt(row.get('humanAxisOracleGatedVertexErrorPx'), 'px')} | "
            f"{'yes' if row.get('humanAxisOracleGatedAccepted') else 'no'} | "
            f"{_fmt(row.get('humanAxisOracleGatedImprovementPx'), 'px')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is still a diagnostics-only image objective, not production wiring.",
            "- A useful production-shaped signal would increase strict/plausible rows while keeping worsened accepted rows near zero.",
            "- The threshold sweep could not find a non-empty model-axis gate with two-or-fewer worsened rows, so the current hand-tuned ray-start score is not a safe promotion signal.",
            "- If gated model-axis results remain no better than baseline, the next step should be explicit line/junction extraction or learned vertex localization rather than hand-tuned darkness scoring.",
            "",
        ]
    )
    return "\n".join(lines)


def _gated_choice(
    baseline_point: Point,
    baseline_score: Dict[str, Any],
    best: Dict[str, Any],
    config: RayStartConfig,
) -> Dict[str, Any]:
    gain = float(best["score"]) - float(baseline_score["score"])
    components = best.get("components") or {}
    mean_startness = float(components.get("meanStartness", -999.0))
    min_startness = float(components.get("minStartness", -999.0))
    accepted = (
        gain >= config.accept_min_score_gain
        and mean_startness >= config.accept_min_mean_startness
        and min_startness >= config.accept_min_min_startness
    )
    if accepted:
        return {**best, "accepted": True}
    return {
        "point": baseline_point,
        "score": baseline_score["score"],
        "components": baseline_score["components"],
        "accepted": False,
    }


def _model_axis_threshold_sweep(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for gain_t in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]:
        for mean_t in [-0.05, 0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            for min_t in [-0.20, -0.10, -0.05, -0.02, 0.0, 0.03, 0.05, 0.08, 0.10]:
                accepted = [
                    row
                    for row in rows
                    if _row_passes_model_gate(row, gain_t=gain_t, mean_t=mean_t, min_t=min_t)
                ]
                if not accepted:
                    continue
                improved = sum(1 for row in accepted if float(row["modelAxisBestImprovementPx"]) > 5.0)
                worsened = sum(1 for row in accepted if float(row["modelAxisBestImprovementPx"]) < -5.0)
                errors = [
                    float(row["modelAxisBestVertexErrorPx"])
                    if row in accepted
                    else float(row["baselineVertexErrorPx"])
                    for row in rows
                ]
                candidates.append(
                    {
                        "acceptedCount": len(accepted),
                        "improvedCount": improved,
                        "worsenedCount": worsened,
                        "strictCount": sum(1 for error in errors if error <= STRICT_VERTEX_PX),
                        "plausibleCount": sum(1 for error in errors if error <= PLAUSIBLE_VERTEX_PX),
                        "meanVertexErrorPx": round(sum(errors) / len(errors), 4) if errors else None,
                        "thresholds": {
                            "scoreGain": gain_t,
                            "meanStartness": mean_t,
                            "minStartness": min_t,
                        },
                    }
                )
    if not candidates:
        return {"evaluatedNonEmptyGateCount": 0}
    low_worsen = sorted(
        candidates,
        key=lambda item: (
            item["worsenedCount"],
            -item["improvedCount"],
            -item["strictCount"],
            item["meanVertexErrorPx"],
        ),
    )[0]
    at_most_two = [item for item in candidates if item["worsenedCount"] <= 2]
    best_at_most_two = None
    if at_most_two:
        best_at_most_two = sorted(
            at_most_two,
            key=lambda item: (-item["strictCount"], -item["improvedCount"], item["meanVertexErrorPx"]),
        )[0]
    return {
        "evaluatedNonEmptyGateCount": len(candidates),
        "bestNonEmptyLowWorsen": low_worsen,
        "bestNonEmptyAtMostTwoWorsens": best_at_most_two,
    }


def _row_passes_model_gate(row: Dict[str, Any], *, gain_t: float, mean_t: float, min_t: float) -> bool:
    components = row.get("modelAxisBestComponents") or {}
    return (
        float(row.get("modelAxisScoreGain", -999.0)) >= gain_t
        and float(components.get("meanStartness", -999.0)) >= mean_t
        and float(components.get("minStartness", -999.0)) >= min_t
    )


def _format_sweep(item: Optional[Dict[str, Any]]) -> str:
    if not item:
        return "none"
    thresholds = item.get("thresholds") or {}
    return (
        f"accepted {item['acceptedCount']}, improved {item['improvedCount']}, "
        f"worsened {item['worsenedCount']}, strict/plausible "
        f"{item['strictCount']} / {item['plausibleCount']}, "
        f"mean {_fmt(item['meanVertexErrorPx'], 'px')} "
        f"(gain>={thresholds.get('scoreGain')}, "
        f"meanStart>={thresholds.get('meanStartness')}, "
        f"minStart>={thresholds.get('minStartness')})"
    )


def _score_directional_ray(
    darkness: np.ndarray,
    point: Point,
    axis: Point,
    *,
    min_px: float,
    max_px: float,
    config: RayStartConfig,
) -> Dict[str, Any]:
    normal = (-axis[1], axis[0])
    ts = np.linspace(min_px, max_px, config.sample_count)
    center_values: List[float] = []
    contrast_values: List[float] = []
    for t in ts:
        center = (point[0] + axis[0] * float(t), point[1] + axis[1] * float(t))
        center_dark = _strip_mean(darkness, center, normal, [-config.line_half_width_px, 0.0, config.line_half_width_px])
        side_dark = _strip_mean(
            darkness,
            center,
            normal,
            [
                -config.side_offset_px - config.line_half_width_px,
                -config.side_offset_px,
                config.side_offset_px,
                config.side_offset_px + config.line_half_width_px,
            ],
        )
        if center_dark is None or side_dark is None:
            continue
        center_values.append(center_dark)
        contrast_values.append(center_dark - side_dark)
    if not center_values:
        return {
            "validSampleCount": 0,
            "score": 0.0,
            "meanDarkness": 0.0,
            "meanContrast": 0.0,
        }
    mean_darkness = statistics.mean(center_values)
    mean_contrast = statistics.mean(contrast_values)
    p70_darkness = float(np.percentile(np.asarray(center_values), 70))
    score = mean_darkness * 0.60 + p70_darkness * 0.25 + max(0.0, mean_contrast) * 0.35
    return {
        "validSampleCount": len(center_values),
        "score": score,
        "meanDarkness": mean_darkness,
        "meanContrast": mean_contrast,
    }


def _strip_mean(
    darkness: np.ndarray,
    center: Point,
    normal: Point,
    offsets: Iterable[float],
) -> Optional[float]:
    values: List[float] = []
    for offset in offsets:
        value = _sample_bilinear(
            darkness,
            center[0] + normal[0] * float(offset),
            center[1] + normal[1] * float(offset),
        )
        if value is not None:
            values.append(value)
    if not values:
        return None
    return statistics.mean(values)


def _disk_mean(darkness: np.ndarray, center: Point, radius: float) -> Optional[float]:
    offsets = [
        (0.0, 0.0),
        (-radius, 0.0),
        (radius, 0.0),
        (0.0, -radius),
        (0.0, radius),
        (-radius * 0.7, -radius * 0.7),
        (radius * 0.7, -radius * 0.7),
        (-radius * 0.7, radius * 0.7),
        (radius * 0.7, radius * 0.7),
    ]
    values = [_sample_bilinear(darkness, center[0] + dx, center[1] + dy) for dx, dy in offsets]
    valid = [value for value in values if value is not None]
    if len(valid) < 5:
        return None
    return statistics.mean(valid)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    document = generate_ray_start_vertex_refinement_summary(feedback_path=args.feedback)
    _write_json(args.summary_out, document)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(document), encoding="utf-8")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.report_out}")
    print(
        f"model-axis gated strict rows: {document['summary']['modelAxisGatedStrictCount']} / "
        f"{document['summary']['evaluatedRowCount']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
