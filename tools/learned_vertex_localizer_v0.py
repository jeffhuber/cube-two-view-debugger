#!/usr/bin/env python3
"""Diagnostics-only learned visible-vertex localization.

The hand-tuned dark-ray probes can find some real cube-edge structure, but the
same objectives also move good vertices to nearby sticker-grid junctions. This
probe tests the next idea in the smallest possible form: learn a local scoring
function from the human-labeled visible-trihedral rows, then evaluate it with
leave-one-row-out validation.

Human labels are used only for training/evaluation inside this diagnostics
artifact. Nothing here alters recognizer behavior.
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
    _unit,
    _values,
    _vertex_status,
    _write_json,
)
from tools.ray_start_vertex_refinement_v0 import RayStartConfig, score_candidate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "learned_vertex_localizer_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "LEARNED_VERTEX_LOCALIZER_V0_REPORT.md"


@dataclass(frozen=True)
class LearnedVertexConfig:
    search_radius_px: int = 220
    search_step_px: int = 16
    target_scale_px: float = 55.0
    ridge_lambda: float = 0.75
    default_gate_min_score_gain: float = 0.04
    default_gate_min_predicted_score: float = 0.45
    ray_config: RayStartConfig = RayStartConfig(
        search_radius_px=220,
        search_step_px=16,
        forward_min_px=18.0,
        forward_max_px=150.0,
        backward_min_px=18.0,
        backward_max_px=105.0,
        sample_count=9,
        distance_prior_weight=0.08,
    )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "searchRadiusPx": self.search_radius_px,
            "searchStepPx": self.search_step_px,
            "targetScalePx": self.target_scale_px,
            "ridgeLambda": self.ridge_lambda,
            "defaultGateMinScoreGain": self.default_gate_min_score_gain,
            "defaultGateMinPredictedScore": self.default_gate_min_predicted_score,
            "strictVertexPx": STRICT_VERTEX_PX,
            "plausibleVertexPx": PLAUSIBLE_VERTEX_PX,
            "axisGoodDeg": AXIS_GOOD_DEG,
            "featureNames": FEATURE_NAMES,
        }


FEATURE_NAMES = [
    "bias",
    "dx_norm",
    "dy_norm",
    "radius_norm",
    "radius_norm_sq",
    "ray_score",
    "center_darkness",
    "mean_forward_score",
    "min_forward_score",
    "mean_backward_score",
    "max_backward_score",
    "mean_startness",
    "min_startness",
    "mean_forward_contrast",
    "fit_quality",
    "fit_residual_norm",
    "ray_score_x_startness",
    "ray_score_x_center_darkness",
]


def generate_learned_vertex_localizer_summary(
    *,
    feedback_path: Path = DEFAULT_FEEDBACK,
    config: LearnedVertexConfig = LearnedVertexConfig(),
) -> Dict[str, Any]:
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    prepared_rows = [
        prepare_row(row, config=config)
        for row in feedback.get("rows", [])
    ]
    ok_rows = [row for row in prepared_rows if row.get("evaluationStatus") == "ok"]
    predictions = _leave_one_out_predictions(ok_rows, config=config)
    prediction_by_key = {row["key"]: row for row in predictions}
    rows = [
        {**row, **prediction_by_key.get(row.get("key"), {})}
        if row.get("evaluationStatus") == "ok"
        else row
        for row in prepared_rows
    ]
    return {
        "schemaVersion": 1,
        "probe": "learned_vertex_localizer_v0",
        "description": (
            "Diagnostics-only leave-one-row-out learned scoring function for "
            "visible trihedral vertex candidates."
        ),
        "sourceFeedback": str(feedback_path),
        "config": config.as_dict(),
        "summary": summarize_rows(rows),
        "rows": rows,
    }


def prepare_row(row: Dict[str, Any], *, config: LearnedVertexConfig) -> Dict[str, Any]:
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

    candidate_rows = _candidate_rows(
        darkness=darkness,
        model_vertex=model_vertex,
        model_axes=model_axes,
        human_vertex=human_vertex,
        model=model,
        config=config,
    )
    if not candidate_rows:
        return {**base, "evaluationStatus": "missing_candidate_features"}

    baseline_error = _distance(model_vertex, human_vertex)
    oracle = min(candidate_rows, key=lambda item: item["targetErrorPx"])
    return {
        **base,
        "evaluationStatus": "ok",
        "axisCategory": axis_category,
        "modelAxisMaxAngleErrorDeg": round(max_axis_error, 2),
        "baselineVertex": _round_point(model_vertex),
        "baselineVertexErrorPx": round(baseline_error, 2),
        "candidateCount": len(candidate_rows),
        "candidateOracleVertex": _round_point(oracle["point"]),
        "candidateOracleVertexErrorPx": round(float(oracle["targetErrorPx"]), 2),
        "candidateOracleImprovementPx": round(baseline_error - float(oracle["targetErrorPx"]), 2),
        "candidateOracleStatus": _vertex_status(float(oracle["targetErrorPx"])),
        "humanVertex": _round_point(human_vertex),
        "_candidateRows": candidate_rows,
    }


def _candidate_rows(
    *,
    darkness: np.ndarray,
    model_vertex: Point,
    model_axes: Sequence[Point],
    human_vertex: Point,
    model: Dict[str, Any],
    config: LearnedVertexConfig,
) -> List[Dict[str, Any]]:
    axis_units = [_unit(axis) for axis in model_axes]
    if any(axis is None for axis in axis_units):
        return []
    units = [axis for axis in axis_units if axis is not None]
    candidates: Dict[Tuple[int, int], Point] = {
        (int(round(model_vertex[0] * 10)), int(round(model_vertex[1] * 10))): model_vertex
    }
    radius = config.search_radius_px
    step = config.search_step_px
    for dy in range(-radius, radius + 1, step):
        for dx in range(-radius, radius + 1, step):
            if dx * dx + dy * dy > radius * radius:
                continue
            point = (model_vertex[0] + float(dx), model_vertex[1] + float(dy))
            candidates[(int(round(point[0] * 10)), int(round(point[1] * 10)))] = point

    fit_quality = float(model.get("fitQuality") or 0.0)
    fit_residual = float((model.get("debug") or {}).get("fitResidualRmsPx") or 0.0)
    rows: List[Dict[str, Any]] = []
    for point in candidates.values():
        scored = score_candidate(darkness, point, units, config=config.ray_config)
        score = float(scored.get("score", float("-inf")))
        components = scored.get("components") or {}
        if not math.isfinite(score) or components.get("status") != "ok":
            continue
        features = _feature_vector(
            point=point,
            model_vertex=model_vertex,
            ray_score=score,
            components=components,
            fit_quality=fit_quality,
            fit_residual=fit_residual,
            config=config,
        )
        error = _distance(point, human_vertex)
        rows.append(
            {
                "point": point,
                "features": features,
                "targetErrorPx": error,
                "targetScore": math.exp(-error / max(1.0, config.target_scale_px)),
                "rayScore": score,
                "components": components,
            }
        )
    return rows


def _feature_vector(
    *,
    point: Point,
    model_vertex: Point,
    ray_score: float,
    components: Dict[str, Any],
    fit_quality: float,
    fit_residual: float,
    config: LearnedVertexConfig,
) -> List[float]:
    dx = (point[0] - model_vertex[0]) / max(1.0, config.search_radius_px)
    dy = (point[1] - model_vertex[1]) / max(1.0, config.search_radius_px)
    radius = math.hypot(dx, dy)
    center = float(components.get("centerDarkness", 0.0))
    mean_forward = float(components.get("meanForwardScore", 0.0))
    min_forward = float(components.get("minForwardScore", 0.0))
    mean_backward = float(components.get("meanBackwardScore", 0.0))
    max_backward = float(components.get("maxBackwardScore", 0.0))
    mean_startness = float(components.get("meanStartness", 0.0))
    min_startness = float(components.get("minStartness", 0.0))
    mean_forward_contrast = float(components.get("meanForwardContrast", 0.0))
    residual_norm = fit_residual / 100.0
    return [
        1.0,
        dx,
        dy,
        radius,
        radius * radius,
        ray_score,
        center,
        mean_forward,
        min_forward,
        mean_backward,
        max_backward,
        mean_startness,
        min_startness,
        mean_forward_contrast,
        fit_quality,
        residual_norm,
        ray_score * mean_startness,
        ray_score * center,
    ]


def _leave_one_out_predictions(
    rows: Sequence[Dict[str, Any]],
    *,
    config: LearnedVertexConfig,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    for held_out in rows:
        training = [row for row in rows if row.get("key") != held_out.get("key")]
        model = _fit_ranker(training, config=config)
        if model is None:
            predictions.append(_fallback_prediction(held_out))
            continue
        scored = _score_candidates(held_out["_candidateRows"], model)
        selected = max(scored, key=lambda item: item["predictedScore"])
        baseline = min(
            scored,
            key=lambda item: _distance(item["point"], tuple(held_out["baselineVertex"])),
        )
        baseline_error = float(held_out["baselineVertexErrorPx"])
        selected_error = float(selected["targetErrorPx"])
        score_gain = float(selected["predictedScore"]) - float(baseline["predictedScore"])
        accepted = (
            score_gain >= config.default_gate_min_score_gain
            and float(selected["predictedScore"]) >= config.default_gate_min_predicted_score
        )
        gated_error = selected_error if accepted else baseline_error
        return_fields = {
            "key": held_out["key"],
            "trainingRowCount": len(training),
            "learnedTop1Vertex": _round_point(selected["point"]),
            "learnedTop1VertexErrorPx": round(selected_error, 2),
            "learnedTop1ImprovementPx": round(baseline_error - selected_error, 2),
            "learnedTop1Status": _vertex_status(selected_error),
            "learnedPredictedScore": round(float(selected["predictedScore"]), 6),
            "learnedBaselinePredictedScore": round(float(baseline["predictedScore"]), 6),
            "learnedScoreGain": round(score_gain, 6),
            "learnedGatedVertex": _round_point(selected["point"] if accepted else tuple(held_out["baselineVertex"])),
            "learnedGatedAccepted": bool(accepted),
            "learnedGatedVertexErrorPx": round(gated_error, 2),
            "learnedGatedImprovementPx": round(baseline_error - gated_error, 2),
            "learnedGatedStatus": _vertex_status(gated_error),
        }
        predictions.append(return_fields)
    return predictions


def _fit_ranker(
    rows: Sequence[Dict[str, Any]],
    *,
    config: LearnedVertexConfig,
) -> Optional[Dict[str, Any]]:
    feature_rows: List[List[float]] = []
    targets: List[float] = []
    weights: List[float] = []
    for row in rows:
        for candidate in row.get("_candidateRows", []):
            target = float(candidate["targetScore"])
            feature_rows.append(candidate["features"])
            targets.append(target)
            weights.append(0.15 + target * 2.0)
    if not feature_rows:
        return None
    x = np.asarray(feature_rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    x_scaled = (x - mean) / std
    x_scaled[:, 0] = 1.0
    weight_sqrt = np.sqrt(w)[:, None]
    xw = x_scaled * weight_sqrt
    yw = y * weight_sqrt[:, 0]
    ridge = np.eye(x_scaled.shape[1], dtype=np.float64) * config.ridge_lambda
    ridge[0, 0] = 1e-6
    coef = np.linalg.solve(xw.T @ xw + ridge, xw.T @ yw)
    return {"mean": mean, "std": std, "coef": coef}


def _score_candidates(candidates: Sequence[Dict[str, Any]], model: Dict[str, Any]) -> List[Dict[str, Any]]:
    x = np.asarray([candidate["features"] for candidate in candidates], dtype=np.float64)
    x_scaled = (x - model["mean"]) / model["std"]
    x_scaled[:, 0] = 1.0
    scores = x_scaled @ model["coef"]
    results: List[Dict[str, Any]] = []
    for candidate, score in zip(candidates, scores):
        results.append({**candidate, "predictedScore": float(score)})
    return results


def _fallback_prediction(row: Dict[str, Any]) -> Dict[str, Any]:
    baseline_error = float(row["baselineVertexErrorPx"])
    return {
        "key": row["key"],
        "trainingRowCount": 0,
        "learnedTop1Vertex": row["baselineVertex"],
        "learnedTop1VertexErrorPx": baseline_error,
        "learnedTop1ImprovementPx": 0.0,
        "learnedTop1Status": _vertex_status(baseline_error),
        "learnedPredictedScore": None,
        "learnedBaselinePredictedScore": None,
        "learnedScoreGain": None,
        "learnedGatedVertex": row["baselineVertex"],
        "learnedGatedAccepted": False,
        "learnedGatedVertexErrorPx": baseline_error,
        "learnedGatedImprovementPx": 0.0,
        "learnedGatedStatus": _vertex_status(baseline_error),
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
        "candidateOracleStrictCount": _count_within(ok, "candidateOracleVertexErrorPx", STRICT_VERTEX_PX),
        "candidateOraclePlausibleCount": _count_within(ok, "candidateOracleVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "learnedTop1StrictCount": _count_within(ok, "learnedTop1VertexErrorPx", STRICT_VERTEX_PX),
        "learnedTop1PlausibleCount": _count_within(ok, "learnedTop1VertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "learnedGatedStrictCount": _count_within(ok, "learnedGatedVertexErrorPx", STRICT_VERTEX_PX),
        "learnedGatedPlausibleCount": _count_within(ok, "learnedGatedVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "axisGoodBaselineStrictCount": _count_within(axis_good, "baselineVertexErrorPx", STRICT_VERTEX_PX),
        "axisGoodLearnedGatedStrictCount": _count_within(axis_good, "learnedGatedVertexErrorPx", STRICT_VERTEX_PX),
        "learnedGatedAcceptedCount": sum(1 for row in ok if row.get("learnedGatedAccepted")),
        "learnedTop1ImprovedRowCount": sum(1 for row in ok if float(row["learnedTop1ImprovementPx"]) > 5.0),
        "learnedTop1WorsenedRowCount": sum(1 for row in ok if float(row["learnedTop1ImprovementPx"]) < -5.0),
        "learnedGatedImprovedRowCount": sum(1 for row in ok if float(row["learnedGatedImprovementPx"]) > 5.0),
        "learnedGatedWorsenedRowCount": sum(1 for row in ok if float(row["learnedGatedImprovementPx"]) < -5.0),
        "meanBaselineVertexErrorPx": _mean(_values(ok, "baselineVertexErrorPx")),
        "meanCandidateOracleVertexErrorPx": _mean(_values(ok, "candidateOracleVertexErrorPx")),
        "meanLearnedTop1VertexErrorPx": _mean(_values(ok, "learnedTop1VertexErrorPx")),
        "meanLearnedGatedVertexErrorPx": _mean(_values(ok, "learnedGatedVertexErrorPx")),
        "medianBaselineVertexErrorPx": _median(_values(ok, "baselineVertexErrorPx")),
        "medianCandidateOracleVertexErrorPx": _median(_values(ok, "candidateOracleVertexErrorPx")),
        "medianLearnedTop1VertexErrorPx": _median(_values(ok, "learnedTop1VertexErrorPx")),
        "medianLearnedGatedVertexErrorPx": _median(_values(ok, "learnedGatedVertexErrorPx")),
        "learnedThresholdSweep": _learned_threshold_sweep(ok),
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    lines = [
        "# Learned Vertex Localizer V0",
        "",
        "Diagnostics-only artifact. This does not alter recognition behavior.",
        "",
        "This probe trains a lightweight ridge-regression scorer over local candidate features using leave-one-row-out validation on the human-labeled visible trihedral rows.",
        "",
        "## Summary",
        "",
        f"- Rows: {summary['rowCount']}",
        f"- Evaluated rows: {summary['evaluatedRowCount']}",
        f"- Axis-good rows: {summary['axisGoodRowCount']}",
        f"- Axis-blocked rows: {summary['axisBlockedRowCount']}",
        f"- Baseline strict/plausible: {summary['baselineStrictCount']} / {summary['baselinePlausibleCount']}",
        f"- Candidate-grid oracle strict/plausible: {summary['candidateOracleStrictCount']} / {summary['candidateOraclePlausibleCount']}",
        f"- Learned top-1 strict/plausible: {summary['learnedTop1StrictCount']} / {summary['learnedTop1PlausibleCount']}",
        f"- Learned gated strict/plausible: {summary['learnedGatedStrictCount']} / {summary['learnedGatedPlausibleCount']}",
        f"- Axis-good strict baseline/learned-gated: {summary['axisGoodBaselineStrictCount']} / {summary['axisGoodLearnedGatedStrictCount']}",
        f"- Learned gated accepted rows: {summary['learnedGatedAcceptedCount']}",
        f"- Learned top-1 improved/worsened rows by >5px: {summary['learnedTop1ImprovedRowCount']} / {summary['learnedTop1WorsenedRowCount']}",
        f"- Learned gated improved/worsened rows by >5px: {summary['learnedGatedImprovedRowCount']} / {summary['learnedGatedWorsenedRowCount']}",
        f"- Mean vertex error baseline/oracle/top-1/gated: {_fmt(summary['meanBaselineVertexErrorPx'], 'px')} / {_fmt(summary['meanCandidateOracleVertexErrorPx'], 'px')} / {_fmt(summary['meanLearnedTop1VertexErrorPx'], 'px')} / {_fmt(summary['meanLearnedGatedVertexErrorPx'], 'px')}",
        f"- Median vertex error baseline/oracle/top-1/gated: {_fmt(summary['medianBaselineVertexErrorPx'], 'px')} / {_fmt(summary['medianCandidateOracleVertexErrorPx'], 'px')} / {_fmt(summary['medianLearnedTop1VertexErrorPx'], 'px')} / {_fmt(summary['medianLearnedGatedVertexErrorPx'], 'px')}",
        f"- Best non-empty low-worsen learned gate: {_format_sweep(summary['learnedThresholdSweep'].get('bestNonEmptyLowWorsen'))}",
        f"- Best non-empty learned gate with <=2 worsens: {_format_sweep(summary['learnedThresholdSweep'].get('bestNonEmptyAtMostTwoWorsens'))}",
        "",
        "## Rows",
        "",
        "| Row | Axis | Base | Oracle | Learned top-1 | Gated | Accepted | Delta | Score gain |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in document["rows"]:
        if row.get("evaluationStatus") != "ok":
            lines.append(f"| `{row.get('key')}` | `{row.get('evaluationStatus')}` | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| `{row.get('key')}` | `{row.get('axisCategory')}` | "
            f"{_fmt(row.get('baselineVertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('candidateOracleVertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('learnedTop1VertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('learnedGatedVertexErrorPx'), 'px')} | "
            f"{'yes' if row.get('learnedGatedAccepted') else 'no'} | "
            f"{_fmt(row.get('learnedGatedImprovementPx'), 'px')} | "
            f"{_fmt_plain(row.get('learnedScoreGain'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is the first supervised vertex-localization diagnostic, not production wiring.",
            "- The candidate-grid oracle tells us whether a local model-axis search could reach the human vertex if ranking were solved.",
            "- The leave-one-row-out scorer is deliberately tiny and dependency-free; success here would justify a richer learned localizer.",
            "- The V0 result is not production-safe: it improves mean/plausible error, but still worsens multiple already-good vertices and the best low-worsen gate underperforms baseline.",
            "- The strongest conclusion is that local candidate generation is no longer the blocker on these rows; learned ranking/confidence is.",
            "- Next work should train a richer vertex-localization model or collect more labels, rather than adding another hand-tuned geometric score.",
            "",
        ]
    )
    return "\n".join(lines)


def _learned_threshold_sweep(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for min_gain in [-0.05, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20]:
        for min_score in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
            accepted = [
                row
                for row in rows
                if _row_passes_learned_gate(row, min_gain=min_gain, min_score=min_score)
            ]
            if not accepted:
                continue
            improved = sum(1 for row in accepted if float(row["learnedTop1ImprovementPx"]) > 5.0)
            worsened = sum(1 for row in accepted if float(row["learnedTop1ImprovementPx"]) < -5.0)
            errors = [
                float(row["learnedTop1VertexErrorPx"])
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
                        "minScoreGain": min_gain,
                        "minPredictedScore": min_score,
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
            -item["plausibleCount"],
            item["meanVertexErrorPx"],
        ),
    )[0]
    at_most_two = [item for item in candidates if item["worsenedCount"] <= 2]
    best_at_most_two = None
    if at_most_two:
        best_at_most_two = sorted(
            at_most_two,
            key=lambda item: (-item["strictCount"], -item["plausibleCount"], -item["improvedCount"], item["meanVertexErrorPx"]),
        )[0]
    return {
        "evaluatedNonEmptyGateCount": len(candidates),
        "bestNonEmptyLowWorsen": low_worsen,
        "bestNonEmptyAtMostTwoWorsens": best_at_most_two,
    }


def _row_passes_learned_gate(row: Dict[str, Any], *, min_gain: float, min_score: float) -> bool:
    gain = row.get("learnedScoreGain")
    score = row.get("learnedPredictedScore")
    if gain is None or score is None:
        return False
    return float(gain) >= min_gain and float(score) >= min_score


def _format_sweep(item: Optional[Dict[str, Any]]) -> str:
    if not item:
        return "none"
    thresholds = item.get("thresholds") or {}
    return (
        f"accepted {item['acceptedCount']}, improved {item['improvedCount']}, "
        f"worsened {item['worsenedCount']}, strict/plausible "
        f"{item['strictCount']} / {item['plausibleCount']}, "
        f"mean {_fmt(item['meanVertexErrorPx'], 'px')} "
        f"(gain>={thresholds.get('minScoreGain')}, "
        f"score>={thresholds.get('minPredictedScore')})"
    )


def _fmt_plain(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _strip_private_candidate_rows(document: Dict[str, Any]) -> Dict[str, Any]:
    stripped_rows = []
    for row in document["rows"]:
        public_row = {key: value for key, value in row.items() if key != "_candidateRows"}
        stripped_rows.append(public_row)
    return {**document, "rows": stripped_rows}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    document = generate_learned_vertex_localizer_summary(feedback_path=args.feedback)
    public_document = _strip_private_candidate_rows(document)
    _write_json(args.summary_out, public_document)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(public_document), encoding="utf-8")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.report_out}")
    print(
        f"learned gated strict rows: {public_document['summary']['learnedGatedStrictCount']} / "
        f"{public_document['summary']['evaluatedRowCount']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
