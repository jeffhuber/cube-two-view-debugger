#!/usr/bin/env python3
"""Diagnostics-only patch/junction visible-vertex localizer.

The expanded-label benchmark showed that scalar gates on the existing local
vertex scores are not safe: the wide grid often contains the human-visible
trihedral vertex, but the ranker/confidence still selects the wrong dark
junction. This probe keeps that wide candidate reach and adds richer local
image evidence:

* patch darkness/gradient/cornerness around each candidate,
* explicit radial dark-line junction structure, and
* face-boundary consistency along the three projected cube axes.

Human labels are used only for leave-one-row-out evaluation. Nothing here
alters recognizer behavior.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tools.axis_ray_vertex_refinement_v0 import (
    AXIS_GOOD_DEG,
    PLAUSIBLE_VERTEX_PX,
    STRICT_VERTEX_PX,
    Point,
    _count_within,
    _darkness_array,
    _distance,
    _fmt,
    _load_image,
    _mean,
    _median,
    _model_axis_vectors,
    _round_point,
    _sample_bilinear,
    _unit,
    _values,
    _vertex_status,
    _write_json,
)
from tools.expanded_vertex_localizer_v0 import (
    DEFAULT_ACTIVE_FEEDBACK,
    DEFAULT_CANONICAL_FEEDBACK,
    build_expanded_feedback,
)
from tools.knn_vertex_localizer_v0 import _format_sweep as _format_knn_sweep
from tools.learned_vertex_localizer_v0 import (
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    LearnedVertexConfig,
    _fmt_plain,
    prepare_row,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "patch_junction_vertex_localizer_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "PATCH_JUNCTION_VERTEX_LOCALIZER_V0_REPORT.md"


@dataclass(frozen=True)
class PatchJunctionVertexConfig:
    positive_error_px: float = 30.0
    negative_error_px: float = 70.0
    max_positive_prototypes_per_row: int = 10
    max_negative_prototypes_per_row: int = 14
    negative_distance_weight: float = 0.22
    default_gate_min_score_gain: float = 1.5
    default_gate_min_score: float = -3.0
    patch_radii_px: Tuple[int, ...] = (10, 22, 44)
    radial_sample_radius_min_px: float = 8.0
    radial_sample_radius_max_px: float = 58.0
    radial_angle_step_deg: int = 30
    axis_near_min_px: float = 6.0
    axis_near_max_px: float = 60.0
    axis_far_max_px: float = 150.0
    line_half_width_px: float = 2.0
    side_offset_px: float = 14.0
    candidate_config: LearnedVertexConfig = LearnedVertexConfig(search_radius_px=520, search_step_px=64)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "positiveErrorPx": self.positive_error_px,
            "negativeErrorPx": self.negative_error_px,
            "maxPositivePrototypesPerRow": self.max_positive_prototypes_per_row,
            "maxNegativePrototypesPerRow": self.max_negative_prototypes_per_row,
            "negativeDistanceWeight": self.negative_distance_weight,
            "defaultGateMinScoreGain": self.default_gate_min_score_gain,
            "defaultGateMinScore": self.default_gate_min_score,
            "patchRadiiPx": list(self.patch_radii_px),
            "radialSampleRadiusMinPx": self.radial_sample_radius_min_px,
            "radialSampleRadiusMaxPx": self.radial_sample_radius_max_px,
            "radialAngleStepDeg": self.radial_angle_step_deg,
            "axisNearMinPx": self.axis_near_min_px,
            "axisNearMaxPx": self.axis_near_max_px,
            "axisFarMaxPx": self.axis_far_max_px,
            "lineHalfWidthPx": self.line_half_width_px,
            "sideOffsetPx": self.side_offset_px,
            "strictVertexPx": STRICT_VERTEX_PX,
            "plausibleVertexPx": PLAUSIBLE_VERTEX_PX,
            "axisGoodDeg": AXIS_GOOD_DEG,
            "candidateConfig": self.candidate_config.as_dict(),
            "featureNames": _feature_names(self),
        }


def generate_patch_junction_vertex_localizer_summary(
    *,
    feedback_path: Optional[Path] = None,
    canonical_feedback_path: Path = DEFAULT_CANONICAL_FEEDBACK,
    active_feedback_path: Path = DEFAULT_ACTIVE_FEEDBACK,
    config: PatchJunctionVertexConfig = PatchJunctionVertexConfig(),
) -> Dict[str, Any]:
    feedback = _load_feedback(
        feedback_path=feedback_path,
        canonical_feedback_path=canonical_feedback_path,
        active_feedback_path=active_feedback_path,
    )
    prepared_rows = [
        prepare_patch_junction_row(row, config=config)
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
    public_rows = [_strip_private_candidate_rows(row) for row in rows]
    summary = summarize_rows(public_rows)
    return {
        "schemaVersion": 1,
        "probe": "patch_junction_vertex_localizer_v0",
        "description": (
            "Diagnostics-only leave-one-row-out localizer using patch texture, "
            "explicit radial junction, and face-boundary consistency features."
        ),
        "sourceFeedback": feedback.get("sources") or str(feedback_path),
        "config": config.as_dict(),
        "summary": summary,
        "conclusion": _conclusion(summary),
        "rows": public_rows,
    }


def prepare_patch_junction_row(
    row: Dict[str, Any],
    *,
    config: PatchJunctionVertexConfig,
) -> Dict[str, Any]:
    prepared = prepare_row(row, config=config.candidate_config)
    for key in ("sourceFeedbackLane", "sourceRowKey"):
        if key in row:
            prepared[key] = row[key]
    if prepared.get("evaluationStatus") != "ok":
        return prepared
    image_path = Path(str(row.get("imagePath") or ""))
    image = _load_image(image_path)
    darkness = _darkness_array(image)
    gy, gx = np.gradient(darkness.astype(np.float32))
    gradient = np.hypot(gx, gy)
    integrals = _integral_channels(darkness, gradient, gx, gy)
    model_axes = _model_axis_vectors(row.get("currentModel") or {})
    axis_units = [_unit(axis) for axis in model_axes]
    axis_units = [axis for axis in axis_units if axis is not None]
    if len(axis_units) != 3:
        return {**prepared, "evaluationStatus": "missing_model_trihedral"}
    for candidate in prepared["_candidateRows"]:
        extras = _patch_junction_feature_vector(
            darkness=darkness,
            gradient=gradient,
            gx=gx,
            gy=gy,
            integrals=integrals,
            point=tuple(candidate["point"]),
            axes=axis_units,
            config=config,
        )
        candidate["features"] = [float(value) for value in candidate["features"]] + extras
    prepared["featureCount"] = len(prepared["_candidateRows"][0]["features"])
    return prepared


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
        "patchJunctionTop1StrictCount": _count_within(ok, "patchJunctionTop1VertexErrorPx", STRICT_VERTEX_PX),
        "patchJunctionTop1PlausibleCount": _count_within(ok, "patchJunctionTop1VertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "patchJunctionGatedStrictCount": _count_within(ok, "patchJunctionGatedVertexErrorPx", STRICT_VERTEX_PX),
        "patchJunctionGatedPlausibleCount": _count_within(ok, "patchJunctionGatedVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "axisGoodBaselineStrictCount": _count_within(axis_good, "baselineVertexErrorPx", STRICT_VERTEX_PX),
        "axisGoodPatchJunctionGatedStrictCount": _count_within(axis_good, "patchJunctionGatedVertexErrorPx", STRICT_VERTEX_PX),
        "patchJunctionGatedAcceptedCount": sum(1 for row in ok if row.get("patchJunctionGatedAccepted")),
        "patchJunctionTop1ImprovedRowCount": sum(1 for row in ok if float(row["patchJunctionTop1ImprovementPx"]) > 5.0),
        "patchJunctionTop1WorsenedRowCount": sum(1 for row in ok if float(row["patchJunctionTop1ImprovementPx"]) < -5.0),
        "patchJunctionGatedImprovedRowCount": sum(1 for row in ok if float(row["patchJunctionGatedImprovementPx"]) > 5.0),
        "patchJunctionGatedWorsenedRowCount": sum(1 for row in ok if float(row["patchJunctionGatedImprovementPx"]) < -5.0),
        "meanBaselineVertexErrorPx": _mean(_values(ok, "baselineVertexErrorPx")),
        "meanCandidateOracleVertexErrorPx": _mean(_values(ok, "candidateOracleVertexErrorPx")),
        "meanPatchJunctionTop1VertexErrorPx": _mean(_values(ok, "patchJunctionTop1VertexErrorPx")),
        "meanPatchJunctionGatedVertexErrorPx": _mean(_values(ok, "patchJunctionGatedVertexErrorPx")),
        "medianBaselineVertexErrorPx": _median(_values(ok, "baselineVertexErrorPx")),
        "medianCandidateOracleVertexErrorPx": _median(_values(ok, "candidateOracleVertexErrorPx")),
        "medianPatchJunctionTop1VertexErrorPx": _median(_values(ok, "patchJunctionTop1VertexErrorPx")),
        "medianPatchJunctionGatedVertexErrorPx": _median(_values(ok, "patchJunctionGatedVertexErrorPx")),
        "patchJunctionThresholdSweep": _threshold_sweep(ok),
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    conclusion = document["conclusion"]
    lines = [
        "# Patch/Junction Vertex Localizer V0",
        "",
        "Diagnostics-only artifact. This does not alter recognition behavior.",
        "",
        "This probe stops adding scalar gates to the prior localizer path and instead asks whether richer local image evidence can rank the visible trihedral vertex candidates better. It uses the 58 completed human vertex+axis labels for leave-one-row-out evaluation.",
        "",
        "## Summary",
        "",
        f"- Rows: {summary['rowCount']}",
        f"- Evaluated rows: {summary['evaluatedRowCount']}",
        f"- Axis-good rows: {summary['axisGoodRowCount']}",
        f"- Axis-blocked rows: {summary['axisBlockedRowCount']}",
        f"- Baseline strict/plausible: {summary['baselineStrictCount']} / {summary['baselinePlausibleCount']}",
        f"- Coarse wide candidate-grid oracle strict/plausible: {summary['candidateOracleStrictCount']} / {summary['candidateOraclePlausibleCount']}",
        f"- Patch/junction top-1 strict/plausible: {summary['patchJunctionTop1StrictCount']} / {summary['patchJunctionTop1PlausibleCount']}",
        f"- Patch/junction gated strict/plausible: {summary['patchJunctionGatedStrictCount']} / {summary['patchJunctionGatedPlausibleCount']}",
        f"- Axis-good strict baseline/patch-gated: {summary['axisGoodBaselineStrictCount']} / {summary['axisGoodPatchJunctionGatedStrictCount']}",
        f"- Patch/junction gated accepted rows: {summary['patchJunctionGatedAcceptedCount']}",
        f"- Patch/junction top-1 improved/worsened rows by >5px: {summary['patchJunctionTop1ImprovedRowCount']} / {summary['patchJunctionTop1WorsenedRowCount']}",
        f"- Patch/junction gated improved/worsened rows by >5px: {summary['patchJunctionGatedImprovedRowCount']} / {summary['patchJunctionGatedWorsenedRowCount']}",
        f"- Mean vertex error baseline/oracle/top-1/gated: {_fmt(summary['meanBaselineVertexErrorPx'], 'px')} / {_fmt(summary['meanCandidateOracleVertexErrorPx'], 'px')} / {_fmt(summary['meanPatchJunctionTop1VertexErrorPx'], 'px')} / {_fmt(summary['meanPatchJunctionGatedVertexErrorPx'], 'px')}",
        f"- Median vertex error baseline/oracle/top-1/gated: {_fmt(summary['medianBaselineVertexErrorPx'], 'px')} / {_fmt(summary['medianCandidateOracleVertexErrorPx'], 'px')} / {_fmt(summary['medianPatchJunctionTop1VertexErrorPx'], 'px')} / {_fmt(summary['medianPatchJunctionGatedVertexErrorPx'], 'px')}",
        f"- Best non-empty low-worsen patch/junction gate: {_format_sweep(summary['patchJunctionThresholdSweep'].get('bestNonEmptyLowWorsen'))}",
        f"- Best non-empty zero-worsen patch/junction gate: {_format_sweep(summary['patchJunctionThresholdSweep'].get('bestNonEmptyZeroWorsen'))}",
        f"- Best non-empty patch/junction gate with <=2 worsens: {_format_sweep(summary['patchJunctionThresholdSweep'].get('bestNonEmptyAtMostTwoWorsens'))}",
        f"- Production wiring recommendation: `{conclusion['productionWiringRecommendation']}`.",
        f"- Reason: {conclusion['reason']}",
        "",
        "## Feature Families",
        "",
        "- Patch texture: local darkness, dark-pixel fraction, gradient strength, and structure-tensor cornerness at 10/22/44 px radii.",
        "- Explicit junction structure: radial dark-line support sampled around the candidate, including top-3 arm strength and peak count.",
        "- Face-boundary consistency: near/far forward ray support, side contrast, and backward-continuation penalties along the three projected cube axes.",
        "",
        "## Rows",
        "",
        "| Row | Axis | Base | Oracle | Patch top-1 | Gated | Accepted | Delta | Score gain |",
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
            f"{_fmt(row.get('patchJunctionTop1VertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('patchJunctionGatedVertexErrorPx'), 'px')} | "
            f"{'yes' if row.get('patchJunctionGatedAccepted') else 'no'} | "
            f"{_fmt(row.get('patchJunctionGatedImprovementPx'), 'px')} | "
            f"{_fmt_plain(row.get('patchJunctionScoreGain'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a supervised diagnostics pass, not a recognizer policy.",
            "- The key success test is not just top-1 improvement. A production-shaped signal would need a non-empty confidence gate that improves coverage without accepting worsened rows.",
            "- Compared with the expanded-label KNN baseline from the prior report, patch/junction features modestly improve top-1 strict count but still have poor ungated reliability.",
            "- The conservative gate is intentionally small. It is useful evidence for a possible confidence feature, not a production policy.",
            "- If patch/junction features do not materially improve top-1 and gated safety over the expanded KNN baseline, the next step should be a real trained image-patch model or direct line/junction detector, not more hand-authored thresholds.",
            "- If a zero-worsen gate appears, it should remain diagnostics-only until validated on more labeled rows and hard-background cases.",
            "",
        ]
    )
    return "\n".join(lines)


def _patch_junction_feature_vector(
    *,
    darkness: np.ndarray,
    gradient: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    integrals: Dict[str, np.ndarray],
    point: Point,
    axes: Sequence[Point],
    config: PatchJunctionVertexConfig,
) -> List[float]:
    features: List[float] = []
    for radius in config.patch_radii_px:
        features.extend(_patch_stats(integrals, point, radius, darkness.shape))
    features.extend(_radial_junction_features(darkness, point, config=config))
    features.extend(_axis_boundary_features(darkness, point, axes, config=config))
    return [_finite(value) for value in features]


def _patch_stats(
    integrals: Dict[str, np.ndarray],
    point: Point,
    radius: int,
    shape: Tuple[int, int],
) -> List[float]:
    rect = _rect_bounds(point, radius, shape)
    if rect is None:
        return [0.0] * 10
    x0, y0, x1, y1 = rect
    area = float(max(1, (x1 - x0) * (y1 - y0)))
    dark_mean, dark_std = _rect_mean_std(integrals["dark"], integrals["dark2"], rect, area)
    grad_mean, grad_std = _rect_mean_std(integrals["grad"], integrals["grad2"], rect, area)
    dark_fraction = _rect_sum(integrals["darkmask"], rect) / area
    jxx = _rect_sum(integrals["gx2"], rect) / area
    jyy = _rect_sum(integrals["gy2"], rect) / area
    jxy = _rect_sum(integrals["gxy"], rect) / area
    trace = jxx + jyy
    det = max(0.0, jxx * jyy - jxy * jxy)
    disc = max(0.0, (jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy)
    eig1 = (trace + math.sqrt(disc)) / 2.0
    eig2 = (trace - math.sqrt(disc)) / 2.0
    coherence = (eig1 - eig2) / (eig1 + eig2 + 1e-6)
    harris = det - 0.04 * trace * trace
    return [
        dark_mean,
        dark_std,
        min(1.0, dark_mean + 0.674 * dark_std),
        min(1.0, dark_mean + 1.282 * dark_std),
        dark_fraction,
        grad_mean,
        grad_mean + 1.282 * grad_std,
        trace,
        harris,
        coherence,
    ]


def _radial_junction_features(
    darkness: np.ndarray,
    point: Point,
    *,
    config: PatchJunctionVertexConfig,
) -> List[float]:
    values: List[float] = []
    for angle_deg in range(0, 360, max(1, int(config.radial_angle_step_deg))):
        angle = math.radians(angle_deg)
        direction = (math.cos(angle), math.sin(angle))
        samples = _sample_line_values(
            darkness,
            point,
            direction,
            min_px=config.radial_sample_radius_min_px,
            max_px=config.radial_sample_radius_max_px,
            count=6,
        )
        values.append(float(np.mean(samples)) if samples else 0.0)
    if not values:
        return [0.0] * 8
    arr = np.asarray(values, dtype=np.float64)
    sorted_values = np.sort(arr)
    top3 = sorted_values[-3:] if len(sorted_values) >= 3 else sorted_values
    rest = sorted_values[:-3] if len(sorted_values) > 3 else sorted_values
    threshold = float(np.median(arr) + 0.35 * np.std(arr))
    peak_count = _local_peak_count(arr, threshold=threshold)
    top3_mean = float(np.mean(top3)) if len(top3) else 0.0
    rest_mean = float(np.mean(rest)) if len(rest) else 0.0
    top3_gap = top3_mean - rest_mean
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.max(arr)),
        top3_mean,
        rest_mean,
        top3_gap,
        float(peak_count) / max(1.0, len(arr)),
        float(np.percentile(arr, 90) - np.percentile(arr, 50)),
    ]


def _axis_boundary_features(
    darkness: np.ndarray,
    point: Point,
    axes: Sequence[Point],
    *,
    config: PatchJunctionVertexConfig,
) -> List[float]:
    near_forward_dark: List[float] = []
    far_forward_dark: List[float] = []
    near_forward_contrast: List[float] = []
    backward_dark: List[float] = []
    backward_contrast: List[float] = []
    startness: List[float] = []
    for axis in axes:
        near = _directional_axis_profile(
            darkness,
            point,
            axis,
            min_px=config.axis_near_min_px,
            max_px=config.axis_near_max_px,
            config=config,
        )
        far = _directional_axis_profile(
            darkness,
            point,
            axis,
            min_px=config.axis_near_max_px,
            max_px=config.axis_far_max_px,
            config=config,
        )
        back = _directional_axis_profile(
            darkness,
            point,
            (-axis[0], -axis[1]),
            min_px=config.axis_near_min_px,
            max_px=config.axis_near_max_px,
            config=config,
        )
        near_forward_dark.append(near["meanDarkness"])
        far_forward_dark.append(far["meanDarkness"])
        near_forward_contrast.append(near["meanContrast"])
        backward_dark.append(back["meanDarkness"])
        backward_contrast.append(back["meanContrast"])
        startness.append(near["meanDarkness"] - back["meanDarkness"])
    return [
        float(np.mean(near_forward_dark)),
        float(np.min(near_forward_dark)),
        float(np.mean(far_forward_dark)),
        float(np.min(far_forward_dark)),
        float(np.mean(near_forward_contrast)),
        float(np.min(near_forward_contrast)),
        float(np.mean(backward_dark)),
        float(np.max(backward_dark)),
        float(np.mean(backward_contrast)),
        float(np.mean(startness)),
        float(np.min(startness)),
        float(np.min(near_forward_dark) - np.max(backward_dark)),
    ]


def _directional_axis_profile(
    darkness: np.ndarray,
    point: Point,
    axis: Point,
    *,
    min_px: float,
    max_px: float,
    config: PatchJunctionVertexConfig,
) -> Dict[str, float]:
    normal = (-axis[1], axis[0])
    center_values: List[float] = []
    contrast_values: List[float] = []
    for t in np.linspace(min_px, max_px, 6):
        center = (point[0] + axis[0] * float(t), point[1] + axis[1] * float(t))
        center_dark = _strip_mean(
            darkness,
            center,
            normal,
            [-config.line_half_width_px, 0.0, config.line_half_width_px],
        )
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
        return {"meanDarkness": 0.0, "meanContrast": 0.0}
    return {
        "meanDarkness": float(np.mean(center_values)),
        "meanContrast": float(np.mean(contrast_values)),
    }


def _leave_one_out_predictions(
    rows: Sequence[Dict[str, Any]],
    *,
    config: PatchJunctionVertexConfig,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    for held_out in rows:
        training = [row for row in rows if row.get("key") != held_out.get("key")]
        model = _fit_prototype_model(training, config=config)
        if model is None:
            predictions.append(_fallback_prediction(held_out))
            continue
        scored = _score_candidates(held_out["_candidateRows"], model, config=config)
        selected = max(scored, key=lambda item: item["patchJunctionScore"])
        baseline = min(
            scored,
            key=lambda item: _distance(item["point"], tuple(held_out["baselineVertex"])),
        )
        baseline_error = float(held_out["baselineVertexErrorPx"])
        selected_error = float(selected["targetErrorPx"])
        score_gain = float(selected["patchJunctionScore"]) - float(baseline["patchJunctionScore"])
        accepted = (
            score_gain >= config.default_gate_min_score_gain
            and float(selected["patchJunctionScore"]) >= config.default_gate_min_score
        )
        gated_error = selected_error if accepted else baseline_error
        predictions.append(
            {
                "key": held_out["key"],
                "trainingRowCount": len(training),
                "patchJunctionPositivePrototypeCount": model["positiveCount"],
                "patchJunctionNegativePrototypeCount": model["negativeCount"],
                "patchJunctionTop1Vertex": _round_point(selected["point"]),
                "patchJunctionTop1VertexErrorPx": round(selected_error, 2),
                "patchJunctionTop1ImprovementPx": round(baseline_error - selected_error, 2),
                "patchJunctionTop1Status": _vertex_status(selected_error),
                "patchJunctionScore": round(float(selected["patchJunctionScore"]), 6),
                "patchJunctionBaselineScore": round(float(baseline["patchJunctionScore"]), 6),
                "patchJunctionScoreGain": round(score_gain, 6),
                "patchJunctionMeanPositiveDistance": round(float(selected["meanPositiveDistance"]), 6),
                "patchJunctionMeanNegativeDistance": round(float(selected["meanNegativeDistance"]), 6),
                "patchJunctionGatedVertex": _round_point(selected["point"] if accepted else tuple(held_out["baselineVertex"])),
                "patchJunctionGatedAccepted": bool(accepted),
                "patchJunctionGatedVertexErrorPx": round(gated_error, 2),
                "patchJunctionGatedImprovementPx": round(baseline_error - gated_error, 2),
                "patchJunctionGatedStatus": _vertex_status(gated_error),
            }
        )
    return predictions


def _fit_prototype_model(
    rows: Sequence[Dict[str, Any]],
    *,
    config: PatchJunctionVertexConfig,
) -> Optional[Dict[str, Any]]:
    feature_rows: List[List[float]] = []
    errors: List[float] = []
    for row in rows:
        for candidate in _prototype_candidates_for_row(row, config=config):
            feature_rows.append(candidate["features"])
            errors.append(float(candidate["targetErrorPx"]))
    if not feature_rows:
        return None
    x = np.asarray(feature_rows, dtype=np.float64)
    y = np.asarray(errors, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std
    z[:, 0] = 1.0
    positives = z[y <= config.positive_error_px]
    negatives = z[y >= config.negative_error_px]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    return {
        "mean": mean,
        "std": std,
        "positives": positives,
        "negatives": negatives,
        "positiveCentroid": positives.mean(axis=0),
        "negativeCentroid": negatives.mean(axis=0),
        "positiveCount": int(len(positives)),
        "negativeCount": int(len(negatives)),
    }


def _prototype_candidates_for_row(
    row: Dict[str, Any],
    *,
    config: PatchJunctionVertexConfig,
) -> List[Dict[str, Any]]:
    candidates = list(row.get("_candidateRows", []))
    positives = [
        candidate
        for candidate in candidates
        if float(candidate["targetErrorPx"]) <= config.positive_error_px
    ]
    positives = sorted(positives, key=lambda item: float(item["targetErrorPx"]))[
        : max(1, config.max_positive_prototypes_per_row)
    ]
    negatives = [
        candidate
        for candidate in candidates
        if float(candidate["targetErrorPx"]) >= config.negative_error_px
    ]
    # Keep hard-ish negatives near the true vertex boundary and strong ray-score
    # negatives. The mix prevents the prototype pool from being dominated by
    # far-away blank image patches.
    half = max(1, config.max_negative_prototypes_per_row // 2)
    near_negatives = sorted(negatives, key=lambda item: float(item["targetErrorPx"]))[:half]
    ray_negatives = sorted(
        negatives,
        key=lambda item: float(item.get("rayScore", 0.0)),
        reverse=True,
    )[: max(1, config.max_negative_prototypes_per_row - len(near_negatives))]
    seen = set()
    selected_negatives: List[Dict[str, Any]] = []
    for candidate in near_negatives + ray_negatives:
        point = candidate.get("point")
        key = tuple(round(float(value), 3) for value in point)
        if key in seen:
            continue
        seen.add(key)
        selected_negatives.append(candidate)
    return positives + selected_negatives


def _score_candidates(
    candidates: Sequence[Dict[str, Any]],
    model: Dict[str, Any],
    *,
    config: PatchJunctionVertexConfig,
) -> List[Dict[str, Any]]:
    x = np.asarray([candidate["features"] for candidate in candidates], dtype=np.float64)
    z = (x - model["mean"]) / model["std"]
    z[:, 0] = 1.0
    pos_dist = np.sqrt(((z - model["positiveCentroid"][None, :]) ** 2).sum(axis=1))
    neg_dist = np.sqrt(((z - model["negativeCentroid"][None, :]) ** 2).sum(axis=1))
    scores = -pos_dist + config.negative_distance_weight * neg_dist
    return [
        {
            **candidate,
            "patchJunctionScore": float(score),
            "meanPositiveDistance": float(pos),
            "meanNegativeDistance": float(neg),
        }
        for candidate, score, pos, neg in zip(candidates, scores, pos_dist, neg_dist)
    ]


def _threshold_sweep(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for min_gain in [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
        for min_score in [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0]:
            accepted = [
                row
                for row in rows
                if _row_passes_gate(row, min_gain=min_gain, min_score=min_score)
            ]
            if not accepted:
                continue
            improved = sum(1 for row in accepted if float(row["patchJunctionTop1ImprovementPx"]) > 5.0)
            worsened = sum(1 for row in accepted if float(row["patchJunctionTop1ImprovementPx"]) < -5.0)
            errors = [
                float(row["patchJunctionTop1VertexErrorPx"])
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
                        "minScore": min_score,
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
    zero_worsen = [item for item in candidates if item["worsenedCount"] == 0]
    at_most_two = [item for item in candidates if item["worsenedCount"] <= 2]
    return {
        "evaluatedNonEmptyGateCount": len(candidates),
        "bestNonEmptyLowWorsen": low_worsen,
        "bestNonEmptyZeroWorsen": _best_gate(zero_worsen),
        "bestNonEmptyAtMostTwoWorsens": _best_gate(at_most_two),
    }


def _best_gate(candidates: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            -item["strictCount"],
            -item["worsenedCount"],
            -item["plausibleCount"],
            -item["improvedCount"],
            item["meanVertexErrorPx"],
        ),
    )[0]


def _row_passes_gate(row: Dict[str, Any], *, min_gain: float, min_score: float) -> bool:
    gain = row.get("patchJunctionScoreGain")
    score = row.get("patchJunctionScore")
    if gain is None or score is None:
        return False
    return float(gain) >= min_gain and float(score) >= min_score


def _conclusion(summary: Dict[str, Any]) -> Dict[str, Any]:
    sweep = summary.get("patchJunctionThresholdSweep") or {}
    zero_gate = sweep.get("bestNonEmptyZeroWorsen")
    baseline_strict = int(summary.get("baselineStrictCount") or 0)
    top1_strict = int(summary.get("patchJunctionTop1StrictCount") or 0)
    gated_worsens = int(summary.get("patchJunctionGatedWorsenedRowCount") or 0)
    if zero_gate and int(zero_gate.get("strictCount") or 0) > baseline_strict:
        return {
            "productionWiringRecommendation": "diagnostics_only_needs_more_validation",
            "reason": (
                "A zero-worsen gate improves strict coverage on this label set, "
                "but this remains too small and too local for production wiring."
            ),
        }
    if top1_strict > baseline_strict and gated_worsens == 0:
        return {
            "productionWiringRecommendation": "diagnostics_only_needs_more_validation",
            "reason": "Top-1 improves strict coverage without default gated worsens on this label set.",
        }
    return {
        "productionWiringRecommendation": "do_not_wire",
        "reason": (
            "Richer patch/junction features still do not produce a safe, "
            "high-coverage confidence policy on the 58 labels."
        ),
    }


def _fallback_prediction(row: Dict[str, Any]) -> Dict[str, Any]:
    baseline_error = float(row["baselineVertexErrorPx"])
    return {
        "key": row["key"],
        "trainingRowCount": 0,
        "patchJunctionPositivePrototypeCount": 0,
        "patchJunctionNegativePrototypeCount": 0,
        "patchJunctionTop1Vertex": row["baselineVertex"],
        "patchJunctionTop1VertexErrorPx": baseline_error,
        "patchJunctionTop1ImprovementPx": 0.0,
        "patchJunctionTop1Status": _vertex_status(baseline_error),
        "patchJunctionScore": None,
        "patchJunctionBaselineScore": None,
        "patchJunctionScoreGain": None,
        "patchJunctionMeanPositiveDistance": None,
        "patchJunctionMeanNegativeDistance": None,
        "patchJunctionGatedVertex": row["baselineVertex"],
        "patchJunctionGatedAccepted": False,
        "patchJunctionGatedVertexErrorPx": baseline_error,
        "patchJunctionGatedImprovementPx": 0.0,
        "patchJunctionGatedStatus": _vertex_status(baseline_error),
    }


def _load_feedback(
    *,
    feedback_path: Optional[Path],
    canonical_feedback_path: Path,
    active_feedback_path: Path,
) -> Dict[str, Any]:
    if feedback_path is not None:
        return json.loads(feedback_path.read_text(encoding="utf-8"))
    return build_expanded_feedback(
        canonical_feedback_path=canonical_feedback_path,
        active_feedback_path=active_feedback_path,
    )


def _feature_names(config: PatchJunctionVertexConfig) -> List[str]:
    names = list(BASE_FEATURE_NAMES)
    for radius in config.patch_radii_px:
        prefix = f"patch_r{radius}"
        names.extend(
            [
                f"{prefix}_dark_mean",
                f"{prefix}_dark_std",
                f"{prefix}_dark_p75",
                f"{prefix}_dark_p90",
                f"{prefix}_dark_fraction",
                f"{prefix}_grad_mean",
                f"{prefix}_grad_p90",
                f"{prefix}_structure_trace",
                f"{prefix}_harris",
                f"{prefix}_coherence",
            ]
        )
    names.extend(
        [
            "radial_dark_mean",
            "radial_dark_std",
            "radial_dark_max",
            "radial_top3_mean",
            "radial_rest_mean",
            "radial_top3_gap",
            "radial_peak_fraction",
            "radial_p90_minus_p50",
            "axis_near_forward_mean_dark",
            "axis_near_forward_min_dark",
            "axis_far_forward_mean_dark",
            "axis_far_forward_min_dark",
            "axis_near_forward_mean_contrast",
            "axis_near_forward_min_contrast",
            "axis_backward_mean_dark",
            "axis_backward_max_dark",
            "axis_backward_mean_contrast",
            "axis_startness_mean",
            "axis_startness_min",
            "axis_min_forward_minus_max_backward",
        ]
    )
    return names


def _integral_channels(
    darkness: np.ndarray,
    gradient: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
) -> Dict[str, np.ndarray]:
    return {
        "dark": _integral_image(darkness),
        "dark2": _integral_image(darkness * darkness),
        "darkmask": _integral_image((darkness >= 0.42).astype(np.float32)),
        "grad": _integral_image(gradient),
        "grad2": _integral_image(gradient * gradient),
        "gx2": _integral_image(gx * gx),
        "gy2": _integral_image(gy * gy),
        "gxy": _integral_image(gx * gy),
    }


def _integral_image(array: np.ndarray) -> np.ndarray:
    integral = np.cumsum(np.cumsum(array.astype(np.float64), axis=0), axis=1)
    return np.pad(integral, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)


def _rect_bounds(point: Point, radius: int, shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    x, y = point
    height, width = shape[:2]
    x0 = max(0, int(math.floor(x - radius)))
    x1 = min(width, int(math.ceil(x + radius + 1)))
    y0 = max(0, int(math.floor(y - radius)))
    y1 = min(height, int(math.ceil(y + radius + 1)))
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1, y1


def _rect_sum(integral: np.ndarray, rect: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = rect
    return float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])


def _rect_mean_std(
    integral: np.ndarray,
    integral_sq: np.ndarray,
    rect: Tuple[int, int, int, int],
    area: float,
) -> Tuple[float, float]:
    mean = _rect_sum(integral, rect) / area
    mean_sq = _rect_sum(integral_sq, rect) / area
    variance = max(0.0, mean_sq - mean * mean)
    return float(mean), math.sqrt(variance)


def _sample_line_values(
    darkness: np.ndarray,
    point: Point,
    direction: Point,
    *,
    min_px: float,
    max_px: float,
    count: int,
) -> List[float]:
    values: List[float] = []
    for t in np.linspace(min_px, max_px, count):
        value = _sample_bilinear(
            darkness,
            point[0] + direction[0] * float(t),
            point[1] + direction[1] * float(t),
        )
        if value is not None:
            values.append(float(value))
    return values


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
            values.append(float(value))
    if not values:
        return None
    return float(np.mean(values))


def _local_peak_count(values: np.ndarray, *, threshold: float) -> int:
    if len(values) < 3:
        return 0
    count = 0
    for index, value in enumerate(values):
        previous_value = values[index - 1]
        next_value = values[(index + 1) % len(values)]
        if value >= threshold and value >= previous_value and value >= next_value:
            count += 1
    return count


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return value


def _strip_private_candidate_rows(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if key != "_candidateRows"}


def _format_sweep(item: Optional[Dict[str, Any]]) -> str:
    return _format_knn_sweep(item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=None)
    parser.add_argument("--canonical-feedback", type=Path, default=DEFAULT_CANONICAL_FEEDBACK)
    parser.add_argument("--active-feedback", type=Path, default=DEFAULT_ACTIVE_FEEDBACK)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    document = generate_patch_junction_vertex_localizer_summary(
        feedback_path=args.feedback,
        canonical_feedback_path=args.canonical_feedback,
        active_feedback_path=args.active_feedback,
    )
    _write_json(args.summary_out, document)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(document), encoding="utf-8")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.report_out}")
    print(
        f"patch/junction top1/gated strict rows: "
        f"{document['summary']['patchJunctionTop1StrictCount']} / "
        f"{document['summary']['patchJunctionGatedStrictCount']} / "
        f"{document['summary']['evaluatedRowCount']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
