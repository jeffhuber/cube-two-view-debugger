#!/usr/bin/env python3
"""Diagnostics-only vertex refinement along fixed trihedral axis rays.

This probe tests the current hypothesis from the completed human axis-label
pass: many rows have plausible axis directions but a bad visible-trihedral
vertex. The production-shaped path uses the current model axes, searches a
bounded neighborhood around the current model vertex, and scores candidate
vertices by dark-line support along the three outgoing rays.

Human labels are used only for evaluation. The optional human-axis oracle is
reported separately to show whether the image objective could work when the
axis family is known.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEEDBACK = ROOT / "tests" / "fixtures" / "vertex_axis_human_feedback_v0.json"
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "axis_ray_vertex_refinement_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "AXIS_RAY_VERTEX_REFINEMENT_V0_REPORT.md"
DEFAULT_OVERLAY_DIR = ROOT / "tmp" / "axis_ray_vertex_refinement_v0_overlays"

STRICT_VERTEX_PX = 30.0
PLAUSIBLE_VERTEX_PX = 50.0
AXIS_GOOD_DEG = 15.0

Point = Tuple[float, float]


@dataclass(frozen=True)
class RefinementConfig:
    search_radius_px: int = 180
    search_step_px: int = 12
    ray_min_px: float = 18.0
    ray_max_px: float = 150.0
    ray_sample_count: int = 9
    center_radius_px: float = 5.0
    line_half_width_px: float = 2.0
    side_offset_px: float = 16.0
    distance_prior_weight: float = 0.08

    def as_dict(self) -> Dict[str, Any]:
        return {
            "searchRadiusPx": self.search_radius_px,
            "searchStepPx": self.search_step_px,
            "rayMinPx": self.ray_min_px,
            "rayMaxPx": self.ray_max_px,
            "raySampleCount": self.ray_sample_count,
            "centerRadiusPx": self.center_radius_px,
            "lineHalfWidthPx": self.line_half_width_px,
            "sideOffsetPx": self.side_offset_px,
            "distancePriorWeight": self.distance_prior_weight,
            "strictVertexPx": STRICT_VERTEX_PX,
            "plausibleVertexPx": PLAUSIBLE_VERTEX_PX,
            "axisGoodDeg": AXIS_GOOD_DEG,
        }


def generate_axis_ray_vertex_refinement_summary(
    *,
    feedback_path: Path = DEFAULT_FEEDBACK,
    config: RefinementConfig = RefinementConfig(),
    overlay_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    feedback = _read_json(feedback_path)
    rows = [
        evaluate_row(row, config=config, overlay_dir=overlay_dir)
        for row in feedback.get("rows", [])
    ]
    return {
        "schemaVersion": 1,
        "probe": "axis_ray_vertex_refinement_v0",
        "description": (
            "Diagnostics-only search for a better visible-trihedral vertex "
            "using dark-line support along fixed outgoing axis rays."
        ),
        "sourceFeedback": str(feedback_path),
        "config": config.as_dict(),
        "summary": summarize_rows(rows),
        "rows": rows,
    }


def evaluate_row(
    row: Dict[str, Any],
    *,
    config: RefinementConfig,
    overlay_dir: Optional[Path],
) -> Dict[str, Any]:
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
    model_refined = refine_vertex(
        darkness,
        start=model_vertex,
        axes=model_axes,
        config=config,
    )
    human_axis_oracle = refine_vertex(
        darkness,
        start=model_vertex,
        axes=human_axis_vectors,
        config=config,
    )
    model_refined_error = _distance(model_refined["point"], human_vertex)
    oracle_error = _distance(human_axis_oracle["point"], human_vertex)
    result = {
        **base,
        "evaluationStatus": "ok",
        "axisCategory": axis_category,
        "modelAxisMaxAngleErrorDeg": round(max_axis_error, 2),
        "baselineVertex": _round_point(model_vertex),
        "baselineVertexErrorPx": round(baseline_error, 2),
        "modelAxisRefinedVertex": _round_point(model_refined["point"]),
        "modelAxisRefinedVertexErrorPx": round(model_refined_error, 2),
        "modelAxisImprovementPx": round(baseline_error - model_refined_error, 2),
        "modelAxisRefinedStatus": _vertex_status(model_refined_error),
        "modelAxisScore": model_refined["score"],
        "modelAxisScoreComponents": model_refined["components"],
        "humanAxisOracleVertex": _round_point(human_axis_oracle["point"]),
        "humanAxisOracleVertexErrorPx": round(oracle_error, 2),
        "humanAxisOracleImprovementPx": round(baseline_error - oracle_error, 2),
        "humanAxisOracleStatus": _vertex_status(oracle_error),
        "humanAxisOracleScore": human_axis_oracle["score"],
        "humanVertex": _round_point(human_vertex),
    }
    if overlay_dir is not None:
        result["overlayPath"] = str(
            render_overlay(
                image=image,
                row=result,
                model_axes=model_axes,
                human_axis_vectors=human_axis_vectors,
                output_dir=overlay_dir,
            )
        )
    return result


def refine_vertex(
    darkness: np.ndarray,
    *,
    start: Point,
    axes: Sequence[Point],
    config: RefinementConfig,
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
    offsets = range(-radius, radius + 1, step)
    for dy in offsets:
        for dx in offsets:
            if dx * dx + dy * dy > radius * radius:
                continue
            candidate = (start[0] + dx, start[1] + dy)
            score, components = score_candidate(
                darkness,
                candidate,
                axis_units,  # type: ignore[arg-type]
                config=config,
            )
            distance_prior = math.hypot(dx, dy) / max(1.0, radius)
            score -= config.distance_prior_weight * distance_prior
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
    config: RefinementConfig,
) -> Tuple[float, Dict[str, Any]]:
    center_dark = _disk_mean(darkness, point, config.center_radius_px)
    if center_dark is None:
        return float("-inf"), {"status": "outside_image"}
    ray_scores: List[float] = []
    ray_darkness: List[float] = []
    ray_contrasts: List[float] = []
    valid_rays = 0
    for axis in axes:
        ray = _score_ray(darkness, point, axis, config=config)
        if ray["validSampleCount"] >= max(3, config.ray_sample_count // 2):
            valid_rays += 1
        ray_scores.append(float(ray["score"]))
        ray_darkness.append(float(ray["meanDarkness"]))
        ray_contrasts.append(float(ray["meanContrast"]))
    if valid_rays < 3:
        return float("-inf"), {"status": "incomplete_ray_samples", "validRays": valid_rays}
    mean_ray_score = statistics.mean(ray_scores)
    min_ray_score = min(ray_scores)
    mean_contrast = statistics.mean(ray_contrasts)
    min_contrast = min(ray_contrasts)
    total = (
        center_dark * 0.85
        + mean_ray_score * 0.95
        + min_ray_score * 0.60
        + max(0.0, mean_contrast) * 0.35
        + max(0.0, min_contrast) * 0.20
    )
    return total, {
        "status": "ok",
        "centerDarkness": round(center_dark, 5),
        "meanRayScore": round(mean_ray_score, 5),
        "minRayScore": round(min_ray_score, 5),
        "meanRayDarkness": round(statistics.mean(ray_darkness), 5),
        "meanRayContrast": round(mean_contrast, 5),
        "minRayContrast": round(min_contrast, 5),
    }


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [row for row in rows if row.get("evaluationStatus") == "ok"]
    axis_good = [row for row in ok if row.get("axisCategory") == "axis_good"]
    axis_blocked = [row for row in ok if row.get("axisCategory") == "axis_blocked"]
    return {
        "rowCount": len(rows),
        "evaluatedRowCount": len(ok),
        "axisGoodRowCount": len(axis_good),
        "axisBlockedRowCount": len(axis_blocked),
        "baselineStrictCount": _count_within(ok, "baselineVertexErrorPx", STRICT_VERTEX_PX),
        "baselinePlausibleCount": _count_within(ok, "baselineVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "modelAxisRefinedStrictCount": _count_within(ok, "modelAxisRefinedVertexErrorPx", STRICT_VERTEX_PX),
        "modelAxisRefinedPlausibleCount": _count_within(ok, "modelAxisRefinedVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "humanAxisOracleStrictCount": _count_within(ok, "humanAxisOracleVertexErrorPx", STRICT_VERTEX_PX),
        "humanAxisOraclePlausibleCount": _count_within(ok, "humanAxisOracleVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "axisGoodBaselineStrictCount": _count_within(axis_good, "baselineVertexErrorPx", STRICT_VERTEX_PX),
        "axisGoodModelAxisRefinedStrictCount": _count_within(axis_good, "modelAxisRefinedVertexErrorPx", STRICT_VERTEX_PX),
        "axisGoodHumanAxisOracleStrictCount": _count_within(axis_good, "humanAxisOracleVertexErrorPx", STRICT_VERTEX_PX),
        "meanBaselineVertexErrorPx": _mean(_values(ok, "baselineVertexErrorPx")),
        "meanModelAxisRefinedVertexErrorPx": _mean(_values(ok, "modelAxisRefinedVertexErrorPx")),
        "meanHumanAxisOracleVertexErrorPx": _mean(_values(ok, "humanAxisOracleVertexErrorPx")),
        "medianBaselineVertexErrorPx": _median(_values(ok, "baselineVertexErrorPx")),
        "medianModelAxisRefinedVertexErrorPx": _median(_values(ok, "modelAxisRefinedVertexErrorPx")),
        "medianHumanAxisOracleVertexErrorPx": _median(_values(ok, "humanAxisOracleVertexErrorPx")),
        "modelAxisImprovedRowCount": sum(1 for row in ok if float(row["modelAxisImprovementPx"]) > 5.0),
        "modelAxisWorsenedRowCount": sum(1 for row in ok if float(row["modelAxisImprovementPx"]) < -5.0),
        "humanAxisOracleImprovedRowCount": sum(1 for row in ok if float(row["humanAxisOracleImprovementPx"]) > 5.0),
        "humanAxisOracleWorsenedRowCount": sum(1 for row in ok if float(row["humanAxisOracleImprovementPx"]) < -5.0),
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    lines = [
        "# Axis-Ray Vertex Refinement V0",
        "",
        "Diagnostics-only artifact. This does not alter recognition behavior.",
        "",
        "This probe searches for a better visible-trihedral vertex by holding the three outgoing axis directions fixed and scoring dark-line support along those rays.",
        "",
        "## Summary",
        "",
        f"- Rows: {summary['rowCount']}",
        f"- Evaluated rows: {summary['evaluatedRowCount']}",
        f"- Axis-good rows: {summary['axisGoodRowCount']}",
        f"- Axis-blocked rows: {summary['axisBlockedRowCount']}",
        f"- Baseline strict/plausible: {summary['baselineStrictCount']} / {summary['baselinePlausibleCount']}",
        f"- Model-axis refined strict/plausible: {summary['modelAxisRefinedStrictCount']} / {summary['modelAxisRefinedPlausibleCount']}",
        f"- Human-axis oracle strict/plausible: {summary['humanAxisOracleStrictCount']} / {summary['humanAxisOraclePlausibleCount']}",
        f"- Axis-good strict baseline/model-axis/oracle: {summary['axisGoodBaselineStrictCount']} / {summary['axisGoodModelAxisRefinedStrictCount']} / {summary['axisGoodHumanAxisOracleStrictCount']}",
        f"- Mean vertex error baseline/model-axis/oracle: {_fmt(summary['meanBaselineVertexErrorPx'], 'px')} / {_fmt(summary['meanModelAxisRefinedVertexErrorPx'], 'px')} / {_fmt(summary['meanHumanAxisOracleVertexErrorPx'], 'px')}",
        f"- Median vertex error baseline/model-axis/oracle: {_fmt(summary['medianBaselineVertexErrorPx'], 'px')} / {_fmt(summary['medianModelAxisRefinedVertexErrorPx'], 'px')} / {_fmt(summary['medianHumanAxisOracleVertexErrorPx'], 'px')}",
        f"- Model-axis improved/worsened rows by >5px: {summary['modelAxisImprovedRowCount']} / {summary['modelAxisWorsenedRowCount']}",
        f"- Human-axis oracle improved/worsened rows by >5px: {summary['humanAxisOracleImprovedRowCount']} / {summary['humanAxisOracleWorsenedRowCount']}",
        "",
        "## Rows",
        "",
        "| Row | Axis category | Baseline | Model-axis refined | Delta | Human-axis oracle | Oracle delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in document["rows"]:
        if row.get("evaluationStatus") != "ok":
            lines.append(f"| `{row.get('key')}` | `{row.get('evaluationStatus')}` | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| `{row.get('key')}` | `{row.get('axisCategory')}` | "
            f"{_fmt(row.get('baselineVertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('modelAxisRefinedVertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('modelAxisImprovementPx'), 'px')} | "
            f"{_fmt(row.get('humanAxisOracleVertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('humanAxisOracleImprovementPx'), 'px')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is an image-objective probe, not production wiring.",
            "- Current readout: model-axis refinement is mixed and not safe. It improves many rows but also worsens many rows, and strict rows do not improve over baseline.",
            "- The human-axis oracle improves mean/median error, but still does not produce a stable strict hit-rate. That points to a weak image objective, not only axis-family selection.",
            "- If model-axis refinement improves the axis-good subset without creating many worsened rows, the next step is to refine the scoring objective and add overlays for manual inspection.",
            "- If the human-axis oracle improves but model-axis refinement does not, axis-family selection remains the blocker.",
            "- If neither improves, the dark-line objective is not enough and we should pivot to stronger junction/line extraction.",
            "",
        ]
    )
    return "\n".join(lines)


def render_overlay(
    *,
    image: Image.Image,
    row: Dict[str, Any],
    model_axes: Sequence[Point],
    human_axis_vectors: Sequence[Point],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    human = _point_or_none(row.get("humanVertex"))
    baseline = _point_or_none(row.get("baselineVertex"))
    model_refined = _point_or_none(row.get("modelAxisRefinedVertex"))
    oracle = _point_or_none(row.get("humanAxisOracleVertex"))
    if baseline is not None:
        _draw_rays(draw, baseline, model_axes, fill=(255, 255, 255, 160))
        _draw_cross(draw, baseline, 16, (255, 255, 255, 240))
    if model_refined is not None:
        _draw_cross(draw, model_refined, 18, (0, 220, 255, 255))
    if oracle is not None:
        _draw_cross(draw, oracle, 14, (255, 180, 0, 255))
    if human is not None:
        _draw_rays(draw, human, human_axis_vectors, fill=(255, 0, 180, 120))
        _draw_cross(draw, human, 20, (255, 0, 180, 255))
    output_path = output_dir / f"{row.get('key')}.jpg"
    Image.alpha_composite(base, overlay).convert("RGB").save(output_path, quality=92)
    return output_path


def _score_ray(
    darkness: np.ndarray,
    point: Point,
    axis: Point,
    *,
    config: RefinementConfig,
) -> Dict[str, Any]:
    normal = (-axis[1], axis[0])
    ts = np.linspace(config.ray_min_px, config.ray_max_px, config.ray_sample_count)
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
        if value is None:
            continue
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


def _sample_bilinear(array: np.ndarray, x: float, y: float) -> Optional[float]:
    height, width = array.shape[:2]
    if x < 0 or y < 0 or x >= width - 1 or y >= height - 1:
        return None
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    dx = x - x0
    dy = y - y0
    v00 = float(array[y0, x0])
    v10 = float(array[y0, x0 + 1])
    v01 = float(array[y0 + 1, x0])
    v11 = float(array[y0 + 1, x0 + 1])
    return (
        v00 * (1.0 - dx) * (1.0 - dy)
        + v10 * dx * (1.0 - dy)
        + v01 * (1.0 - dx) * dy
        + v11 * dx * dy
    )


def _model_axis_vectors(model: Dict[str, Any]) -> List[Point]:
    axes: List[Point] = []
    for axis in model.get("axes", []):
        vector = _point_or_none(axis.get("vector"))
        if axis.get("status") == "ok" and vector is not None:
            axes.append(vector)
    return axes


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
    au = _unit(a)
    bu = _unit(b)
    if au is None or bu is None:
        return 180.0
    dot = au[0] * bu[0] + au[1] * bu[1]
    dot = max(-1.0, min(1.0, dot))
    return abs(math.degrees(math.acos(dot)))


def _unit(value: Point) -> Optional[Point]:
    length = math.hypot(float(value[0]), float(value[1]))
    if length <= 1e-6:
        return None
    return float(value[0]) / length, float(value[1]) / length


def _vertex_status(error_px: float) -> str:
    if error_px <= STRICT_VERTEX_PX:
        return "strict"
    if error_px <= PLAUSIBLE_VERTEX_PX:
        return "plausible"
    return "blocked"


def _count_within(rows: Sequence[Dict[str, Any]], key: str, threshold: float) -> int:
    return sum(1 for row in rows if row.get(key) is not None and float(row[key]) <= threshold)


def _values(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


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


def _load_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _darkness_array(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    return 1.0 - (gray / 255.0)


def _point_or_none(value: Any) -> Optional[Point]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _distance(a: Point, b: Point) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _round_point(point: Point) -> List[float]:
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def _fmt(value: Any, unit: str) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f} {unit}"


def _draw_rays(draw: ImageDraw.ImageDraw, vertex: Point, axes: Sequence[Point], *, fill: Tuple[int, int, int, int]) -> None:
    for axis in axes:
        unit = _unit(axis)
        if unit is None:
            continue
        end = (vertex[0] + unit[0] * 220.0, vertex[1] + unit[1] * 220.0)
        draw.line([vertex, end], fill=fill, width=5)


def _draw_cross(draw: ImageDraw.ImageDraw, point: Point, radius: int, fill: Tuple[int, int, int, int]) -> None:
    x, y = point
    draw.line([(x - radius, y), (x + radius, y)], fill=fill, width=5)
    draw.line([(x, y - radius), (x, y + radius)], fill=fill, width=5)
    draw.ellipse((x - radius * 0.45, y - radius * 0.45, x + radius * 0.45, y + radius * 0.45), fill=fill)


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
    parser.add_argument("--overlay-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    document = generate_axis_ray_vertex_refinement_summary(
        feedback_path=args.feedback,
        overlay_dir=args.overlay_dir,
    )
    _write_json(args.summary_out, document)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(document), encoding="utf-8")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.report_out}")
    if args.overlay_dir is not None:
        print(f"wrote overlays under {args.overlay_dir}")
    print(
        f"model-axis strict rows: {document['summary']['modelAxisRefinedStrictCount']} / "
        f"{document['summary']['evaluatedRowCount']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
