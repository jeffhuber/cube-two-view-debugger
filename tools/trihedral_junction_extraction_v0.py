#!/usr/bin/env python3
"""Diagnostics-only trihedral junction extraction from explicit dark lines.

The previous axis-ray and ray-start probes showed that sampling darkness
along guessed rays is not a safe vertex promotion signal. This probe moves one
step more structural: it extracts three dark line hypotheses near the current
visible-trihedral region, intersects those lines, and evaluates whether the
resulting junction improves over the current global-model vertex.

Human labels are used only for evaluation.
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
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "trihedral_junction_extraction_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "TRIHEDRAL_JUNCTION_EXTRACTION_V0_REPORT.md"


@dataclass(frozen=True)
class JunctionConfig:
    angle_search_deg: Tuple[float, ...] = (-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0)
    offset_radius_px: float = 180.0
    offset_step_px: float = 12.0
    line_extent_px: float = 260.0
    line_sample_count: int = 49
    line_half_width_px: float = 2.0
    side_offset_px: float = 16.0
    max_intersection_spread_px: float = 55.0
    min_line_score: float = 0.70
    min_line_contrast: float = 0.03
    max_move_px: float = 190.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "angleSearchDeg": list(self.angle_search_deg),
            "offsetRadiusPx": self.offset_radius_px,
            "offsetStepPx": self.offset_step_px,
            "lineExtentPx": self.line_extent_px,
            "lineSampleCount": self.line_sample_count,
            "lineHalfWidthPx": self.line_half_width_px,
            "sideOffsetPx": self.side_offset_px,
            "maxIntersectionSpreadPx": self.max_intersection_spread_px,
            "minLineScore": self.min_line_score,
            "minLineContrast": self.min_line_contrast,
            "maxMovePx": self.max_move_px,
            "strictVertexPx": STRICT_VERTEX_PX,
            "plausibleVertexPx": PLAUSIBLE_VERTEX_PX,
            "axisGoodDeg": AXIS_GOOD_DEG,
        }


def generate_trihedral_junction_extraction_summary(
    *,
    feedback_path: Path = DEFAULT_FEEDBACK,
    config: JunctionConfig = JunctionConfig(),
) -> Dict[str, Any]:
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    rows = [evaluate_row(row, config=config) for row in feedback.get("rows", [])]
    return {
        "schemaVersion": 1,
        "probe": "trihedral_junction_extraction_v0",
        "description": (
            "Diagnostics-only explicit dark-line extraction for visible "
            "trihedral junction candidates."
        ),
        "sourceFeedback": str(feedback_path),
        "config": config.as_dict(),
        "summary": summarize_rows(rows),
        "rows": rows,
    }


def evaluate_row(row: Dict[str, Any], *, config: JunctionConfig) -> Dict[str, Any]:
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
    model_junction = extract_junction(darkness, start=model_vertex, axes=model_axes, config=config)
    human_axis_oracle = extract_junction(
        darkness,
        start=model_vertex,
        axes=human_axis_vectors,
        config=config,
    )
    model_gated = _gated_choice(model_vertex, model_junction, config)
    oracle_gated = _gated_choice(model_vertex, human_axis_oracle, config)
    model_best_error = _distance(model_junction["point"], human_vertex) if model_junction["point"] else None
    model_gated_error = _distance(model_gated["point"], human_vertex)
    oracle_best_error = _distance(human_axis_oracle["point"], human_vertex) if human_axis_oracle["point"] else None
    oracle_gated_error = _distance(oracle_gated["point"], human_vertex)
    return {
        **base,
        "evaluationStatus": "ok",
        "axisCategory": axis_category,
        "modelAxisMaxAngleErrorDeg": round(max_axis_error, 2),
        "baselineVertex": _round_point(model_vertex),
        "baselineVertexErrorPx": round(baseline_error, 2),
        "modelJunctionVertex": _round_point(model_junction["point"]) if model_junction["point"] else None,
        "modelJunctionVertexErrorPx": _round_optional(model_best_error),
        "modelJunctionImprovementPx": _round_optional(
            baseline_error - model_best_error if model_best_error is not None else None
        ),
        "modelJunctionGatedVertex": _round_point(model_gated["point"]),
        "modelJunctionGatedAccepted": bool(model_gated["accepted"]),
        "modelJunctionGatedVertexErrorPx": round(model_gated_error, 2),
        "modelJunctionGatedImprovementPx": round(baseline_error - model_gated_error, 2),
        "modelJunctionGatedStatus": _vertex_status(model_gated_error),
        "modelJunctionConfidence": model_junction["confidence"],
        "modelJunction": model_junction["diagnostics"],
        "humanAxisOracleVertex": (
            _round_point(human_axis_oracle["point"]) if human_axis_oracle["point"] else None
        ),
        "humanAxisOracleVertexErrorPx": _round_optional(oracle_best_error),
        "humanAxisOracleImprovementPx": _round_optional(
            baseline_error - oracle_best_error if oracle_best_error is not None else None
        ),
        "humanAxisOracleGatedVertex": _round_point(oracle_gated["point"]),
        "humanAxisOracleGatedAccepted": bool(oracle_gated["accepted"]),
        "humanAxisOracleGatedVertexErrorPx": round(oracle_gated_error, 2),
        "humanAxisOracleGatedImprovementPx": round(baseline_error - oracle_gated_error, 2),
        "humanAxisOracleGatedStatus": _vertex_status(oracle_gated_error),
        "humanAxisOracleConfidence": human_axis_oracle["confidence"],
        "humanVertex": _round_point(human_vertex),
    }


def extract_junction(
    darkness: np.ndarray,
    *,
    start: Point,
    axes: Sequence[Point],
    config: JunctionConfig,
) -> Dict[str, Any]:
    line_candidates = [
        extract_best_line(darkness, start=start, axis=axis, config=config)
        for axis in axes
    ]
    if any(line.get("status") != "ok" for line in line_candidates):
        return {
            "point": None,
            "confidence": 0.0,
            "diagnostics": {
                "status": "missing_line",
                "lines": line_candidates,
            },
        }
    intersections = _pairwise_intersections(line_candidates)
    if len(intersections) != 3:
        return {
            "point": None,
            "confidence": 0.0,
            "diagnostics": {
                "status": "missing_intersection",
                "lines": line_candidates,
                "intersectionCount": len(intersections),
            },
        }
    point = (
        statistics.mean([intersection[0] for intersection in intersections]),
        statistics.mean([intersection[1] for intersection in intersections]),
    )
    spread = max(_distance(point, intersection) for intersection in intersections)
    min_line_score = min(float(line["score"]) for line in line_candidates)
    mean_line_score = statistics.mean(float(line["score"]) for line in line_candidates)
    min_contrast = min(float(line["meanContrast"]) for line in line_candidates)
    move_px = _distance(start, point)
    confidence = (
        mean_line_score * 0.55
        + min_line_score * 0.45
        + max(0.0, min_contrast) * 0.75
        - min(1.0, spread / max(1.0, config.max_intersection_spread_px)) * 0.45
        - min(1.0, move_px / max(1.0, config.max_move_px)) * 0.15
    )
    return {
        "point": point,
        "confidence": round(confidence, 6),
        "diagnostics": {
            "status": "ok",
            "intersectionSpreadPx": round(spread, 2),
            "minLineScore": round(min_line_score, 6),
            "meanLineScore": round(mean_line_score, 6),
            "minLineContrast": round(min_contrast, 6),
            "movePx": round(move_px, 2),
            "pairwiseIntersections": [_round_point(intersection) for intersection in intersections],
            "lines": line_candidates,
        },
    }


def extract_best_line(
    darkness: np.ndarray,
    *,
    start: Point,
    axis: Point,
    config: JunctionConfig,
) -> Dict[str, Any]:
    unit = _unit(axis)
    if unit is None:
        return {"status": "invalid_axis"}
    best: Optional[Dict[str, Any]] = None
    for angle_delta in config.angle_search_deg:
        direction = _rotate(unit, angle_delta)
        normal = (-direction[1], direction[0])
        offsets = _frange(-config.offset_radius_px, config.offset_radius_px, config.offset_step_px)
        for offset in offsets:
            anchor = (start[0] + normal[0] * offset, start[1] + normal[1] * offset)
            score = score_line(darkness, anchor=anchor, direction=direction, config=config)
            if best is None or float(score["score"]) > float(best["score"]):
                best = {
                    "status": "ok",
                    "anchor": _round_point(anchor),
                    "direction": _round_point(direction),
                    "normal": _round_point(normal),
                    "offsetFromStartPx": round(offset, 2),
                    "angleDeltaDeg": angle_delta,
                    **score,
                }
    assert best is not None
    return best


def score_line(
    darkness: np.ndarray,
    *,
    anchor: Point,
    direction: Point,
    config: JunctionConfig,
) -> Dict[str, Any]:
    normal = (-direction[1], direction[0])
    ts = np.linspace(-config.line_extent_px, config.line_extent_px, config.line_sample_count)
    darkness_values: List[float] = []
    contrast_values: List[float] = []
    for t in ts:
        center = (anchor[0] + direction[0] * float(t), anchor[1] + direction[1] * float(t))
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
        darkness_values.append(center_dark)
        contrast_values.append(center_dark - side_dark)
    if len(darkness_values) < max(5, config.line_sample_count // 3):
        return {
            "score": 0.0,
            "meanDarkness": 0.0,
            "meanContrast": 0.0,
            "supportFraction": 0.0,
            "validSampleCount": len(darkness_values),
        }
    dark = np.asarray(darkness_values, dtype=np.float32)
    contrast = np.asarray(contrast_values, dtype=np.float32)
    positive_contrast = np.maximum(contrast, 0.0)
    support = np.logical_and(dark >= 0.45, contrast >= 0.02)
    support_fraction = float(np.count_nonzero(support) / max(1, len(darkness_values)))
    score = (
        float(np.percentile(dark, 80)) * 0.35
        + float(np.percentile(positive_contrast, 80)) * 0.75
        + float(np.mean(np.sort(dark)[-max(3, len(dark) // 4):])) * 0.20
        + support_fraction * 0.35
    )
    return {
        "score": round(score, 6),
        "meanDarkness": round(float(np.mean(dark)), 6),
        "meanContrast": round(float(np.mean(contrast)), 6),
        "supportFraction": round(support_fraction, 6),
        "validSampleCount": len(darkness_values),
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
        "modelJunctionStrictCount": _count_within(ok, "modelJunctionVertexErrorPx", STRICT_VERTEX_PX),
        "modelJunctionPlausibleCount": _count_within(ok, "modelJunctionVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "modelJunctionGatedStrictCount": _count_within(ok, "modelJunctionGatedVertexErrorPx", STRICT_VERTEX_PX),
        "modelJunctionGatedPlausibleCount": _count_within(ok, "modelJunctionGatedVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "humanAxisOracleStrictCount": _count_within(ok, "humanAxisOracleVertexErrorPx", STRICT_VERTEX_PX),
        "humanAxisOraclePlausibleCount": _count_within(ok, "humanAxisOracleVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "humanAxisOracleGatedStrictCount": _count_within(ok, "humanAxisOracleGatedVertexErrorPx", STRICT_VERTEX_PX),
        "humanAxisOracleGatedPlausibleCount": _count_within(ok, "humanAxisOracleGatedVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "axisGoodBaselineStrictCount": _count_within(axis_good, "baselineVertexErrorPx", STRICT_VERTEX_PX),
        "axisGoodModelGatedStrictCount": _count_within(axis_good, "modelJunctionGatedVertexErrorPx", STRICT_VERTEX_PX),
        "modelJunctionGatedAcceptedCount": sum(1 for row in ok if row.get("modelJunctionGatedAccepted")),
        "humanAxisOracleGatedAcceptedCount": sum(1 for row in ok if row.get("humanAxisOracleGatedAccepted")),
        "modelJunctionGatedImprovedRowCount": sum(1 for row in ok if float(row["modelJunctionGatedImprovementPx"]) > 5.0),
        "modelJunctionGatedWorsenedRowCount": sum(1 for row in ok if float(row["modelJunctionGatedImprovementPx"]) < -5.0),
        "humanAxisOracleGatedImprovedRowCount": sum(1 for row in ok if float(row["humanAxisOracleGatedImprovementPx"]) > 5.0),
        "humanAxisOracleGatedWorsenedRowCount": sum(1 for row in ok if float(row["humanAxisOracleGatedImprovementPx"]) < -5.0),
        "meanBaselineVertexErrorPx": _mean(_values(ok, "baselineVertexErrorPx")),
        "meanModelJunctionGatedVertexErrorPx": _mean(_values(ok, "modelJunctionGatedVertexErrorPx")),
        "meanHumanAxisOracleGatedVertexErrorPx": _mean(_values(ok, "humanAxisOracleGatedVertexErrorPx")),
        "medianBaselineVertexErrorPx": _median(_values(ok, "baselineVertexErrorPx")),
        "medianModelJunctionGatedVertexErrorPx": _median(_values(ok, "modelJunctionGatedVertexErrorPx")),
        "medianHumanAxisOracleGatedVertexErrorPx": _median(_values(ok, "humanAxisOracleGatedVertexErrorPx")),
        "modelJunctionThresholdSweep": _model_junction_threshold_sweep(ok),
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    lines = [
        "# Trihedral Junction Extraction V0",
        "",
        "Diagnostics-only artifact. This does not alter recognition behavior.",
        "",
        "This probe extracts three dark line hypotheses near the visible trihedral region, intersects them, and evaluates the resulting junction against human labels.",
        "",
        "## Summary",
        "",
        f"- Rows: {summary['rowCount']}",
        f"- Evaluated rows: {summary['evaluatedRowCount']}",
        f"- Axis-good rows: {summary['axisGoodRowCount']}",
        f"- Axis-blocked rows: {summary['axisBlockedRowCount']}",
        f"- Baseline strict/plausible: {summary['baselineStrictCount']} / {summary['baselinePlausibleCount']}",
        f"- Model junction strict/plausible: {summary['modelJunctionStrictCount']} / {summary['modelJunctionPlausibleCount']}",
        f"- Model junction gated strict/plausible: {summary['modelJunctionGatedStrictCount']} / {summary['modelJunctionGatedPlausibleCount']}",
        f"- Human-axis oracle gated strict/plausible: {summary['humanAxisOracleGatedStrictCount']} / {summary['humanAxisOracleGatedPlausibleCount']}",
        f"- Axis-good strict baseline/model-gated: {summary['axisGoodBaselineStrictCount']} / {summary['axisGoodModelGatedStrictCount']}",
        f"- Model junction gated accepted rows: {summary['modelJunctionGatedAcceptedCount']}",
        f"- Human-axis oracle gated accepted rows: {summary['humanAxisOracleGatedAcceptedCount']}",
        f"- Model junction gated improved/worsened rows by >5px: {summary['modelJunctionGatedImprovedRowCount']} / {summary['modelJunctionGatedWorsenedRowCount']}",
        f"- Human-axis oracle gated improved/worsened rows by >5px: {summary['humanAxisOracleGatedImprovedRowCount']} / {summary['humanAxisOracleGatedWorsenedRowCount']}",
        f"- Mean vertex error baseline/model-gated/oracle-gated: {_fmt(summary['meanBaselineVertexErrorPx'], 'px')} / {_fmt(summary['meanModelJunctionGatedVertexErrorPx'], 'px')} / {_fmt(summary['meanHumanAxisOracleGatedVertexErrorPx'], 'px')}",
        f"- Median vertex error baseline/model-gated/oracle-gated: {_fmt(summary['medianBaselineVertexErrorPx'], 'px')} / {_fmt(summary['medianModelJunctionGatedVertexErrorPx'], 'px')} / {_fmt(summary['medianHumanAxisOracleGatedVertexErrorPx'], 'px')}",
        f"- Best non-empty low-worsen model gate: {_format_sweep(summary['modelJunctionThresholdSweep'].get('bestNonEmptyLowWorsen'))}",
        f"- Best non-empty model gate with <=2 worsens: {_format_sweep(summary['modelJunctionThresholdSweep'].get('bestNonEmptyAtMostTwoWorsens'))}",
        "",
        "## Rows",
        "",
        "| Row | Axis | Base | Model gated | Accepted | Delta | Spread | Min line score | Oracle gated | Oracle accepted | Oracle delta |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in document["rows"]:
        if row.get("evaluationStatus") != "ok":
            lines.append(f"| `{row.get('key')}` | `{row.get('evaluationStatus')}` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        diagnostics = row.get("modelJunction") or {}
        lines.append(
            f"| `{row.get('key')}` | `{row.get('axisCategory')}` | "
            f"{_fmt(row.get('baselineVertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('modelJunctionGatedVertexErrorPx'), 'px')} | "
            f"{'yes' if row.get('modelJunctionGatedAccepted') else 'no'} | "
            f"{_fmt(row.get('modelJunctionGatedImprovementPx'), 'px')} | "
            f"{_fmt(diagnostics.get('intersectionSpreadPx'), 'px')} | "
            f"{_fmt_plain(diagnostics.get('minLineScore'))} | "
            f"{_fmt(row.get('humanAxisOracleGatedVertexErrorPx'), 'px')} | "
            f"{'yes' if row.get('humanAxisOracleGatedAccepted') else 'no'} | "
            f"{_fmt(row.get('humanAxisOracleGatedImprovementPx'), 'px')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is explicit line/junction extraction, not production wiring.",
            "- The current implementation is a negative result: default gating reduces strict/plausible rows and accepts too many worsened vertices.",
            "- A useful promotion-shaped signal would increase strict/plausible rows while keeping worsened accepted rows near zero.",
            "- The best low-worsen gate still underperforms baseline, so this line-extraction path should remain a diagnostic and the next step should be learned vertex localization.",
            "",
        ]
    )
    return "\n".join(lines)


def _gated_choice(
    baseline_point: Point,
    junction: Dict[str, Any],
    config: JunctionConfig,
) -> Dict[str, Any]:
    point = junction.get("point")
    diagnostics = junction.get("diagnostics") or {}
    accepted = (
        point is not None
        and float(junction.get("confidence", 0.0)) > 0.0
        and float(diagnostics.get("intersectionSpreadPx", 9999.0)) <= config.max_intersection_spread_px
        and float(diagnostics.get("minLineScore", 0.0)) >= config.min_line_score
        and float(diagnostics.get("minLineContrast", -9999.0)) >= config.min_line_contrast
        and float(diagnostics.get("movePx", 9999.0)) <= config.max_move_px
    )
    if accepted:
        return {"point": point, "accepted": True}
    return {"point": baseline_point, "accepted": False}


def _model_junction_threshold_sweep(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for max_spread in [20.0, 35.0, 55.0, 80.0, 120.0]:
        for min_score in [0.45, 0.55, 0.65, 0.70, 0.80, 0.90, 1.00]:
            for min_contrast in [-0.02, 0.0, 0.02, 0.03, 0.05, 0.08, 0.10]:
                for max_move in [80.0, 120.0, 160.0, 220.0, 320.0]:
                    accepted = [
                        row
                        for row in rows
                        if _row_passes_model_gate(
                            row,
                            max_spread=max_spread,
                            min_score=min_score,
                            min_contrast=min_contrast,
                            max_move=max_move,
                        )
                    ]
                    if not accepted:
                        continue
                    improved = sum(1 for row in accepted if float(row["modelJunctionImprovementPx"]) > 5.0)
                    worsened = sum(1 for row in accepted if float(row["modelJunctionImprovementPx"]) < -5.0)
                    errors = [
                        float(row["modelJunctionVertexErrorPx"])
                        if row in accepted and row.get("modelJunctionVertexErrorPx") is not None
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
                                "maxSpreadPx": max_spread,
                                "minLineScore": min_score,
                                "minLineContrast": min_contrast,
                                "maxMovePx": max_move,
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


def _row_passes_model_gate(
    row: Dict[str, Any],
    *,
    max_spread: float,
    min_score: float,
    min_contrast: float,
    max_move: float,
) -> bool:
    if row.get("modelJunctionVertexErrorPx") is None:
        return False
    diagnostics = row.get("modelJunction") or {}
    return (
        float(diagnostics.get("intersectionSpreadPx", 9999.0)) <= max_spread
        and float(diagnostics.get("minLineScore", 0.0)) >= min_score
        and float(diagnostics.get("minLineContrast", -9999.0)) >= min_contrast
        and float(diagnostics.get("movePx", 9999.0)) <= max_move
    )


def _pairwise_intersections(lines: Sequence[Dict[str, Any]]) -> List[Point]:
    intersections: List[Point] = []
    for first, second in itertools.combinations(lines, 2):
        a = _point_or_none(first.get("anchor"))
        u = _point_or_none(first.get("direction"))
        b = _point_or_none(second.get("anchor"))
        v = _point_or_none(second.get("direction"))
        if a is None or u is None or b is None or v is None:
            continue
        intersection = _intersect_lines(a, u, b, v)
        if intersection is not None:
            intersections.append(intersection)
    return intersections


def _intersect_lines(a: Point, u: Point, b: Point, v: Point) -> Optional[Point]:
    determinant = u[0] * v[1] - u[1] * v[0]
    if abs(determinant) < 1e-6:
        return None
    delta = (b[0] - a[0], b[1] - a[1])
    t = (delta[0] * v[1] - delta[1] * v[0]) / determinant
    return a[0] + t * u[0], a[1] + t * u[1]


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


def _rotate(point: Point, angle_deg: float) -> Point:
    radians = math.radians(angle_deg)
    cos_v = math.cos(radians)
    sin_v = math.sin(radians)
    return point[0] * cos_v - point[1] * sin_v, point[0] * sin_v + point[1] * cos_v


def _frange(start: float, stop: float, step: float) -> List[float]:
    values: List[float] = []
    current = start
    while current <= stop + step * 0.5:
        values.append(current)
        current += step
    return values


def _round_optional(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 2)


def _fmt_plain(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    document = generate_trihedral_junction_extraction_summary(feedback_path=args.feedback)
    _write_json(args.summary_out, document)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(document), encoding="utf-8")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.report_out}")
    print(
        f"model-junction gated strict rows: {document['summary']['modelJunctionGatedStrictCount']} / "
        f"{document['summary']['evaluatedRowCount']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
