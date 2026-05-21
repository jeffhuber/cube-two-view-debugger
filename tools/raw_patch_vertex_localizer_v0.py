#!/usr/bin/env python3
"""Diagnostics-only raw-patch visible-vertex localizer.

The hand-authored patch/junction scorer found a tiny zero-worsen confidence
pocket, while explicit dark-line intersections stayed negative on the expanded
58-label set. This probe tests the next smallest "real learned patch" idea:
use the same local candidate grid, but train a leave-one-row-out ridge ranker
on raw darkness/gradient patches around each candidate.

Human labels are used only for training/evaluation. Nothing here alters
recognizer behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from tools.axis_ray_vertex_refinement_v0 import (
    AXIS_GOOD_DEG,
    PLAUSIBLE_VERTEX_PX,
    STRICT_VERTEX_PX,
    _count_within,
    _darkness_array,
    _distance,
    _fmt,
    _load_image,
    _mean,
    _median,
    _round_point,
    _sample_bilinear,
    _values,
    _vertex_status,
    _write_json,
)
from tools.expanded_vertex_localizer_v0 import (
    DEFAULT_ACTIVE_FEEDBACK,
    DEFAULT_CANONICAL_FEEDBACK,
    build_expanded_feedback,
)
from tools.learned_vertex_localizer_v0 import (
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    LearnedVertexConfig,
    _fmt_plain,
    prepare_row,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "raw_patch_vertex_localizer_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "RAW_PATCH_VERTEX_LOCALIZER_V0_REPORT.md"


@dataclass(frozen=True)
class RawPatchVertexConfig:
    patch_radius_px: float = 42.0
    patch_size: int = 11
    include_gradient_channel: bool = True
    include_base_features: bool = True
    target_scale_px: float = 55.0
    ridge_lambda: float = 4.0
    default_gate_min_score_gain: float = 0.04
    default_gate_min_predicted_score: float = 0.45
    candidate_config: LearnedVertexConfig = LearnedVertexConfig(search_radius_px=520, search_step_px=64)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "patchRadiusPx": self.patch_radius_px,
            "patchSize": self.patch_size,
            "includeGradientChannel": self.include_gradient_channel,
            "includeBaseFeatures": self.include_base_features,
            "targetScalePx": self.target_scale_px,
            "ridgeLambda": self.ridge_lambda,
            "defaultGateMinScoreGain": self.default_gate_min_score_gain,
            "defaultGateMinPredictedScore": self.default_gate_min_predicted_score,
            "strictVertexPx": STRICT_VERTEX_PX,
            "plausibleVertexPx": PLAUSIBLE_VERTEX_PX,
            "axisGoodDeg": AXIS_GOOD_DEG,
            "candidateConfig": self.candidate_config.as_dict(),
            "featureNames": _feature_names(self),
        }


def generate_raw_patch_vertex_localizer_summary(
    *,
    feedback_path: Optional[Path] = None,
    canonical_feedback_path: Path = DEFAULT_CANONICAL_FEEDBACK,
    active_feedback_path: Path = DEFAULT_ACTIVE_FEEDBACK,
    config: RawPatchVertexConfig = RawPatchVertexConfig(),
) -> Dict[str, Any]:
    feedback = _load_feedback(
        feedback_path=feedback_path,
        canonical_feedback_path=canonical_feedback_path,
        active_feedback_path=active_feedback_path,
    )
    prepared_rows = [prepare_raw_patch_row(row, config=config) for row in feedback.get("rows", [])]
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
        "probe": "raw_patch_vertex_localizer_v0",
        "description": (
            "Diagnostics-only leave-one-row-out raw darkness/gradient patch "
            "ranker for visible trihedral vertex candidates."
        ),
        "sourceFeedback": feedback.get("sources") or str(feedback_path),
        "config": config.as_dict(),
        "summary": summary,
        "conclusion": _conclusion(summary),
        "rows": public_rows,
    }


def prepare_raw_patch_row(row: Dict[str, Any], *, config: RawPatchVertexConfig) -> Dict[str, Any]:
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
    for candidate in prepared["_candidateRows"]:
        base = [float(value) for value in candidate["features"]] if config.include_base_features else []
        patch = _patch_features(darkness, gradient, tuple(candidate["point"]), config=config)
        candidate["features"] = base + patch
        candidate["targetScore"] = math.exp(
            -float(candidate["targetErrorPx"]) / max(1.0, config.target_scale_px)
        )
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
        "rawPatchTop1StrictCount": _count_within(ok, "rawPatchTop1VertexErrorPx", STRICT_VERTEX_PX),
        "rawPatchTop1PlausibleCount": _count_within(ok, "rawPatchTop1VertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "rawPatchGatedStrictCount": _count_within(ok, "rawPatchGatedVertexErrorPx", STRICT_VERTEX_PX),
        "rawPatchGatedPlausibleCount": _count_within(ok, "rawPatchGatedVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "axisGoodBaselineStrictCount": _count_within(axis_good, "baselineVertexErrorPx", STRICT_VERTEX_PX),
        "axisGoodRawPatchGatedStrictCount": _count_within(axis_good, "rawPatchGatedVertexErrorPx", STRICT_VERTEX_PX),
        "rawPatchGatedAcceptedCount": sum(1 for row in ok if row.get("rawPatchGatedAccepted")),
        "rawPatchTop1ImprovedRowCount": sum(1 for row in ok if float(row["rawPatchTop1ImprovementPx"]) > 5.0),
        "rawPatchTop1WorsenedRowCount": sum(1 for row in ok if float(row["rawPatchTop1ImprovementPx"]) < -5.0),
        "rawPatchGatedImprovedRowCount": sum(1 for row in ok if float(row["rawPatchGatedImprovementPx"]) > 5.0),
        "rawPatchGatedWorsenedRowCount": sum(1 for row in ok if float(row["rawPatchGatedImprovementPx"]) < -5.0),
        "meanBaselineVertexErrorPx": _mean(_values(ok, "baselineVertexErrorPx")),
        "meanCandidateOracleVertexErrorPx": _mean(_values(ok, "candidateOracleVertexErrorPx")),
        "meanRawPatchTop1VertexErrorPx": _mean(_values(ok, "rawPatchTop1VertexErrorPx")),
        "meanRawPatchGatedVertexErrorPx": _mean(_values(ok, "rawPatchGatedVertexErrorPx")),
        "medianBaselineVertexErrorPx": _median(_values(ok, "baselineVertexErrorPx")),
        "medianCandidateOracleVertexErrorPx": _median(_values(ok, "candidateOracleVertexErrorPx")),
        "medianRawPatchTop1VertexErrorPx": _median(_values(ok, "rawPatchTop1VertexErrorPx")),
        "medianRawPatchGatedVertexErrorPx": _median(_values(ok, "rawPatchGatedVertexErrorPx")),
        "rawPatchThresholdSweep": _threshold_sweep(ok),
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    conclusion = document["conclusion"]
    lines = [
        "# Raw Patch Vertex Localizer V0",
        "",
        "Diagnostics-only artifact. This does not alter recognition behavior.",
        "",
        "This probe is the first dependency-light raw image-patch vertex localizer over the 58 completed vertex+axis labels. It trains a leave-one-row-out ridge ranker on darkness/gradient patches around each local candidate, with the existing model/ray features kept only as a spatial prior.",
        "",
        "## Summary",
        "",
        f"- Rows: {summary['rowCount']}",
        f"- Evaluated rows: {summary['evaluatedRowCount']}",
        f"- Axis-good rows: {summary['axisGoodRowCount']}",
        f"- Axis-blocked rows: {summary['axisBlockedRowCount']}",
        f"- Baseline strict/plausible: {summary['baselineStrictCount']} / {summary['baselinePlausibleCount']}",
        f"- Coarse wide candidate-grid oracle strict/plausible: {summary['candidateOracleStrictCount']} / {summary['candidateOraclePlausibleCount']}",
        f"- Raw-patch top-1 strict/plausible: {summary['rawPatchTop1StrictCount']} / {summary['rawPatchTop1PlausibleCount']}",
        f"- Raw-patch gated strict/plausible: {summary['rawPatchGatedStrictCount']} / {summary['rawPatchGatedPlausibleCount']}",
        f"- Axis-good strict baseline/raw-patch gated: {summary['axisGoodBaselineStrictCount']} / {summary['axisGoodRawPatchGatedStrictCount']}",
        f"- Raw-patch gated accepted rows: {summary['rawPatchGatedAcceptedCount']}",
        f"- Raw-patch top-1 improved/worsened rows by >5px: {summary['rawPatchTop1ImprovedRowCount']} / {summary['rawPatchTop1WorsenedRowCount']}",
        f"- Raw-patch gated improved/worsened rows by >5px: {summary['rawPatchGatedImprovedRowCount']} / {summary['rawPatchGatedWorsenedRowCount']}",
        f"- Mean vertex error baseline/oracle/top-1/gated: {_fmt(summary['meanBaselineVertexErrorPx'], 'px')} / {_fmt(summary['meanCandidateOracleVertexErrorPx'], 'px')} / {_fmt(summary['meanRawPatchTop1VertexErrorPx'], 'px')} / {_fmt(summary['meanRawPatchGatedVertexErrorPx'], 'px')}",
        f"- Median vertex error baseline/oracle/top-1/gated: {_fmt(summary['medianBaselineVertexErrorPx'], 'px')} / {_fmt(summary['medianCandidateOracleVertexErrorPx'], 'px')} / {_fmt(summary['medianRawPatchTop1VertexErrorPx'], 'px')} / {_fmt(summary['medianRawPatchGatedVertexErrorPx'], 'px')}",
        f"- Best non-empty low-worsen raw-patch gate: {_format_sweep(summary['rawPatchThresholdSweep'].get('bestNonEmptyLowWorsen'))}",
        f"- Best non-empty zero-worsen raw-patch gate: {_format_sweep(summary['rawPatchThresholdSweep'].get('bestNonEmptyZeroWorsen'))}",
        f"- Best non-empty raw-patch gate with <=2 worsens: {_format_sweep(summary['rawPatchThresholdSweep'].get('bestNonEmptyAtMostTwoWorsens'))}",
        f"- Production wiring recommendation: `{conclusion['productionWiringRecommendation']}`.",
        f"- Reason: {conclusion['reason']}",
        "",
        "## Rows",
        "",
        "| Row | Axis | Base | Oracle | Raw top-1 | Gated | Accepted | Delta | Score gain |",
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
            f"{_fmt(row.get('rawPatchTop1VertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('rawPatchGatedVertexErrorPx'), 'px')} | "
            f"{'yes' if row.get('rawPatchGatedAccepted') else 'no'} | "
            f"{_fmt(row.get('rawPatchGatedImprovementPx'), 'px')} | "
            f"{_fmt_plain(row.get('rawPatchScoreGain'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a supervised raw-patch diagnostic, not production wiring.",
            "- It is deliberately small: no CNN dependency and no persistent model artifact. The only question is whether local image patches contain ranking signal beyond hand-authored features.",
            "- If it produces a meaningful zero-worsen gate, the next step is a proper trainable patch model with more labels and held-out validation.",
            "- If it does not improve over the patch/junction feature probe, the limiting factor is likely data volume or candidate/model geometry, not another small ranker.",
            "",
        ]
    )
    return "\n".join(lines)


def _leave_one_out_predictions(
    rows: Sequence[Dict[str, Any]],
    *,
    config: RawPatchVertexConfig,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    for held_out in rows:
        training = [row for row in rows if row.get("key") != held_out.get("key")]
        model = _fit_ranker(training, config=config)
        if model is None:
            predictions.append(_fallback_prediction(held_out))
            continue
        scored = _score_candidates(held_out["_candidateRows"], model)
        selected = max(scored, key=lambda item: item["rawPatchPredictedScore"])
        baseline = min(
            scored,
            key=lambda item: _distance(item["point"], tuple(held_out["baselineVertex"])),
        )
        baseline_error = float(held_out["baselineVertexErrorPx"])
        selected_error = float(selected["targetErrorPx"])
        score_gain = float(selected["rawPatchPredictedScore"]) - float(baseline["rawPatchPredictedScore"])
        accepted = (
            score_gain >= config.default_gate_min_score_gain
            and float(selected["rawPatchPredictedScore"]) >= config.default_gate_min_predicted_score
        )
        gated_error = selected_error if accepted else baseline_error
        predictions.append(
            {
                "key": held_out["key"],
                "trainingRowCount": len(training),
                "rawPatchTop1Vertex": _round_point(selected["point"]),
                "rawPatchTop1VertexErrorPx": round(selected_error, 2),
                "rawPatchTop1ImprovementPx": round(baseline_error - selected_error, 2),
                "rawPatchTop1Status": _vertex_status(selected_error),
                "rawPatchPredictedScore": round(float(selected["rawPatchPredictedScore"]), 6),
                "rawPatchBaselinePredictedScore": round(float(baseline["rawPatchPredictedScore"]), 6),
                "rawPatchScoreGain": round(score_gain, 6),
                "rawPatchGatedVertex": _round_point(selected["point"] if accepted else tuple(held_out["baselineVertex"])),
                "rawPatchGatedAccepted": bool(accepted),
                "rawPatchGatedVertexErrorPx": round(gated_error, 2),
                "rawPatchGatedImprovementPx": round(baseline_error - gated_error, 2),
                "rawPatchGatedStatus": _vertex_status(gated_error),
            }
        )
    return predictions


def _fit_ranker(
    rows: Sequence[Dict[str, Any]],
    *,
    config: RawPatchVertexConfig,
) -> Optional[Dict[str, Any]]:
    feature_rows: List[List[float]] = []
    targets: List[float] = []
    weights: List[float] = []
    for row in rows:
        for candidate in row.get("_candidateRows", []):
            target = float(candidate["targetScore"])
            feature_rows.append(candidate["features"])
            targets.append(target)
            weights.append(0.10 + target * 2.5)
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
    return [
        {**candidate, "rawPatchPredictedScore": float(score)}
        for candidate, score in zip(candidates, scores)
    ]


def _patch_features(
    darkness: np.ndarray,
    gradient: np.ndarray,
    point: Sequence[float],
    *,
    config: RawPatchVertexConfig,
) -> List[float]:
    size = max(3, int(config.patch_size))
    if size % 2 == 0:
        size += 1
    offsets = np.linspace(-config.patch_radius_px, config.patch_radius_px, size)
    values: List[float] = []
    for dy in offsets:
        for dx in offsets:
            values.append(_sample_or_zero(darkness, float(point[0]) + float(dx), float(point[1]) + float(dy)))
    if config.include_gradient_channel:
        for dy in offsets:
            for dx in offsets:
                values.append(_sample_or_zero(gradient, float(point[0]) + float(dx), float(point[1]) + float(dy)))
    return values


def _threshold_sweep(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for min_gain in [-0.10, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20, 0.30]:
        for min_score in [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75]:
            accepted = [
                row
                for row in rows
                if _row_passes_gate(row, min_gain=min_gain, min_score=min_score)
            ]
            if not accepted:
                continue
            improved = sum(1 for row in accepted if float(row["rawPatchTop1ImprovementPx"]) > 5.0)
            worsened = sum(1 for row in accepted if float(row["rawPatchTop1ImprovementPx"]) < -5.0)
            errors = [
                float(row["rawPatchTop1VertexErrorPx"])
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
            item["worsenedCount"],
            -item["plausibleCount"],
            -item["improvedCount"],
            item["meanVertexErrorPx"],
        ),
    )[0]


def _row_passes_gate(row: Dict[str, Any], *, min_gain: float, min_score: float) -> bool:
    gain = row.get("rawPatchScoreGain")
    score = row.get("rawPatchPredictedScore")
    if gain is None or score is None:
        return False
    return float(gain) >= min_gain and float(score) >= min_score


def _conclusion(summary: Dict[str, Any]) -> Dict[str, Any]:
    baseline_strict = int(summary.get("baselineStrictCount") or 0)
    top1_strict = int(summary.get("rawPatchTop1StrictCount") or 0)
    gated_worsens = int(summary.get("rawPatchGatedWorsenedRowCount") or 0)
    zero_gate = (summary.get("rawPatchThresholdSweep") or {}).get("bestNonEmptyZeroWorsen")
    if zero_gate and int(zero_gate.get("strictCount") or 0) > baseline_strict:
        return {
            "productionWiringRecommendation": "diagnostics_only_needs_more_labels",
            "reason": (
                "A zero-worsen raw-patch gate improves strict coverage on this "
                "label set, but it needs more labeled examples before wiring."
            ),
        }
    if top1_strict > baseline_strict and gated_worsens == 0:
        return {
            "productionWiringRecommendation": "diagnostics_only_needs_more_labels",
            "reason": "Top-1 improves strict coverage without default gated worsens on this label set.",
        }
    return {
        "productionWiringRecommendation": "do_not_wire",
        "reason": "Raw-patch ridge ranking is still not safe enough for recognizer wiring.",
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


def _sample_or_zero(array: np.ndarray, x: float, y: float) -> float:
    value = _sample_bilinear(array, x, y)
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return float(value)


def _fallback_prediction(row: Dict[str, Any]) -> Dict[str, Any]:
    baseline_error = float(row["baselineVertexErrorPx"])
    return {
        "key": row["key"],
        "trainingRowCount": 0,
        "rawPatchTop1Vertex": row["baselineVertex"],
        "rawPatchTop1VertexErrorPx": baseline_error,
        "rawPatchTop1ImprovementPx": 0.0,
        "rawPatchTop1Status": _vertex_status(baseline_error),
        "rawPatchPredictedScore": None,
        "rawPatchBaselinePredictedScore": None,
        "rawPatchScoreGain": None,
        "rawPatchGatedVertex": row["baselineVertex"],
        "rawPatchGatedAccepted": False,
        "rawPatchGatedVertexErrorPx": baseline_error,
        "rawPatchGatedImprovementPx": 0.0,
        "rawPatchGatedStatus": _vertex_status(baseline_error),
    }


def _feature_names(config: RawPatchVertexConfig) -> List[str]:
    names: List[str] = list(BASE_FEATURE_NAMES) if config.include_base_features else []
    size = config.patch_size if config.patch_size % 2 else config.patch_size + 1
    for channel in ("dark", "grad") if config.include_gradient_channel else ("dark",):
        names.extend(f"{channel}_{row}_{col}" for row in range(size) for col in range(size))
    return names


def _strip_private_candidate_rows(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if key != "_candidateRows"}


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=None)
    parser.add_argument("--canonical-feedback", type=Path, default=DEFAULT_CANONICAL_FEEDBACK)
    parser.add_argument("--active-feedback", type=Path, default=DEFAULT_ACTIVE_FEEDBACK)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    document = generate_raw_patch_vertex_localizer_summary(
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
        f"raw-patch top1/gated strict rows: "
        f"{document['summary']['rawPatchTop1StrictCount']} / "
        f"{document['summary']['rawPatchGatedStrictCount']} / "
        f"{document['summary']['evaluatedRowCount']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
