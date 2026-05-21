#!/usr/bin/env python3
"""Diagnostics-only non-linear learned visible-vertex localizer.

The linear learned V0 showed that the local candidate grid reaches the human
visible trihedral vertex on every labeled row, but a tiny ridge scorer still
mis-ranked too many candidates. This probe keeps the same candidate/features
and tests a deliberately simple non-linear ranker: leave-one-row-out k-nearest
positive/negative feature prototypes.

Human labels are used only for training/evaluation inside this diagnostics
artifact. Nothing here alters recognizer behavior.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.axis_ray_vertex_refinement_v0 import (
    AXIS_GOOD_DEG,
    DEFAULT_FEEDBACK,
    PLAUSIBLE_VERTEX_PX,
    STRICT_VERTEX_PX,
    _count_within,
    _distance,
    _fmt,
    _mean,
    _median,
    _round_point,
    _values,
    _vertex_status,
    _write_json,
)
from tools.learned_vertex_localizer_v0 import (
    LearnedVertexConfig,
    _fmt_plain,
    prepare_row,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "knn_vertex_localizer_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "KNN_VERTEX_LOCALIZER_V0_REPORT.md"


@dataclass(frozen=True)
class KnnVertexConfig:
    positive_error_px: float = 30.0
    negative_error_px: float = 70.0
    neighbor_count: int = 5
    negative_distance_weight: float = 0.20
    default_gate_min_score_gain: float = 1.0
    default_gate_min_score: float = -6.0
    candidate_config: LearnedVertexConfig = LearnedVertexConfig()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "positiveErrorPx": self.positive_error_px,
            "negativeErrorPx": self.negative_error_px,
            "neighborCount": self.neighbor_count,
            "negativeDistanceWeight": self.negative_distance_weight,
            "defaultGateMinScoreGain": self.default_gate_min_score_gain,
            "defaultGateMinScore": self.default_gate_min_score,
            "strictVertexPx": STRICT_VERTEX_PX,
            "plausibleVertexPx": PLAUSIBLE_VERTEX_PX,
            "axisGoodDeg": AXIS_GOOD_DEG,
            "candidateConfig": self.candidate_config.as_dict(),
        }


def generate_knn_vertex_localizer_summary(
    *,
    feedback_path: Path = DEFAULT_FEEDBACK,
    config: KnnVertexConfig = KnnVertexConfig(),
) -> Dict[str, Any]:
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    prepared_rows = [
        prepare_row(row, config=config.candidate_config)
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
    return {
        "schemaVersion": 1,
        "probe": "knn_vertex_localizer_v0",
        "description": (
            "Diagnostics-only leave-one-row-out kNN prototype scorer for "
            "visible trihedral vertex candidates."
        ),
        "sourceFeedback": str(feedback_path),
        "config": config.as_dict(),
        "summary": summarize_rows(public_rows),
        "rows": public_rows,
    }


def _leave_one_out_predictions(
    rows: Sequence[Dict[str, Any]],
    *,
    config: KnnVertexConfig,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    for held_out in rows:
        training = [row for row in rows if row.get("key") != held_out.get("key")]
        model = _fit_knn_model(training, config=config)
        if model is None:
            predictions.append(_fallback_prediction(held_out))
            continue
        scored = _score_candidates(held_out["_candidateRows"], model, config=config)
        selected = max(scored, key=lambda item: item["knnScore"])
        baseline = min(
            scored,
            key=lambda item: _distance(item["point"], tuple(held_out["baselineVertex"])),
        )
        baseline_error = float(held_out["baselineVertexErrorPx"])
        selected_error = float(selected["targetErrorPx"])
        score_gain = float(selected["knnScore"]) - float(baseline["knnScore"])
        accepted = (
            score_gain >= config.default_gate_min_score_gain
            and float(selected["knnScore"]) >= config.default_gate_min_score
        )
        gated_error = selected_error if accepted else baseline_error
        return_fields = {
            "key": held_out["key"],
            "trainingRowCount": len(training),
            "knnPositivePrototypeCount": model["positiveCount"],
            "knnNegativePrototypeCount": model["negativeCount"],
            "knnTop1Vertex": _round_point(selected["point"]),
            "knnTop1VertexErrorPx": round(selected_error, 2),
            "knnTop1ImprovementPx": round(baseline_error - selected_error, 2),
            "knnTop1Status": _vertex_status(selected_error),
            "knnScore": round(float(selected["knnScore"]), 6),
            "knnBaselineScore": round(float(baseline["knnScore"]), 6),
            "knnScoreGain": round(score_gain, 6),
            "knnMeanPositiveDistance": round(float(selected["meanPositiveDistance"]), 6),
            "knnMeanNegativeDistance": round(float(selected["meanNegativeDistance"]), 6),
            "knnGatedVertex": _round_point(selected["point"] if accepted else tuple(held_out["baselineVertex"])),
            "knnGatedAccepted": bool(accepted),
            "knnGatedVertexErrorPx": round(gated_error, 2),
            "knnGatedImprovementPx": round(baseline_error - gated_error, 2),
            "knnGatedStatus": _vertex_status(gated_error),
        }
        predictions.append(return_fields)
    return predictions


def _fit_knn_model(
    rows: Sequence[Dict[str, Any]],
    *,
    config: KnnVertexConfig,
) -> Optional[Dict[str, Any]]:
    feature_rows: List[List[float]] = []
    errors: List[float] = []
    for row in rows:
        for candidate in row.get("_candidateRows", []):
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
        "positiveCount": int(len(positives)),
        "negativeCount": int(len(negatives)),
    }


def _score_candidates(
    candidates: Sequence[Dict[str, Any]],
    model: Dict[str, Any],
    *,
    config: KnnVertexConfig,
) -> List[Dict[str, Any]]:
    x = np.asarray([candidate["features"] for candidate in candidates], dtype=np.float64)
    z = (x - model["mean"]) / model["std"]
    z[:, 0] = 1.0
    pos_dist = _mean_knn_distance(z, model["positives"], config.neighbor_count)
    neg_dist = _mean_knn_distance(z, model["negatives"], config.neighbor_count)
    scores = -pos_dist + config.negative_distance_weight * neg_dist
    scored: List[Dict[str, Any]] = []
    for candidate, score, pos, neg in zip(candidates, scores, pos_dist, neg_dist):
        scored.append(
            {
                **candidate,
                "knnScore": float(score),
                "meanPositiveDistance": float(pos),
                "meanNegativeDistance": float(neg),
            }
        )
    return scored


def _mean_knn_distance(candidates: np.ndarray, prototypes: np.ndarray, neighbor_count: int) -> np.ndarray:
    if len(prototypes) == 0:
        return np.full(len(candidates), float("inf"), dtype=np.float64)
    k = min(max(1, neighbor_count), len(prototypes))
    values: List[np.ndarray] = []
    for start in range(0, len(candidates), 256):
        chunk = candidates[start:start + 256]
        distances = ((chunk[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
        nearest = np.partition(distances, k - 1, axis=1)[:, :k]
        values.append(np.sqrt(nearest).mean(axis=1))
    return np.concatenate(values, axis=0)


def _fallback_prediction(row: Dict[str, Any]) -> Dict[str, Any]:
    baseline_error = float(row["baselineVertexErrorPx"])
    return {
        "key": row["key"],
        "trainingRowCount": 0,
        "knnPositivePrototypeCount": 0,
        "knnNegativePrototypeCount": 0,
        "knnTop1Vertex": row["baselineVertex"],
        "knnTop1VertexErrorPx": baseline_error,
        "knnTop1ImprovementPx": 0.0,
        "knnTop1Status": _vertex_status(baseline_error),
        "knnScore": None,
        "knnBaselineScore": None,
        "knnScoreGain": None,
        "knnMeanPositiveDistance": None,
        "knnMeanNegativeDistance": None,
        "knnGatedVertex": row["baselineVertex"],
        "knnGatedAccepted": False,
        "knnGatedVertexErrorPx": baseline_error,
        "knnGatedImprovementPx": 0.0,
        "knnGatedStatus": _vertex_status(baseline_error),
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
        "knnTop1StrictCount": _count_within(ok, "knnTop1VertexErrorPx", STRICT_VERTEX_PX),
        "knnTop1PlausibleCount": _count_within(ok, "knnTop1VertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "knnGatedStrictCount": _count_within(ok, "knnGatedVertexErrorPx", STRICT_VERTEX_PX),
        "knnGatedPlausibleCount": _count_within(ok, "knnGatedVertexErrorPx", PLAUSIBLE_VERTEX_PX),
        "axisGoodBaselineStrictCount": _count_within(axis_good, "baselineVertexErrorPx", STRICT_VERTEX_PX),
        "axisGoodKnnGatedStrictCount": _count_within(axis_good, "knnGatedVertexErrorPx", STRICT_VERTEX_PX),
        "knnGatedAcceptedCount": sum(1 for row in ok if row.get("knnGatedAccepted")),
        "knnTop1ImprovedRowCount": sum(1 for row in ok if float(row["knnTop1ImprovementPx"]) > 5.0),
        "knnTop1WorsenedRowCount": sum(1 for row in ok if float(row["knnTop1ImprovementPx"]) < -5.0),
        "knnGatedImprovedRowCount": sum(1 for row in ok if float(row["knnGatedImprovementPx"]) > 5.0),
        "knnGatedWorsenedRowCount": sum(1 for row in ok if float(row["knnGatedImprovementPx"]) < -5.0),
        "meanBaselineVertexErrorPx": _mean(_values(ok, "baselineVertexErrorPx")),
        "meanCandidateOracleVertexErrorPx": _mean(_values(ok, "candidateOracleVertexErrorPx")),
        "meanKnnTop1VertexErrorPx": _mean(_values(ok, "knnTop1VertexErrorPx")),
        "meanKnnGatedVertexErrorPx": _mean(_values(ok, "knnGatedVertexErrorPx")),
        "medianBaselineVertexErrorPx": _median(_values(ok, "baselineVertexErrorPx")),
        "medianCandidateOracleVertexErrorPx": _median(_values(ok, "candidateOracleVertexErrorPx")),
        "medianKnnTop1VertexErrorPx": _median(_values(ok, "knnTop1VertexErrorPx")),
        "medianKnnGatedVertexErrorPx": _median(_values(ok, "knnGatedVertexErrorPx")),
        "knnThresholdSweep": _knn_threshold_sweep(ok),
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    lines = [
        "# KNN Vertex Localizer V0",
        "",
        "Diagnostics-only artifact. This does not alter recognition behavior.",
        "",
        "This probe evaluates a leave-one-row-out k-nearest positive/negative prototype scorer over the same local candidate grid and features as the linear learned V0 localizer.",
        "",
        "## Summary",
        "",
        f"- Rows: {summary['rowCount']}",
        f"- Evaluated rows: {summary['evaluatedRowCount']}",
        f"- Axis-good rows: {summary['axisGoodRowCount']}",
        f"- Axis-blocked rows: {summary['axisBlockedRowCount']}",
        f"- Baseline strict/plausible: {summary['baselineStrictCount']} / {summary['baselinePlausibleCount']}",
        f"- Candidate-grid oracle strict/plausible: {summary['candidateOracleStrictCount']} / {summary['candidateOraclePlausibleCount']}",
        f"- KNN top-1 strict/plausible: {summary['knnTop1StrictCount']} / {summary['knnTop1PlausibleCount']}",
        f"- KNN gated strict/plausible: {summary['knnGatedStrictCount']} / {summary['knnGatedPlausibleCount']}",
        f"- Axis-good strict baseline/KNN-gated: {summary['axisGoodBaselineStrictCount']} / {summary['axisGoodKnnGatedStrictCount']}",
        f"- KNN gated accepted rows: {summary['knnGatedAcceptedCount']}",
        f"- KNN top-1 improved/worsened rows by >5px: {summary['knnTop1ImprovedRowCount']} / {summary['knnTop1WorsenedRowCount']}",
        f"- KNN gated improved/worsened rows by >5px: {summary['knnGatedImprovedRowCount']} / {summary['knnGatedWorsenedRowCount']}",
        f"- Mean vertex error baseline/oracle/top-1/gated: {_fmt(summary['meanBaselineVertexErrorPx'], 'px')} / {_fmt(summary['meanCandidateOracleVertexErrorPx'], 'px')} / {_fmt(summary['meanKnnTop1VertexErrorPx'], 'px')} / {_fmt(summary['meanKnnGatedVertexErrorPx'], 'px')}",
        f"- Median vertex error baseline/oracle/top-1/gated: {_fmt(summary['medianBaselineVertexErrorPx'], 'px')} / {_fmt(summary['medianCandidateOracleVertexErrorPx'], 'px')} / {_fmt(summary['medianKnnTop1VertexErrorPx'], 'px')} / {_fmt(summary['medianKnnGatedVertexErrorPx'], 'px')}",
        f"- Best non-empty low-worsen KNN gate: {_format_sweep(summary['knnThresholdSweep'].get('bestNonEmptyLowWorsen'))}",
        f"- Best non-empty zero-worsen KNN gate: {_format_sweep(summary['knnThresholdSweep'].get('bestNonEmptyZeroWorsen'))}",
        f"- Best non-empty KNN gate with <=2 worsens: {_format_sweep(summary['knnThresholdSweep'].get('bestNonEmptyAtMostTwoWorsens'))}",
        "",
        "## Rows",
        "",
        "| Row | Axis | Base | Oracle | KNN top-1 | Gated | Accepted | Delta | Score gain |",
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
            f"{_fmt(row.get('knnTop1VertexErrorPx'), 'px')} | "
            f"{_fmt(row.get('knnGatedVertexErrorPx'), 'px')} | "
            f"{'yes' if row.get('knnGatedAccepted') else 'no'} | "
            f"{_fmt(row.get('knnGatedImprovementPx'), 'px')} | "
            f"{_fmt_plain(row.get('knnScoreGain'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- KNN V0 is the first learned ranker to materially beat baseline on strict/plausible counts.",
            "- It is still not production-safe: the default confidence gate has one worsened accepted row and the broad top-1 policy has several.",
            "- The zero-worsen gate is useful evidence that confidence is possible, but it accepts too few rows to be a recognizer policy.",
            "- The next useful step is a real trainable localizer or more labeled rows, using this candidate-grid oracle and KNN gate as baselines.",
            "",
        ]
    )
    return "\n".join(lines)


def _knn_threshold_sweep(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for min_gain in [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        for min_score in [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0]:
            accepted = [
                row
                for row in rows
                if _row_passes_knn_gate(row, min_gain=min_gain, min_score=min_score)
            ]
            if not accepted:
                continue
            improved = sum(1 for row in accepted if float(row["knnTop1ImprovementPx"]) > 5.0)
            worsened = sum(1 for row in accepted if float(row["knnTop1ImprovementPx"]) < -5.0)
            errors = [
                float(row["knnTop1VertexErrorPx"])
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
            -item["plausibleCount"],
            -item["improvedCount"],
            item["worsenedCount"],
            item["meanVertexErrorPx"],
        ),
    )[0]


def _row_passes_knn_gate(row: Dict[str, Any], *, min_gain: float, min_score: float) -> bool:
    gain = row.get("knnScoreGain")
    score = row.get("knnScore")
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
        f"score>={thresholds.get('minScore')})"
    )


def _strip_private_candidate_rows(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if key != "_candidateRows"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    document = generate_knn_vertex_localizer_summary(feedback_path=args.feedback)
    _write_json(args.summary_out, document)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(document), encoding="utf-8")
    print(f"wrote {args.summary_out}")
    print(f"wrote {args.report_out}")
    print(
        f"KNN gated strict rows: {document['summary']['knnGatedStrictCount']} / "
        f"{document['summary']['evaluatedRowCount']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
