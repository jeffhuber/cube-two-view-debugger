#!/usr/bin/env python3
"""Label-driven vertex candidate ranker V0.

Diagnostics/data-only. This module does not alter recognizer behavior.

PR #189 showed that the true visible trihedral vertex is often present in an
expanded candidate source pool, but not near the top of each source's own
ranking. This tool tests whether simple rankers can move those candidates into
top-1/top-3/top-5 without using the row's human label at inference time.

The rankers are intentionally small and auditable:

* ``baseline_model_ranked`` preserves the pre-#189 model-ranked baseline.
* ``source_heuristic_v0`` is a fixed, hand-written source/feature score.
* ``leave_one_out_feature_prior_v0`` trains tiny source/feature priors from
  the other labeled rows, then evaluates the held-out row.
* ``combined_oracle`` is not a deployable ranker; it is the source-pool ceiling.

The human labels are used for diagnostics/evaluation only. No production
recognizer path imports this tool.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_hybrid_pipeline import _load_processing_image  # noqa: E402
from tools.global_cube_model_v0 import _round_point  # noqa: E402
from tools.interior_bezel_detection import detect_interior_bezel_lines  # noqa: E402
from tools.render_global_cube_model_v0_overlays import (  # noqa: E402
    _compute_rembg_mask,
    missing_required_optional_dependencies,
)
from tools.vertex_candidate_source_probe import (  # noqa: E402
    _bezel_axis_ray_candidates,
    _bezel_line_intersection_candidates,
    _dark_junction_grid_candidates,
    _dedupe_ranked,
    _distance_px,
    _model_local_grid_candidates,
    _model_ranked_candidates,
    _parse_point,
)


Point = Tuple[float, float]

DEFAULT_FEEDBACK = ROOT / "tests" / "fixtures" / "vertex_point_human_feedback.json"
DEFAULT_SUMMARY = (
    ROOT
    / "tests"
    / "fixtures"
    / "vertex_candidate_ranker_v0_easy_corpus_summary.json"
)
DEFAULT_REPORT = ROOT / "tools" / "VERTEX_CANDIDATE_RANKER_V0_REPORT.md"
DEFAULT_THRESHOLDS_PX = (10.0, 20.0)
SOURCE_ORDER = (
    "model_ranked",
    "bezel_line_intersection",
    "bezel_axis_ray",
    "model_local_grid",
    "dark_junction_grid",
)
POLICIES = (
    "baseline_model_ranked",
    "source_heuristic_v0",
    "leave_one_out_feature_prior_v0",
    "combined_oracle",
)


@dataclass(frozen=True)
class LabeledCandidateRow:
    set_id: str
    side: str
    evaluation_tier: str
    image_path: str
    human_vertex: Point
    candidates: Tuple[Dict[str, Any], ...]
    notes: str = ""

    @property
    def row_id(self) -> str:
        return f"{self.set_id}:{self.side}"


def generate_ranker_artifacts(
    feedback_document: Dict[str, Any],
    *,
    thresholds_px: Sequence[float] = DEFAULT_THRESHOLDS_PX,
) -> Dict[str, Any]:
    """Generate candidate pools, run ranker policies, and evaluate labels."""
    candidate_rows = [
        _candidate_row_from_feedback(row)
        for row in feedback_document.get("rows", [])
        if row.get("status") == "labeled"
    ]
    candidate_rows = [row for row in candidate_rows if row is not None]

    rows = [
        _evaluate_ranker_row(candidate_rows, index, thresholds_px)
        for index in range(len(candidate_rows))
    ]
    return {
        "schemaVersion": 1,
        "probe": "vertex_candidate_ranker_v0",
        "description": (
            "Diagnostics-only benchmark for ranking expanded visible-trihedral "
            "vertex candidate pools against human labels."
        ),
        "sourceFeedback": str(DEFAULT_FEEDBACK),
        "thresholdsPx": [float(value) for value in thresholds_px],
        "policies": list(POLICIES),
        "summary": summarize_ranker_rows(rows, thresholds_px),
        "rows": rows,
    }


def summarize_ranker_rows(
    rows: Sequence[Dict[str, Any]],
    thresholds_px: Sequence[float] = DEFAULT_THRESHOLDS_PX,
) -> Dict[str, Any]:
    policy_summaries = {
        policy: _summarize_policy(rows, policy, thresholds_px)
        for policy in POLICIES
    }
    return {
        "rowCount": len(rows),
        "errorRowCount": sum(1 for row in rows if row.get("notes")),
        "policySummaries": policy_summaries,
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    thresholds = [float(value) for value in document["thresholdsPx"]]
    primary = thresholds[0]
    secondary = thresholds[1] if len(thresholds) > 1 else thresholds[0]

    lines = [
        "# Vertex Candidate Ranker V0",
        "",
        "Diagnostics/data-only artifact. This does not alter recognition behavior.",
        "",
        "This report tests whether simple label-driven ranking policies can move the human-visible trihedral vertex from the expanded #189 source pool into top-1/top-3/top-5.",
        "",
        "## Summary",
        "",
        f"- Rows: {summary['rowCount']}",
        f"- Rows with notes/errors: {summary['errorRowCount']}",
        f"- Thresholds: {', '.join(f'{value:g}px' for value in thresholds)}",
        "",
        "## Policy Metrics",
        "",
        "| Policy | Rows | Mean pool | Top-1 @10 | Top-3 @10 | Top-5 @10 | Oracle @10 | Top-3 @20 | Top-5 @20 | Oracle @20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in document["policies"]:
        item = summary["policySummaries"][policy]
        lines.append(
            f"| `{policy}` | {item['rowCount']} | {item['meanCandidatePoolSize']:.1f} | "
            f"{item[f'top1HitCount@{primary:g}px']} | "
            f"{item[f'top3HitCount@{primary:g}px']} | "
            f"{item[f'top5HitCount@{primary:g}px']} | "
            f"{item[f'oracleHitCount@{primary:g}px']} | "
            f"{item[f'top3HitCount@{secondary:g}px']} | "
            f"{item[f'top5HitCount@{secondary:g}px']} | "
            f"{item[f'oracleHitCount@{secondary:g}px']} |"
        )

    lines.extend([
        "",
        "## Per-Row Readout",
        "",
        "| Set | Side | Pool | Best oracle | Baseline top3 | Heuristic top3 | Feature-prior top3 | Notes |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in document["rows"]:
        policies = row.get("policyResults", {})
        oracle = policies.get("combined_oracle", {})
        baseline = policies.get("baseline_model_ranked", {})
        heuristic = policies.get("source_heuristic_v0", {})
        feature_prior = policies.get("leave_one_out_feature_prior_v0", {})
        lines.append(
            f"| {row.get('setId')} | {row.get('side')} | {row.get('candidatePoolSize')} | "
            f"{oracle.get('bestDistancePx', '')} | "
            f"{baseline.get('top3DistancePx', '')} | "
            f"{heuristic.get('top3DistancePx', '')} | "
            f"{feature_prior.get('top3DistancePx', '')} | {row.get('notes', '')} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `baseline_model_ranked` is the #188/#189 starting point.",
        "- `source_heuristic_v0` is the best current deployable-shaped static score, but it remains far below the oracle ceiling.",
        "- `leave_one_out_feature_prior_v0` uses labels from other rows only; it is a sanity check against tiny-sample overfitting, not a production model.",
        "- The gap between ranker top-3 and `combined_oracle` means the next useful work is richer ranking signal, not production wiring.",
        "",
    ])
    return "\n".join(lines)


def _candidate_row_from_feedback(feedback_row: Dict[str, Any]) -> Optional[LabeledCandidateRow]:
    human = _parse_point(feedback_row.get("humanVertexPoint"))
    if human is None:
        return None
    notes: List[str] = []
    image_path = Path(str(feedback_row.get("imagePath") or ""))
    source_candidates: Dict[str, List[Dict[str, Any]]] = {
        "model_ranked": _model_ranked_candidates(feedback_row),
    }
    if image_path.exists():
        try:
            image, image_rgb = _load_processing_image(image_path)
            mask = _compute_rembg_mask(image)
            detection = detect_interior_bezel_lines(image_rgb, mask)
            source_candidates["bezel_line_intersection"] = _bezel_line_intersection_candidates(detection, mask)
            source_candidates["bezel_axis_ray"] = _bezel_axis_ray_candidates(detection, mask)
            source_candidates["model_local_grid"] = _model_local_grid_candidates(feedback_row, mask)
            source_candidates["dark_junction_grid"] = _dark_junction_grid_candidates(image_rgb, mask)
        except Exception as exc:  # pragma: no cover - local CLI/deps path
            notes.append(f"image probe error: {exc.__class__.__name__}: {exc}")
    else:
        notes.append("image missing")

    candidates = _flatten_sources(source_candidates)
    return LabeledCandidateRow(
        set_id=str(feedback_row.get("setId")),
        side=str(feedback_row.get("side")),
        evaluation_tier=str(feedback_row.get("evaluationTier")),
        image_path=str(feedback_row.get("imagePath")),
        human_vertex=human,
        candidates=tuple(candidates),
        notes="; ".join(notes),
    )


def _flatten_sources(source_candidates: Dict[str, Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for source in SOURCE_ORDER:
        min_distance = 0.0 if source == "model_ranked" else 4.0
        ranked = _dedupe_ranked(
            source_candidates.get(source, []),
            limit=80,
            min_distance_px=min_distance,
        )
        for rank, candidate in enumerate(ranked, start=1):
            flattened.append({
                "source": source,
                "sourceRank": rank,
                "point": candidate["point"],
                "sourceScore": float(candidate.get("score", 0.0)),
                "details": candidate.get("details", {}),
            })
    return flattened


def _evaluate_ranker_row(
    rows: Sequence[LabeledCandidateRow],
    index: int,
    thresholds_px: Sequence[float],
) -> Dict[str, Any]:
    row = rows[index]
    training_rows = [other for i, other in enumerate(rows) if i != index]
    feature_prior = _train_feature_prior(training_rows, positive_threshold_px=10.0)
    policies: Dict[str, List[Dict[str, Any]]] = {
        "baseline_model_ranked": _rank_baseline_model(row.candidates),
        "source_heuristic_v0": _rank_by_score(row.candidates, _source_heuristic_score),
        "leave_one_out_feature_prior_v0": _rank_by_score(
            row.candidates,
            lambda candidate: _feature_prior_score(candidate, feature_prior),
        ),
        "combined_oracle": _rank_by_score(
            row.candidates,
            lambda candidate: -_distance_px(row.human_vertex, candidate["point"]),
        ),
    }
    policy_results = {
        policy: _evaluate_ranked_candidates(ranked, row.human_vertex, thresholds_px)
        for policy, ranked in policies.items()
    }
    return {
        "setId": row.set_id,
        "side": row.side,
        "evaluationTier": row.evaluation_tier,
        "imagePath": row.image_path,
        "humanVertexPoint": _round_point(row.human_vertex),
        "candidatePoolSize": len(row.candidates),
        "sourceCounts": dict(collections.Counter(candidate["source"] for candidate in row.candidates)),
        "policyResults": policy_results,
        "notes": row.notes,
    }


def _rank_baseline_model(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        candidate
        for candidate in sorted(
            candidates,
            key=lambda item: item["sourceRank"],
        )
        if candidate["source"] == "model_ranked"
    ]


def _rank_by_score(
    candidates: Sequence[Dict[str, Any]],
    score_fn: Callable[[Dict[str, Any]], float],
) -> List[Dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda candidate: (score_fn(candidate), -candidate["sourceRank"]),
        reverse=True,
    )


def _source_heuristic_score(candidate: Dict[str, Any]) -> float:
    source = candidate["source"]
    details = candidate.get("details") or {}
    base = {
        "model_ranked": 2.8,
        "model_local_grid": 3.2,
        "bezel_axis_ray": 2.6,
        "dark_junction_grid": 1.4,
        "bezel_line_intersection": 1.0,
    }[source]
    if source == "model_ranked":
        return (
            base
            - 0.18 * float(details.get("baselineRank", candidate["sourceRank"]))
            + 0.05 * float(details.get("modelScore") or 0.0)
        )
    if source == "model_local_grid":
        offset = details.get("offset") or [0.0, 0.0]
        offset_distance = math.hypot(float(offset[0]), float(offset[1]))
        return (
            base
            - 0.035 * offset_distance
            - 0.12 * float(details.get("seedRank", 5))
        )
    if source == "bezel_axis_ray":
        return (
            base
            + 0.70 * float(details.get("lineQuality") or 0.0)
            - 0.004 * float(details.get("distanceFromDetectorPx") or 0.0)
        )
    if source == "dark_junction_grid":
        return (
            base
            + 0.80 * float(details.get("centerDarkness") or 0.0)
            + 0.60 * float(details.get("thirdSectorScore") or 0.0)
            - 0.04 * float(details.get("activeSectors") or 0.0)
        )
    return base + 0.30 * float(details.get("qualitySum") or 0.0)


def _train_feature_prior(
    rows: Sequence[LabeledCandidateRow],
    *,
    positive_threshold_px: float,
) -> Dict[str, Any]:
    counts: collections.Counter[Tuple[str, Any]] = collections.Counter()
    positives: collections.Counter[Tuple[str, Any]] = collections.Counter()
    source_counts: collections.Counter[str] = collections.Counter()
    source_positives: collections.Counter[str] = collections.Counter()
    for row in rows:
        for candidate in row.candidates:
            positive = _distance_px(row.human_vertex, candidate["point"]) <= positive_threshold_px
            source = str(candidate["source"])
            source_counts[source] += 1
            source_positives[source] += int(positive)
            for key in _feature_keys(candidate):
                counts[key] += 1
                positives[key] += int(positive)
    return {
        "counts": counts,
        "positives": positives,
        "sourceCounts": source_counts,
        "sourcePositives": source_positives,
    }


def _feature_prior_score(candidate: Dict[str, Any], prior: Dict[str, Any]) -> float:
    source = str(candidate["source"])
    source_rate = _smoothed_rate(
        prior["sourcePositives"][source],
        prior["sourceCounts"][source],
        prior_positive=0.20,
        strength=3.0,
    )
    feature_rates = [
        _smoothed_rate(
            prior["positives"][key],
            prior["counts"][key],
            prior_positive=0.15,
            strength=2.0,
        )
        for key in _feature_keys(candidate)
    ]
    feature_rate = sum(feature_rates) / max(1, len(feature_rates))
    details = candidate.get("details") or {}
    score = 1.20 * source_rate + 1.50 * feature_rate
    score += 0.15 * _smoothed_rate(
        prior["positives"][("source_rank_bin", min(candidate["sourceRank"], 10))],
        prior["counts"][("source_rank_bin", min(candidate["sourceRank"], 10))],
        prior_positive=0.12,
        strength=2.0,
    )
    if source == "model_local_grid":
        offset = details.get("offset") or [0.0, 0.0]
        score -= 0.0015 * math.hypot(float(offset[0]), float(offset[1]))
        score -= 0.02 * (float(details.get("seedRank", 5)) - 1.0)
    elif source == "bezel_axis_ray":
        score += 0.10 * float(details.get("lineQuality") or 0.0)
        score -= 0.0005 * float(details.get("distanceFromDetectorPx") or 0.0)
    elif source == "model_ranked":
        score -= 0.03 * (float(details.get("baselineRank", candidate["sourceRank"])) - 1.0)
    elif source == "dark_junction_grid":
        score += 0.03 * float(details.get("thirdSectorScore") or 0.0)
        score -= 0.003 * float(details.get("activeSectors") or 0.0)
    return float(score)


def _feature_keys(candidate: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    source = str(candidate["source"])
    details = candidate.get("details") or {}
    keys: List[Tuple[str, Any]] = [
        ("source", source),
        ("source_rank_bin", min(int(candidate["sourceRank"]), 10)),
    ]
    if source == "model_local_grid":
        offset = details.get("offset") or [0.0, 0.0]
        keys.extend([
            ("local_offset", (float(offset[0]), float(offset[1]))),
            ("local_seed", int(details.get("seedRank", 99))),
            ("local_offset_abs", (abs(float(offset[0])), abs(float(offset[1])))),
        ])
    elif source == "model_ranked":
        keys.append(("model_baseline_rank", int(details.get("baselineRank", candidate["sourceRank"]))))
    elif source == "bezel_axis_ray":
        keys.extend([
            ("axis_line", int(details.get("lineIndex", -1))),
            ("axis_dist", float(details.get("distanceFromDetectorPx") or 0.0)),
            ("axis_sign", int(details.get("sign", 0))),
        ])
    elif source == "dark_junction_grid":
        keys.append(("dark_active", min(int(details.get("activeSectors", 0)), 9)))
    elif source == "bezel_line_intersection":
        keys.append(("line_pair", tuple(details.get("lineIndexes") or [])))
    return tuple(keys)


def _evaluate_ranked_candidates(
    ranked: Sequence[Dict[str, Any]],
    human_vertex: Point,
    thresholds_px: Sequence[float],
) -> Dict[str, Any]:
    candidates = []
    for rank, candidate in enumerate(ranked, start=1):
        point = candidate["point"]
        candidates.append({
            "rank": rank,
            "source": candidate["source"],
            "sourceRank": candidate["sourceRank"],
            "point": _round_point(point),
            "distancePx": round(_distance_px(human_vertex, point), 2),
            "details": candidate.get("details", {}),
        })
    best = min(candidates, key=lambda item: item["distancePx"], default=None)
    result: Dict[str, Any] = {
        "candidateCount": len(candidates),
        "topCandidates": candidates[:5],
        "bestCandidate": best,
        "bestDistancePx": best["distancePx"] if best else None,
        "top1DistancePx": candidates[0]["distancePx"] if candidates else None,
        "top3DistancePx": _min_distance(candidates[:3]),
        "top5DistancePx": _min_distance(candidates[:5]),
    }
    for threshold in thresholds_px:
        label = f"@{threshold:g}px"
        result[f"top1Hit{label}"] = any(item["distancePx"] <= threshold for item in candidates[:1])
        result[f"top3Hit{label}"] = any(item["distancePx"] <= threshold for item in candidates[:3])
        result[f"top5Hit{label}"] = any(item["distancePx"] <= threshold for item in candidates[:5])
        result[f"oracleHit{label}"] = best is not None and best["distancePx"] <= threshold
    return result


def _summarize_policy(
    rows: Sequence[Dict[str, Any]],
    policy: str,
    thresholds_px: Sequence[float],
) -> Dict[str, Any]:
    results = [
        row.get("policyResults", {}).get(policy, {})
        for row in rows
        if policy in row.get("policyResults", {})
    ]
    pool_sizes = [int(row.get("candidatePoolSize") or 0) for row in rows]
    summary: Dict[str, Any] = {
        "rowCount": len(results),
        "meanCandidatePoolSize": (
            round(sum(pool_sizes) / len(pool_sizes), 2) if pool_sizes else 0.0
        ),
    }
    for threshold in thresholds_px:
        label = f"@{threshold:g}px"
        summary[f"top1HitCount{label}"] = sum(1 for item in results if item.get(f"top1Hit{label}"))
        summary[f"top3HitCount{label}"] = sum(1 for item in results if item.get(f"top3Hit{label}"))
        summary[f"top5HitCount{label}"] = sum(1 for item in results if item.get(f"top5Hit{label}"))
        summary[f"oracleHitCount{label}"] = sum(1 for item in results if item.get(f"oracleHit{label}"))
    return summary


def _min_distance(candidates: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not candidates:
        return None
    return min(float(candidate["distancePx"]) for candidate in candidates)


def _smoothed_rate(
    positive_count: float,
    total_count: float,
    *,
    prior_positive: float,
    strength: float,
) -> float:
    return (positive_count + prior_positive * strength) / max(strength, total_count + strength)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--thresholds-px",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS_PX),
        help="Distance thresholds for hit metrics.",
    )
    args = parser.parse_args(argv)

    missing = missing_required_optional_dependencies()
    if missing:
        deps = ", ".join(missing)
        print(
            "error: vertex_candidate_ranker_v0.py requires optional diagnostic "
            f"dependencies to regenerate outputs: {deps}.\n"
            "Install them in the repo venv, for example:\n"
            "  .venv/bin/pip install rembg scipy onnxruntime",
            file=sys.stderr,
        )
        return 2

    feedback = _read_json(args.feedback)
    document = generate_ranker_artifacts(feedback, thresholds_px=args.thresholds_px)
    _write_json(args.summary, document)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(document), encoding="utf-8")
    print(f"wrote {args.summary}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
