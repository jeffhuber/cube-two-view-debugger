#!/usr/bin/env python3
"""Label-driven vertex candidate source probe.

Diagnostics/data-only. This module does not alter recognizer behavior.

PR #188 showed that the existing coherent-model ranking usually does not put
the human-visible trihedral vertex in the top candidates. This probe keeps
that model-ranked baseline intact and evaluates additional candidate-source
families against the same human labels:

* model-ranked candidates already committed in the feedback fixture
* pairwise intersections of detected bezel-line equations
* rays stepped outward along detected bezel axes
* local grids around existing model-ranked candidates
* dark-junction candidates from black-plastic evidence inside the silhouette

The goal is source recall, not production selection. A source with high oracle
recall but poor top-3 ranking is still useful: it tells us where the next
ranking/fitter iteration should look.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_hybrid_pipeline import _load_processing_image  # noqa: E402
from tools.global_cube_model_v0 import _round_point, _unit_from_angle  # noqa: E402
from tools.interior_bezel_detection import (  # noqa: E402
    InteriorBezelDetection,
    _rgb_to_gray,
    detect_interior_bezel_lines,
)
from tools.render_global_cube_model_v0_overlays import (  # noqa: E402
    _compute_rembg_mask,
    missing_required_optional_dependencies,
)


Point = Tuple[float, float]

DEFAULT_FEEDBACK = ROOT / "tests" / "fixtures" / "vertex_point_human_feedback.json"
DEFAULT_SUMMARY = (
    ROOT
    / "tests"
    / "fixtures"
    / "vertex_candidate_source_probe_easy_corpus_summary.json"
)
DEFAULT_REPORT = ROOT / "tools" / "VERTEX_CANDIDATE_SOURCE_PROBE_REPORT.md"
DEFAULT_THRESHOLDS_PX = (10.0, 20.0)

SOURCE_ORDER = (
    "model_ranked",
    "bezel_line_intersection",
    "bezel_axis_ray",
    "model_local_grid",
    "dark_junction_grid",
)


def generate_source_probe_artifacts(
    feedback_document: Dict[str, Any],
    *,
    thresholds_px: Sequence[float] = DEFAULT_THRESHOLDS_PX,
) -> Dict[str, Any]:
    """Generate and evaluate candidate-source families for labeled rows."""
    rows: List[Dict[str, Any]] = []
    for feedback_row in feedback_document.get("rows", []):
        rows.append(_evaluate_feedback_row_sources(feedback_row, thresholds_px))

    return {
        "schemaVersion": 1,
        "probe": "vertex_candidate_source_probe_v0",
        "description": (
            "Diagnostics-only source-recall probe for visible trihedral vertex "
            "candidate generation. Human labels are used only for evaluation."
        ),
        "sourceFeedback": str(DEFAULT_FEEDBACK),
        "thresholdsPx": [float(value) for value in thresholds_px],
        "sourceOrder": list(SOURCE_ORDER),
        "summary": summarize_source_rows(rows, thresholds_px),
        "rows": rows,
    }


def summarize_source_rows(
    rows: Sequence[Dict[str, Any]],
    thresholds_px: Sequence[float] = DEFAULT_THRESHOLDS_PX,
) -> Dict[str, Any]:
    labeled_rows = [row for row in rows if row.get("evaluationStatus") == "labeled"]
    source_summaries = {
        source: _summarize_one_source(labeled_rows, source, thresholds_px)
        for source in SOURCE_ORDER
    }
    oracle = _summarize_oracle(labeled_rows, thresholds_px)
    return {
        "rowCount": len(rows),
        "labeledRowCount": len(labeled_rows),
        "errorRowCount": sum(1 for row in rows if row.get("evaluationStatus") == "error"),
        "sourceSummaries": source_summaries,
        "combinedOracle": oracle,
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    thresholds = [float(value) for value in document["thresholdsPx"]]
    primary = thresholds[0]
    secondary = thresholds[1] if len(thresholds) > 1 else thresholds[0]

    lines = [
        "# Vertex Candidate Source Probe",
        "",
        "Diagnostics/data-only artifact. This does not alter recognition behavior.",
        "",
        "This report evaluates whether new candidate-source families contain the human-visible trihedral vertex. The current model-ranked candidates remain as the baseline.",
        "",
        "## Summary",
        "",
        f"- Rows: {summary['rowCount']}",
        f"- Labeled rows: {summary['labeledRowCount']}",
        f"- Error rows: {summary['errorRowCount']}",
        f"- Thresholds: {', '.join(f'{value:g}px' for value in thresholds)}",
        "",
        "## Source Recall",
        "",
        "| Source | Rows with candidates | Mean candidates | Top-1 @10 | Top-3 @10 | Top-5 @10 | Oracle @10 | Top-3 @20 | Top-5 @20 | Oracle @20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in document["sourceOrder"]:
        item = summary["sourceSummaries"][source]
        lines.append(
            f"| `{source}` | {item['rowsWithCandidates']} | {item['meanCandidateCount']:.1f} | "
            f"{item[f'top1HitCount@{primary:g}px']} | "
            f"{item[f'top3HitCount@{primary:g}px']} | "
            f"{item[f'top5HitCount@{primary:g}px']} | "
            f"{item[f'oracleHitCount@{primary:g}px']} | "
            f"{item[f'top3HitCount@{secondary:g}px']} | "
            f"{item[f'top5HitCount@{secondary:g}px']} | "
            f"{item[f'oracleHitCount@{secondary:g}px']} |"
        )

    oracle = summary["combinedOracle"]
    lines.extend([
        "",
        "## Combined Oracle",
        "",
        f"- Best source within {primary:g}px: {oracle[f'oracleHitCount@{primary:g}px']} / {summary['labeledRowCount']}",
        f"- Best source within {secondary:g}px: {oracle[f'oracleHitCount@{secondary:g}px']} / {summary['labeledRowCount']}",
        "",
        "## Per-Row Best Source",
        "",
        "| Set | Side | Status | Best source | Best dist | Best rank | Top source @10 | Source notes |",
        "|---:|---|---|---|---:|---:|---|---|",
    ])
    for row in document["rows"]:
        best = row.get("combinedBest") or {}
        source_hits = [
            source["source"]
            for source in row.get("sources", [])
            if source.get(f"oracleWithin{primary:g}px")
        ]
        notes = row.get("notes") or ""
        lines.append(
            f"| {row.get('setId')} | {row.get('side', '')} | `{row.get('evaluationStatus')}` | "
            f"`{best.get('source', '')}` | {best.get('distancePx', '')} | "
            f"{best.get('rank', '')} | {', '.join(f'`{source}`' for source in source_hits)} | "
            f"{notes} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `Top-N` columns use each source's own deterministic ranking; low top-N with high oracle means the source sees the neighborhood but still needs a better ranking policy.",
        "- `Oracle` columns ignore ranking and ask whether the source contains any candidate within the threshold.",
        "- The next geometry step should only use a source for model fitting once source top-3 recall is strong on easy rows, or once a downstream fitter can safely consume larger candidate sets.",
        "",
    ])
    return "\n".join(lines)


def _evaluate_feedback_row_sources(
    feedback_row: Dict[str, Any],
    thresholds_px: Sequence[float],
) -> Dict[str, Any]:
    base = {
        "setId": str(feedback_row.get("setId")),
        "side": feedback_row.get("side"),
        "evaluationTier": feedback_row.get("evaluationTier"),
        "imagePath": feedback_row.get("imagePath"),
        "labelStatus": feedback_row.get("status"),
    }
    if feedback_row.get("status") != "labeled":
        return {**base, "evaluationStatus": str(feedback_row.get("status") or "unlabeled")}

    human = _parse_point(feedback_row.get("humanVertexPoint"))
    if human is None:
        return {
            **base,
            "evaluationStatus": "error",
            "error": "labeled row is missing humanVertexPoint=[x,y]",
        }

    source_candidates: Dict[str, List[Dict[str, Any]]] = {
        "model_ranked": _model_ranked_candidates(feedback_row),
    }
    image_path = Path(str(feedback_row.get("imagePath") or ""))
    notes: List[str] = []
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
            for source in SOURCE_ORDER:
                source_candidates.setdefault(source, [])
    else:
        notes.append("image missing")
        for source in SOURCE_ORDER:
            source_candidates.setdefault(source, [])

    sources = [
        _evaluate_source(source, source_candidates.get(source, []), human, thresholds_px)
        for source in SOURCE_ORDER
    ]
    combined_best = _combined_best_candidate(sources)
    return {
        **base,
        "evaluationStatus": "labeled",
        "humanVertexPoint": _round_point(human),
        "candidateStatus": feedback_row.get("candidateStatus"),
        "sources": sources,
        "combinedBest": combined_best,
        "notes": "; ".join(notes),
    }


def _model_ranked_candidates(feedback_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = []
    for candidate in feedback_row.get("topCandidates", []):
        point = _parse_point(candidate.get("vertexPoint"))
        if point is None:
            continue
        rank = int(candidate.get("rank") or len(candidates) + 1)
        candidates.append({
            "source": "model_ranked",
            "point": point,
            "score": float(1000 - rank),
            "details": {
                "baselineRank": rank,
                "baselineSource": candidate.get("source"),
                "modelScore": candidate.get("modelScore"),
            },
        })
    return candidates


def _bezel_line_intersection_candidates(
    detection: InteriorBezelDetection,
    mask: np.ndarray,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    equations = detection.line_equations[:3]
    qualities = detection.line_qualities[:3]
    for i in range(len(equations)):
        for j in range(i + 1, len(equations)):
            point = _line_intersection(equations[i], equations[j])
            if point is None or not _inside_mask(point, mask):
                continue
            quality = float(qualities[i] if i < len(qualities) else 0.0) + float(
                qualities[j] if j < len(qualities) else 0.0
            )
            candidates.append({
                "source": "bezel_line_intersection",
                "point": point,
                "score": quality,
                "details": {"lineIndexes": [i, j], "qualitySum": round(quality, 4)},
            })
    return _dedupe_ranked(candidates, limit=8, min_distance_px=4.0)


def _bezel_axis_ray_candidates(
    detection: InteriorBezelDetection,
    mask: np.ndarray,
) -> List[Dict[str, Any]]:
    if detection.cube_center is None:
        return []
    center = (float(detection.cube_center[0]), float(detection.cube_center[1]))
    distances = (32.0, 56.0, 80.0, 112.0, 144.0, 176.0, 216.0, 256.0)
    candidates: List[Dict[str, Any]] = []
    for line_idx, angle in enumerate(detection.boundary_angles[:3]):
        unit = _unit_from_angle(angle)
        quality = float(detection.line_qualities[line_idx]) if line_idx < len(detection.line_qualities) else 0.0
        for sign in (-1.0, 1.0):
            for distance in distances:
                point = (
                    center[0] + sign * unit[0] * distance,
                    center[1] + sign * unit[1] * distance,
                )
                if not _inside_mask(point, mask):
                    continue
                candidates.append({
                    "source": "bezel_axis_ray",
                    "point": point,
                    "score": quality - (distance / 1000.0),
                    "details": {
                        "lineIndex": line_idx,
                        "sign": int(sign),
                        "distanceFromDetectorPx": distance,
                        "lineQuality": round(quality, 4),
                    },
                })
    return _dedupe_ranked(candidates, limit=36, min_distance_px=10.0)


def _model_local_grid_candidates(
    feedback_row: Dict[str, Any],
    mask: np.ndarray,
) -> List[Dict[str, Any]]:
    offsets = (-48.0, -32.0, -16.0, 0.0, 16.0, 32.0, 48.0)
    candidates: List[Dict[str, Any]] = []
    for candidate in feedback_row.get("topCandidates", [])[:5]:
        seed = _parse_point(candidate.get("vertexPoint"))
        if seed is None:
            continue
        baseline_rank = int(candidate.get("rank") or 99)
        model_score = float(candidate.get("modelScore") or 0.0)
        for dx in offsets:
            for dy in offsets:
                point = (seed[0] + dx, seed[1] + dy)
                if not _inside_mask(point, mask):
                    continue
                offset_distance = math.hypot(dx, dy)
                candidates.append({
                    "source": "model_local_grid",
                    "point": point,
                    "score": model_score - baseline_rank * 0.05 - offset_distance / 100.0,
                    "details": {
                        "seedRank": baseline_rank,
                        "offset": _round_point((dx, dy)),
                    },
                })
    return _dedupe_ranked(candidates, limit=40, min_distance_px=10.0)


def _dark_junction_grid_candidates(
    image_rgb: np.ndarray,
    mask: np.ndarray,
) -> List[Dict[str, Any]]:
    dark = _darkness_image(image_rgb, mask)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    candidates: List[Dict[str, Any]] = []
    step = 14
    for y in range(int(ys.min()), int(ys.max()) + 1, step):
        for x in range(int(xs.min()), int(xs.max()) + 1, step):
            if not bool(mask[y, x]):
                continue
            score, details = _dark_junction_score(dark, mask, float(x), float(y))
            if score < 0.54:
                continue
            candidates.append({
                "source": "dark_junction_grid",
                "point": (float(x), float(y)),
                "score": score,
                "details": details,
            })
    return _dedupe_ranked(candidates, limit=40, min_distance_px=18.0)


def _evaluate_source(
    source: str,
    candidates: Sequence[Dict[str, Any]],
    human: Point,
    thresholds_px: Sequence[float],
) -> Dict[str, Any]:
    min_distance = 0.0 if source == "model_ranked" else 4.0
    ranked = _dedupe_ranked(candidates, limit=80, min_distance_px=min_distance)
    distances = []
    for rank, candidate in enumerate(ranked, start=1):
        point = candidate["point"]
        distance = _distance_px(human, point)
        distances.append({
            "rank": rank,
            "point": _round_point(point),
            "score": round(float(candidate.get("score", 0.0)), 4),
            "distancePx": round(distance, 2),
            "details": candidate.get("details", {}),
        })
    best = min(distances, key=lambda item: float(item["distancePx"]), default=None)
    evaluated = {
        "source": source,
        "candidateCount": len(distances),
        "topCandidates": distances[:5],
        "bestCandidate": best,
    }
    for threshold in thresholds_px:
        suffix = _threshold_suffix(threshold)
        top1 = distances[:1]
        top3 = distances[:3]
        top5 = distances[:5]
        evaluated[f"top1Within{suffix}"] = any(item["distancePx"] <= threshold for item in top1)
        evaluated[f"top3Within{suffix}"] = any(item["distancePx"] <= threshold for item in top3)
        evaluated[f"top5Within{suffix}"] = any(item["distancePx"] <= threshold for item in top5)
        evaluated[f"oracleWithin{suffix}"] = best is not None and best["distancePx"] <= threshold
    return evaluated


def _summarize_one_source(
    rows: Sequence[Dict[str, Any]],
    source: str,
    thresholds_px: Sequence[float],
) -> Dict[str, Any]:
    source_rows = [
        next((item for item in row.get("sources", []) if item.get("source") == source), None)
        for row in rows
    ]
    source_rows = [item for item in source_rows if item is not None]
    candidate_counts = [int(item.get("candidateCount") or 0) for item in source_rows]
    summary: Dict[str, Any] = {
        "rowCount": len(source_rows),
        "rowsWithCandidates": sum(1 for count in candidate_counts if count > 0),
        "meanCandidateCount": (
            round(sum(candidate_counts) / len(candidate_counts), 2)
            if candidate_counts
            else 0.0
        ),
    }
    for threshold in thresholds_px:
        suffix = _threshold_suffix(threshold)
        label = f"@{threshold:g}px"
        summary[f"top1HitCount{label}"] = sum(1 for item in source_rows if item.get(f"top1Within{suffix}"))
        summary[f"top3HitCount{label}"] = sum(1 for item in source_rows if item.get(f"top3Within{suffix}"))
        summary[f"top5HitCount{label}"] = sum(1 for item in source_rows if item.get(f"top5Within{suffix}"))
        summary[f"oracleHitCount{label}"] = sum(1 for item in source_rows if item.get(f"oracleWithin{suffix}"))
    return summary


def _summarize_oracle(
    rows: Sequence[Dict[str, Any]],
    thresholds_px: Sequence[float],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for threshold in thresholds_px:
        summary[f"oracleHitCount@{threshold:g}px"] = sum(
            1
            for row in rows
            if (row.get("combinedBest") or {}).get("distancePx", float("inf")) <= threshold
        )
    return summary


def _combined_best_candidate(sources: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for source in sources:
        candidate = source.get("bestCandidate")
        if not candidate:
            continue
        item = {
            "source": source.get("source"),
            "rank": candidate.get("rank"),
            "point": candidate.get("point"),
            "distancePx": candidate.get("distancePx"),
            "score": candidate.get("score"),
        }
        if best is None or float(item["distancePx"]) < float(best["distancePx"]):
            best = item
    return best


def _line_intersection(
    left: Tuple[float, float, float],
    right: Tuple[float, float, float],
) -> Optional[Point]:
    a1, b1, c1 = left
    a2, b2, c2 = right
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return (float(x), float(y))


def _darkness_image(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = _rgb_to_gray(image_rgb).astype(np.float32)
    masked = gray[mask.astype(bool)]
    if len(masked) == 0:
        return np.zeros_like(gray, dtype=np.float32)
    low = float(np.percentile(masked, 2))
    high = float(np.percentile(masked, 45))
    return np.clip((high - gray) / max(1.0, high - low), 0.0, 1.0).astype(np.float32)


def _dark_junction_score(
    dark: np.ndarray,
    mask: np.ndarray,
    x: float,
    y: float,
) -> Tuple[float, Dict[str, Any]]:
    height, width = mask.shape
    ix = int(round(x))
    iy = int(round(y))
    if not (0 <= ix < width and 0 <= iy < height and bool(mask[iy, ix])):
        return (-1.0, {})

    center_values = []
    for yy in range(iy - 4, iy + 5):
        for xx in range(ix - 4, ix + 5):
            if (
                0 <= xx < width
                and 0 <= yy < height
                and (xx - ix) ** 2 + (yy - iy) ** 2 <= 16
                and bool(mask[yy, xx])
            ):
                center_values.append(float(dark[yy, xx]))
    center_darkness = float(np.mean(center_values)) if center_values else 0.0

    sector_scores: List[float] = []
    for sector in range(18):
        theta = 2.0 * math.pi * sector / 18.0
        ray = []
        for radius in range(8, 76, 4):
            xx = int(round(x + math.cos(theta) * radius))
            yy = int(round(y + math.sin(theta) * radius))
            if 0 <= xx < width and 0 <= yy < height and bool(mask[yy, xx]):
                ray.append(float(dark[yy, xx]))
        if len(ray) < 5:
            sector_scores.append(0.0)
            continue
        ray.sort(reverse=True)
        sector_scores.append(float(np.mean(ray[: max(1, len(ray) // 4)])))

    top = sorted(sector_scores, reverse=True)
    if len(top) < 3:
        return (center_darkness * 0.45, {"centerDarkness": round(center_darkness, 4)})
    sector_mean = float(sum(top[:3]) / 3.0)
    balance = float(top[2])
    active_sectors = sum(1 for value in sector_scores if value >= 0.55)
    overfill_penalty = max(0, active_sectors - 7) * 0.04
    score = (
        0.45 * center_darkness
        + 0.35 * sector_mean
        + 0.25 * balance
        - overfill_penalty
    )
    return (
        float(score),
        {
            "centerDarkness": round(center_darkness, 4),
            "top3SectorMean": round(sector_mean, 4),
            "thirdSectorScore": round(balance, 4),
            "activeSectors": active_sectors,
        },
    )


def _dedupe_ranked(
    candidates: Sequence[Dict[str, Any]],
    *,
    limit: int,
    min_distance_px: float,
) -> List[Dict[str, Any]]:
    ranked = sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    result: List[Dict[str, Any]] = []
    seen = set()
    for candidate in ranked:
        point = _parse_point(candidate.get("point"))
        if point is None:
            continue
        key = (round(point[0], 3), round(point[1], 3))
        if key in seen:
            continue
        if any(_distance_px(point, existing["point"]) < min_distance_px for existing in result):
            continue
        seen.add(key)
        result.append({**candidate, "point": point})
        if len(result) >= limit:
            break
    return result


def _inside_mask(point: Point, mask: np.ndarray) -> bool:
    height, width = mask.shape
    ix = int(round(point[0]))
    iy = int(round(point[1]))
    return 0 <= ix < width and 0 <= iy < height and bool(mask[iy, ix])


def _parse_point(value: Any) -> Optional[Point]:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return (float(value[0]), float(value[1]))
    return None


def _distance_px(left: Point, right: Point) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _threshold_suffix(threshold: float) -> str:
    return f"{threshold:g}px"


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
            "error: vertex_candidate_source_probe.py requires optional "
            f"diagnostic dependencies to regenerate outputs: {deps}.\n"
            "Install them in the repo venv, for example:\n"
            "  .venv/bin/pip install rembg scipy onnxruntime",
            file=sys.stderr,
        )
        return 2

    feedback = _read_json(args.feedback)
    document = generate_source_probe_artifacts(
        feedback,
        thresholds_px=args.thresholds_px,
    )
    _write_json(args.summary, document)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(document), encoding="utf-8")
    print(f"wrote {args.summary}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
