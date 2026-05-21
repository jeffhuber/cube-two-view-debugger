#!/usr/bin/env python3
"""Fitter-assisted visible-vertex candidate selection V0.

Diagnostics/data-only. This module does not alter recognizer behavior.

PR #189 showed that the human-visible trihedral vertex is often present in the
expanded source pool. PR #190 showed that simple source/feature rankers do not
select it reliably. This tool asks the next question: if every candidate is
allowed to seed a coherent projected cube model, can model fit quality select
the true vertex?

The answer is evaluated against human labels only after ranking. Labels are
not used by the fitter-assisted policies.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cube_mesh_anchor_fitter_v0 import (  # noqa: E402
    _anchors_from_model,
    _score_anchor_mesh,
)
from tools.evaluate_hybrid_pipeline import _load_processing_image  # noqa: E402
from tools.global_cube_model_v0 import (  # noqa: E402
    ProjectedCubeModel,
    _edge_search_range,
    _model_from_axes,
    _round_point,
    _status_from_components,
    _unit_from_angle,
)
from tools.interior_bezel_detection import (  # noqa: E402
    InteriorBezelDetection,
    detect_interior_bezel_lines,
)
from tools.render_global_cube_model_v0_overlays import (  # noqa: E402
    _compute_rembg_mask,
    missing_required_optional_dependencies,
)
from tools.vertex_candidate_ranker_v0 import (  # noqa: E402
    _evaluate_ranked_candidates,
    _flatten_sources,
    _rank_baseline_model,
    _rank_by_score,
    _source_heuristic_score,
)
from tools.vertex_candidate_source_probe import (  # noqa: E402
    _bezel_axis_ray_candidates,
    _bezel_line_intersection_candidates,
    _dark_junction_grid_candidates,
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
    / "vertex_fitter_assisted_ranker_v0_easy_corpus_summary.json"
)
DEFAULT_REPORT = ROOT / "tools" / "VERTEX_FITTER_ASSISTED_RANKER_V0_REPORT.md"
DEFAULT_OUTPUT_DIR = Path("/tmp/vertex_fitter_assisted_ranker_v0_overlays")
DEFAULT_THRESHOLDS_PX = (10.0, 20.0)
DEFAULT_EDGE_STEPS = 10
DEFAULT_SCORING_MAX_DIM = 360
POLICIES = (
    "baseline_model_ranked",
    "source_heuristic_v0",
    "fitter_model_score_v0",
    "fitter_assisted_v0",
    "combined_oracle",
)


def generate_fitter_assisted_artifacts(
    feedback_document: Dict[str, Any],
    *,
    thresholds_px: Sequence[float] = DEFAULT_THRESHOLDS_PX,
    edge_steps: int = DEFAULT_EDGE_STEPS,
    scoring_max_dim: int = DEFAULT_SCORING_MAX_DIM,
    output_dir: Optional[Path] = DEFAULT_OUTPUT_DIR,
    max_candidates_per_row: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate fitter-assisted ranker diagnostics for labeled rows."""
    rows = [
        _evaluate_feedback_row(
            row,
            thresholds_px=thresholds_px,
            edge_steps=edge_steps,
            scoring_max_dim=scoring_max_dim,
            output_dir=output_dir,
            max_candidates_per_row=max_candidates_per_row,
        )
        for row in feedback_document.get("rows", [])
        if row.get("status") == "labeled"
    ]
    return {
        "schemaVersion": 1,
        "probe": "vertex_fitter_assisted_ranker_v0",
        "description": (
            "Diagnostics-only fitter-assisted ranking of expanded visible "
            "trihedral vertex candidate pools against human labels."
        ),
        "sourceFeedback": str(DEFAULT_FEEDBACK),
        "thresholdsPx": [float(value) for value in thresholds_px],
        "config": {
            "edgeSteps": int(edge_steps),
            "scoringMaxDim": int(scoring_max_dim),
            "outputDir": str(output_dir) if output_dir is not None else None,
            "maxCandidatesPerRow": max_candidates_per_row,
        },
        "policies": list(POLICIES),
        "summary": summarize_fitter_rows(rows, thresholds_px),
        "rows": rows,
    }


def summarize_fitter_rows(
    rows: Sequence[Dict[str, Any]],
    thresholds_px: Sequence[float] = DEFAULT_THRESHOLDS_PX,
) -> Dict[str, Any]:
    labeled_rows = [row for row in rows if row.get("evaluationStatus") == "labeled"]
    policy_summaries = {
        policy: _summarize_policy(labeled_rows, policy, thresholds_px)
        for policy in POLICIES
    }
    return {
        "rowCount": len(rows),
        "labeledRowCount": len(labeled_rows),
        "errorRowCount": sum(1 for row in rows if row.get("evaluationStatus") != "labeled"),
        "meanCandidatePoolSize": _mean(row.get("candidatePoolSize", 0) for row in labeled_rows),
        "meanFittedCandidateCount": _mean(row.get("fittedCandidateCount", 0) for row in labeled_rows),
        "policySummaries": policy_summaries,
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    thresholds = [float(value) for value in document["thresholdsPx"]]
    primary = thresholds[0]
    secondary = thresholds[1] if len(thresholds) > 1 else thresholds[0]

    lines = [
        "# Vertex Fitter-Assisted Ranker V0",
        "",
        "Diagnostics/data-only artifact. This does not alter recognition behavior.",
        "",
        "This report tests whether coherent projected-cube fit quality can select the human-visible trihedral vertex from the expanded #189 candidate pool.",
        "",
        "## Headline",
        "",
        _headline(summary, primary),
        "",
        "## Summary",
        "",
        f"- Rows: {summary['rowCount']}",
        f"- Labeled rows: {summary['labeledRowCount']}",
        f"- Error rows: {summary['errorRowCount']}",
        f"- Mean candidate pool: {summary['meanCandidatePoolSize']:.1f}",
        f"- Mean fitted candidates: {summary['meanFittedCandidateCount']:.1f}",
        f"- Thresholds: {', '.join(f'{value:g}px' for value in thresholds)}",
        f"- Edge steps per candidate: {document['config']['edgeSteps']}",
        f"- Scoring max dimension: {document['config']['scoringMaxDim']}px",
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
        "| Set | Side | Pool | Fitted | Best oracle | Baseline top3 | Fitter top3 | Assisted top3 | Assisted top source | Overlay | Notes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ])
    for row in document["rows"]:
        policies = row.get("policyResults", {})
        oracle = policies.get("combined_oracle", {})
        baseline = policies.get("baseline_model_ranked", {})
        model_score = policies.get("fitter_model_score_v0", {})
        assisted = policies.get("fitter_assisted_v0", {})
        assisted_top = (assisted.get("topCandidates") or [{}])[0]
        overlay = row.get("overlayPath") or ""
        overlay_text = f"`{overlay}`" if overlay else ""
        lines.append(
            f"| {row.get('setId')} | {row.get('side')} | {row.get('candidatePoolSize', '')} | "
            f"{row.get('fittedCandidateCount', '')} | {oracle.get('bestDistancePx', '')} | "
            f"{baseline.get('top3DistancePx', '')} | {model_score.get('top3DistancePx', '')} | "
            f"{assisted.get('top3DistancePx', '')} | `{assisted_top.get('source', '')}` | "
            f"{overlay_text} | {row.get('notes', '')} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `baseline_model_ranked` preserves the original #188/#189 ranking.",
        "- `source_heuristic_v0` preserves the best simple #190 static ranker.",
        "- `fitter_model_score_v0` ranks each expanded-pool candidate by its best coherent projected cube model.",
        "- `fitter_assisted_v0` adds weak anchor-mesh and source-prior terms to the model fit score.",
        "- `combined_oracle` is the source-pool ceiling and uses the human label only for evaluation.",
        "- If fitter policies stay far below oracle, the model objective is still missing the signal needed to identify the visible trihedral vertex.",
        "",
    ])
    return "\n".join(lines)


def _headline(summary: Dict[str, Any], threshold_px: float) -> str:
    label = f"@{threshold_px:g}px"
    baseline = summary["policySummaries"]["baseline_model_ranked"]
    source = summary["policySummaries"]["source_heuristic_v0"]
    model = summary["policySummaries"]["fitter_model_score_v0"]
    assisted = summary["policySummaries"]["fitter_assisted_v0"]
    oracle = summary["policySummaries"]["combined_oracle"]
    return (
        f"At {threshold_px:g}px, `fitter_model_score_v0` top-3 recall is "
        f"{model[f'top3HitCount{label}']} / {model['rowCount']} and "
        f"`fitter_assisted_v0` top-3 recall is "
        f"{assisted[f'top3HitCount{label}']} / {assisted['rowCount']}, versus "
        f"{baseline[f'top3HitCount{label}']} / {baseline['rowCount']} for the "
        f"baseline, {source[f'top3HitCount{label}']} / {source['rowCount']} for "
        f"the simple source heuristic, and {oracle[f'oracleHitCount{label}']} / "
        f"{oracle['rowCount']} source-pool oracle. The current fit objective "
        "therefore does not reliably select the true vertex even when the pool "
        "contains it."
    )


def _evaluate_feedback_row(
    feedback_row: Dict[str, Any],
    *,
    thresholds_px: Sequence[float],
    edge_steps: int,
    scoring_max_dim: int,
    output_dir: Optional[Path],
    max_candidates_per_row: Optional[int],
) -> Dict[str, Any]:
    base = {
        "setId": str(feedback_row.get("setId")),
        "side": str(feedback_row.get("side")),
        "evaluationTier": str(feedback_row.get("evaluationTier")),
        "imagePath": str(feedback_row.get("imagePath")),
    }
    human = _parse_point(feedback_row.get("humanVertexPoint"))
    if human is None:
        return {**base, "evaluationStatus": "missing_label", "notes": "missing humanVertexPoint"}

    image_path = Path(str(feedback_row.get("imagePath") or ""))
    if not image_path.exists():
        return {**base, "evaluationStatus": "image_missing", "notes": "image missing"}

    try:
        image, image_rgb = _load_processing_image(image_path)
        mask = _compute_rembg_mask(image)
        detection = detect_interior_bezel_lines(image_rgb, mask)
        candidates = _expanded_candidate_pool(feedback_row, image_rgb, mask, detection)
        fitted = _fit_candidate_pool(
            candidates,
            detection,
            mask,
            edge_steps=edge_steps,
            scoring_max_dim=scoring_max_dim,
            max_candidates=max_candidates_per_row,
        )
        policies = {
            "baseline_model_ranked": _rank_baseline_model(candidates),
            "source_heuristic_v0": _rank_by_score(candidates, _source_heuristic_score),
            "fitter_model_score_v0": _rank_fitted_by_model_score(fitted),
            "fitter_assisted_v0": _rank_by_score(fitted, lambda item: float(item.get("fitterScore", -9999.0))),
            "combined_oracle": _rank_by_score(candidates, lambda item: -_distance_px(human, item["point"])),
        }
        policy_results = {
            policy: _evaluate_ranked_candidates(ranked, human, thresholds_px)
            for policy, ranked in policies.items()
        }
        overlay_path: Optional[Path] = None
        if output_dir is not None and fitted:
            output_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = output_dir / f"set_{base['setId']}_{base['side']}_fitter_assisted_vertex_v0.png"
            render_overlay(image, mask, detection, human, policies["fitter_assisted_v0"], overlay_path)

        return {
            **base,
            "evaluationStatus": "labeled",
            "humanVertexPoint": _round_point(human),
            "candidatePoolSize": len(candidates),
            "fittedCandidateCount": len(fitted),
            "sourceCounts": _source_counts(candidates),
            "fitDiagnostics": {
                "edgeSteps": int(edge_steps),
                "scoringMaxDim": int(scoring_max_dim),
                "scoringScale": round(_mask_scale(mask, scoring_max_dim), 6),
                "detectorSignalQuality": round(float(detection.signal_quality), 4),
                "detectorLineQualities": [round(float(q), 4) for q in detection.line_qualities],
            },
            "policyResults": policy_results,
            "topFitterCandidates": [
                _serialize_fitted_candidate(candidate, rank)
                for rank, candidate in enumerate(policies["fitter_assisted_v0"][:5], start=1)
            ],
            "overlayPath": str(overlay_path) if overlay_path is not None else None,
            "notes": "",
        }
    except Exception as exc:  # pragma: no cover - local CLI/deps path
        return {
            **base,
            "evaluationStatus": "error",
            "notes": f"{exc.__class__.__name__}: {exc}",
        }


def _expanded_candidate_pool(
    feedback_row: Dict[str, Any],
    image_rgb: np.ndarray,
    mask: np.ndarray,
    detection: InteriorBezelDetection,
) -> List[Dict[str, Any]]:
    source_candidates = {
        "model_ranked": _model_ranked_candidates(feedback_row),
        "bezel_line_intersection": _bezel_line_intersection_candidates(detection, mask),
        "bezel_axis_ray": _bezel_axis_ray_candidates(detection, mask),
        "model_local_grid": _model_local_grid_candidates(feedback_row, mask),
        "dark_junction_grid": _dark_junction_grid_candidates(image_rgb, mask),
    }
    return _flatten_sources(source_candidates)


def _fit_candidate_pool(
    candidates: Sequence[Dict[str, Any]],
    detection: InteriorBezelDetection,
    mask: np.ndarray,
    *,
    edge_steps: int,
    scoring_max_dim: int,
    max_candidates: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if detection.cube_center is None or len(detection.boundary_angles) < 3:
        return []
    scale = _mask_scale(mask, scoring_max_dim)
    scaled_mask = _scaled_mask(mask, scale)
    base_units = [_unit_from_angle(angle) for angle in detection.boundary_angles[:3]]
    limited_candidates = list(candidates)
    if max_candidates is not None and max_candidates > 0:
        limited_candidates = limited_candidates[:max_candidates]

    fitted: List[Dict[str, Any]] = []
    for candidate in limited_candidates:
        point = _parse_point(candidate.get("point"))
        if point is None:
            continue
        scaled_point = (point[0] * scale, point[1] * scale)
        model, evaluated = _best_model_for_vertex_point(
            scaled_point,
            base_units,
            scaled_mask,
            detection,
            edge_steps=edge_steps,
        )
        if model is None:
            continue
        anchor_components = _score_anchor_mesh(
            model,
            _anchors_from_model(model),
            scaled_mask,
            anchor_near_radius_px=max(2, int(round(8 * scale))),
        )
        fit_status = _status_from_components(model.score_components)
        fitter_score = _fitter_assisted_score(candidate, model, anchor_components, fit_status)
        fitted.append({
            **candidate,
            "fitStatus": fit_status,
            "modelScore": float(model.score),
            "fitterScore": float(fitter_score),
            "scoreComponents": {
                key: float(value)
                for key, value in sorted(anchor_components.items())
            },
            "edgeLength": float(model.edge_length / scale),
            "signChoice": list(model.sign_choice),
            "evaluatedModels": evaluated,
            "_fitModel": model,
            "_fitScale": scale,
        })
    return fitted


def _best_model_for_vertex_point(
    vertex_point: Point,
    base_units: Sequence[Point],
    mask: np.ndarray,
    detection: InteriorBezelDetection,
    *,
    edge_steps: int,
) -> Tuple[Optional[ProjectedCubeModel], int]:
    edge_min, edge_max = _edge_search_range(vertex_point, mask)
    if edge_max <= edge_min:
        return None, 0
    best: Optional[ProjectedCubeModel] = None
    evaluated = 0
    for signs in _SIGN_CHOICES:
        signed_units = [
            (sign * unit[0], sign * unit[1])
            for sign, unit in zip(signs, base_units)
        ]
        for edge_length in np.linspace(edge_min, edge_max, max(1, int(edge_steps))):
            axes = tuple(
                (float(edge_length * unit[0]), float(edge_length * unit[1]))
                for unit in signed_units
            )
            model = _model_from_axes(
                vertex_point,
                axes,  # type: ignore[arg-type]
                float(edge_length),
                tuple(int(sign) for sign in signs),
                mask,
                detection,
            )
            evaluated += 1
            if best is None or model.score > best.score:
                best = model
    return best, evaluated


_SIGN_CHOICES: Tuple[Tuple[int, int, int], ...] = (
    (-1, -1, -1),
    (-1, -1, 1),
    (-1, 1, -1),
    (-1, 1, 1),
    (1, -1, -1),
    (1, -1, 1),
    (1, 1, -1),
    (1, 1, 1),
)


def _fitter_assisted_score(
    candidate: Dict[str, Any],
    model: ProjectedCubeModel,
    anchor_components: Dict[str, float],
    fit_status: str,
) -> float:
    status_bonus = 0.30 if fit_status == "ok" else 0.0
    source_prior = 0.035 * _source_heuristic_score(candidate)
    source_rank_penalty = 0.004 * float(candidate.get("sourceRank") or 0.0)
    return (
        float(model.score)
        + status_bonus
        + 0.22 * float(anchor_components.get("anchorNearSilhouetteRatio", 0.0))
        + 0.16 * float(anchor_components.get("faceAreaBalance", 0.0))
        + 0.10 * float(anchor_components.get("axisAngleSeparationScore", 0.0))
        + source_prior
        - source_rank_penalty
    )


def _rank_fitted_by_model_score(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda item: (
            item.get("fitStatus") == "ok",
            float(item.get("modelScore", -9999.0)),
            -int(item.get("sourceRank") or 0),
        ),
        reverse=True,
    )


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
    pool_key = "fittedCandidateCount" if policy.startswith("fitter_") else "candidatePoolSize"
    pool_sizes = [int(row.get(pool_key) or 0) for row in rows]
    summary: Dict[str, Any] = {
        "rowCount": len(results),
        "meanCandidatePoolSize": _mean(pool_sizes),
    }
    for threshold in thresholds_px:
        label = f"@{threshold:g}px"
        summary[f"top1HitCount{label}"] = sum(1 for item in results if item.get(f"top1Hit{label}"))
        summary[f"top3HitCount{label}"] = sum(1 for item in results if item.get(f"top3Hit{label}"))
        summary[f"top5HitCount{label}"] = sum(1 for item in results if item.get(f"top5Hit{label}"))
        summary[f"oracleHitCount{label}"] = sum(1 for item in results if item.get(f"oracleHit{label}"))
    return summary


def _serialize_fitted_candidate(candidate: Dict[str, Any], rank: int) -> Dict[str, Any]:
    return {
        "rank": rank,
        "source": candidate.get("source"),
        "sourceRank": candidate.get("sourceRank"),
        "point": _round_point(candidate["point"]),
        "fitStatus": candidate.get("fitStatus"),
        "modelScore": round(float(candidate.get("modelScore", 0.0)), 4),
        "fitterScore": round(float(candidate.get("fitterScore", 0.0)), 4),
        "edgeLength": round(float(candidate.get("edgeLength", 0.0)), 3),
        "signChoice": candidate.get("signChoice"),
        "scoreComponents": {
            key: round(float(value), 4)
            for key, value in sorted((candidate.get("scoreComponents") or {}).items())
        },
        "details": candidate.get("details", {}),
    }


def render_overlay(
    image: Image.Image,
    mask: np.ndarray,
    detection: InteriorBezelDetection,
    human_vertex: Point,
    ranked_candidates: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    mask_alpha = (mask.astype(np.uint8) * 34)
    mask_layer = Image.new("RGBA", base.size, (0, 180, 220, 0))
    mask_layer.putalpha(Image.fromarray(mask_alpha, mode="L"))
    base = Image.alpha_composite(base, mask_layer)

    _draw_detection_lines(draw, detection)
    if ranked_candidates:
        _draw_scaled_model(draw, ranked_candidates[0])
    _draw_ranked_points(draw, ranked_candidates[:5])
    _draw_human_vertex(draw, human_vertex)
    _draw_text_panel(draw, ranked_candidates)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, overlay).convert("RGB").save(output_path, quality=92)


def _draw_detection_lines(draw: ImageDraw.ImageDraw, detection: InteriorBezelDetection) -> None:
    colors = [(255, 0, 255, 210), (0, 220, 255, 210), (255, 215, 0, 210)]
    for idx, line in enumerate(detection.boundary_lines[:3]):
        draw.line([line[0], line[1]], fill=colors[idx % len(colors)], width=3)


def _draw_scaled_model(draw: ImageDraw.ImageDraw, candidate: Dict[str, Any]) -> None:
    model = candidate.get("_fitModel")
    scale = float(candidate.get("_fitScale") or 1.0)
    if model is None:
        return
    face_colors = [(255, 80, 80, 44), (80, 220, 120, 44), (80, 130, 255, 44)]
    line_colors = [(255, 80, 80, 225), (80, 220, 120, 225), (80, 130, 255, 225)]
    for idx, face in enumerate(model.faces):
        quad = [_unscale_point(point, scale) for point in face.quad]
        draw.polygon(quad, fill=face_colors[idx % len(face_colors)])
        draw.line(quad + [quad[0]], fill=line_colors[idx % len(line_colors)], width=4)
        for cell in face.cells:
            cell_quad = [_unscale_point(point, scale) for point in cell["quad"]]
            draw.line(cell_quad + [cell_quad[0]], fill=(255, 255, 255, 105), width=1)


def _draw_ranked_points(draw: ImageDraw.ImageDraw, candidates: Sequence[Dict[str, Any]]) -> None:
    colors = [
        (255, 255, 0, 255),
        (255, 255, 255, 245),
        (255, 150, 0, 245),
        (180, 110, 255, 245),
        (0, 255, 180, 245),
    ]
    for rank, candidate in enumerate(candidates, start=1):
        x, y = candidate["point"]
        radius = 15 if rank == 1 else 10
        color = colors[(rank - 1) % len(colors)]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0, 255), width=3)
        draw.text((x + radius + 4, y - radius), str(rank), fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))


def _draw_human_vertex(draw: ImageDraw.ImageDraw, human_vertex: Point) -> None:
    x, y = human_vertex
    radius = 18
    draw.line((x - radius, y, x + radius, y), fill=(255, 0, 0, 255), width=4)
    draw.line((x, y - radius, x, y + radius), fill=(255, 0, 0, 255), width=4)
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 0, 0, 255))


def _draw_text_panel(draw: ImageDraw.ImageDraw, candidates: Sequence[Dict[str, Any]]) -> None:
    top = candidates[0] if candidates else {}
    components = top.get("scoreComponents") or {}
    lines = [
        f"fitter_assisted_v0: {top.get('fitStatus', 'none')}",
        f"score={float(top.get('fitterScore', 0.0)):.3f} model={float(top.get('modelScore', 0.0)):.3f}",
        f"iou={float(components.get('silhouetteIoU', 0.0)):.3f} anchor={float(components.get('anchorNearSilhouetteRatio', 0.0)):.3f}",
    ]
    x0, y0 = 12, 12
    line_h = 18
    width = 530
    height = 18 + line_h * len(lines)
    draw.rectangle((x0 - 6, y0 - 6, x0 + width, y0 + height), fill=(0, 0, 0, 165))
    for idx, text in enumerate(lines):
        draw.text((x0, y0 + idx * line_h), text, fill=(255, 255, 255, 255))


def _mask_scale(mask: np.ndarray, scoring_max_dim: int) -> float:
    height, width = mask.shape
    max_dim = max(height, width)
    if scoring_max_dim <= 0 or max_dim <= scoring_max_dim:
        return 1.0
    return float(scoring_max_dim) / float(max_dim)


def _scaled_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999999:
        return mask.astype(bool)
    height, width = mask.shape
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return np.asarray(image.resize(size, resample=Image.Resampling.NEAREST), dtype=np.uint8) > 0


def _unscale_point(point: Point, scale: float) -> Point:
    return (float(point[0]) / scale, float(point[1]) / scale)


def _source_counts(candidates: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for candidate in candidates:
        source = str(candidate.get("source"))
        counts[source] = counts.get(source, 0) + 1
    return counts


def _mean(values: Sequence[Any]) -> float:
    normalized = [float(value) for value in values]
    return round(sum(normalized) / len(normalized), 2) if normalized else 0.0


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_strip_runtime_fields(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _strip_runtime_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_runtime_fields(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_strip_runtime_fields(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_runtime_fields(item) for item in value]
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--edge-steps", type=int, default=DEFAULT_EDGE_STEPS)
    parser.add_argument("--scoring-max-dim", type=int, default=DEFAULT_SCORING_MAX_DIM)
    parser.add_argument("--max-candidates-per-row", type=int, default=0)
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
            "error: vertex_fitter_assisted_ranker_v0.py requires optional "
            f"diagnostic dependencies to regenerate outputs: {deps}.\n"
            "Install them in the repo venv, for example:\n"
            "  .venv/bin/pip install rembg scipy onnxruntime",
            file=sys.stderr,
        )
        return 2

    max_candidates = args.max_candidates_per_row if args.max_candidates_per_row > 0 else None
    feedback = _read_json(args.feedback)
    document = generate_fitter_assisted_artifacts(
        feedback,
        thresholds_px=args.thresholds_px,
        edge_steps=args.edge_steps,
        scoring_max_dim=args.scoring_max_dim,
        output_dir=args.out_dir,
        max_candidates_per_row=max_candidates,
    )
    _write_json(args.summary, document)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(_strip_runtime_fields(document)), encoding="utf-8")
    print(f"wrote {args.summary}")
    print(f"wrote {args.report}")
    print(f"wrote overlays to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
