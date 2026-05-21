#!/usr/bin/env python3
"""Foundation segmentation bakeoff V0 for SAM3/Falcon-style masks.

Diagnostics/data-only. This module does not alter recognizer behavior.

The goal is to test whether promptable foundation segmentation can provide a
new signal for the visible trihedral vertex and future face rectification:

* whole cube mask
* top/left/right visible face masks
* sticker masks
* black grid/bezel masks

The repo does not take a hard runtime dependency on SAM3 or Falcon Perception.
Those packages/weights are large and their Python APIs are still moving. This
tool therefore supports a stable external-mask interchange first:

    <mask-dir>/<provider>/set_<SET>_<SIDE>_<prompt>.png

where provider is ``sam3`` or ``falcon`` and prompt is one of the prompt keys
in ``PROMPT_SPECS``. Mask pixels are read from alpha or grayscale >128.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.global_cube_model_v0 import _round_point  # noqa: E402
from tools.vertex_candidate_source_probe import _distance_px, _parse_point  # noqa: E402
from tools.evaluate_hybrid_pipeline import _load_processing_image  # noqa: E402


Point = Tuple[float, float]
MaskMap = Dict[str, np.ndarray]

DEFAULT_FEEDBACK = ROOT / "tests" / "fixtures" / "vertex_point_human_feedback.json"
DEFAULT_SUMMARY = (
    ROOT
    / "tests"
    / "fixtures"
    / "foundation_segmentation_bakeoff_v0_easy_corpus_summary.json"
)
DEFAULT_REPORT = ROOT / "tools" / "FOUNDATION_SEGMENTATION_BAKEOFF_V0_REPORT.md"
DEFAULT_OUTPUT_DIR = Path("/tmp/foundation_segmentation_bakeoff_v0_overlays")
DEFAULT_THRESHOLDS_PX = (10.0, 20.0)
FACE_PROMPTS = ("top_face", "left_face", "right_face")


@dataclass(frozen=True)
class PromptSpec:
    key: str
    prompt: str
    role: str


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    display_name: str
    package_names: Tuple[str, ...]
    notes: str


PROMPT_SPECS: Tuple[PromptSpec, ...] = (
    PromptSpec("whole_cube", "Rubik's cube", "cube_silhouette"),
    PromptSpec("top_face", "top visible face of the Rubik's cube", "visible_face"),
    PromptSpec("left_face", "left visible face of the Rubik's cube", "visible_face"),
    PromptSpec("right_face", "right visible face of the Rubik's cube", "visible_face"),
    PromptSpec("stickers", "colored sticker squares on the Rubik's cube", "stickers"),
    PromptSpec("black_grid_lines", "black plastic grid lines between stickers", "grid_or_bezel"),
)

PROVIDER_SPECS: Tuple[ProviderSpec, ...] = (
    ProviderSpec(
        key="sam3",
        display_name="SAM3 / SAM3.1",
        package_names=("sam3",),
        notes="Promptable segmentation candidate; direct adapter intentionally not wired until local package/API is stable.",
    ),
    ProviderSpec(
        key="falcon",
        display_name="Falcon Perception",
        package_names=("falcon_perception", "falcon_perception_models"),
        notes="Open-vocabulary detection/segmentation candidate; external masks keep the repo dependency-free.",
    ),
)


def generate_segmentation_bakeoff_artifacts(
    feedback_document: Dict[str, Any],
    *,
    external_mask_dir: Optional[Path] = None,
    output_dir: Optional[Path] = DEFAULT_OUTPUT_DIR,
    thresholds_px: Sequence[float] = DEFAULT_THRESHOLDS_PX,
    provider_specs: Sequence[ProviderSpec] = PROVIDER_SPECS,
) -> Dict[str, Any]:
    """Evaluate provider masks, when present, against vertex labels."""
    rows = [
        _evaluate_feedback_row(
            feedback_row,
            external_mask_dir=external_mask_dir,
            output_dir=output_dir,
            thresholds_px=thresholds_px,
            provider_specs=provider_specs,
        )
        for feedback_row in feedback_document.get("rows", [])
        if feedback_row.get("status") == "labeled"
    ]
    provider_status = {
        provider.key: _provider_status(provider, external_mask_dir)
        for provider in provider_specs
    }
    return {
        "schemaVersion": 1,
        "probe": "foundation_segmentation_bakeoff_v0",
        "description": (
            "Diagnostics-only bakeoff harness for SAM3/Falcon-style promptable "
            "segmentation masks as vertex and face-rectification signals."
        ),
        "sourceFeedback": str(DEFAULT_FEEDBACK),
        "thresholdsPx": [float(value) for value in thresholds_px],
        "config": {
            "externalMaskDir": str(external_mask_dir) if external_mask_dir else None,
            "outputDir": str(output_dir) if output_dir else None,
            "providers": [provider.key for provider in provider_specs],
            "prompts": [
                {"key": prompt.key, "prompt": prompt.prompt, "role": prompt.role}
                for prompt in PROMPT_SPECS
            ],
        },
        "providerStatus": provider_status,
        "summary": summarize_rows(rows, provider_specs, thresholds_px),
        "rows": rows,
    }


def summarize_rows(
    rows: Sequence[Dict[str, Any]],
    provider_specs: Sequence[ProviderSpec] = PROVIDER_SPECS,
    thresholds_px: Sequence[float] = DEFAULT_THRESHOLDS_PX,
) -> Dict[str, Any]:
    provider_summaries = {
        provider.key: _summarize_provider(rows, provider.key, thresholds_px)
        for provider in provider_specs
    }
    return {
        "rowCount": len(rows),
        "providerSummaries": provider_summaries,
    }


def render_report(document: Dict[str, Any]) -> str:
    thresholds = [float(value) for value in document["thresholdsPx"]]
    primary = thresholds[0]
    secondary = thresholds[1] if len(thresholds) > 1 else thresholds[0]
    summary = document["summary"]

    lines = [
        "# Foundation Segmentation Bakeoff V0",
        "",
        "Diagnostics/data-only artifact. This does not alter recognition behavior.",
        "",
        "This harness evaluates whether SAM3/Falcon-style masks can provide face-plane or sticker/grid evidence for the visible trihedral vertex and future rectification.",
        "",
        "## Current Status",
        "",
        _headline(document, primary),
        "",
        "## Provider Availability",
        "",
        "| Provider | Package installed | External mask files | Status | Notes |",
        "|---|---:|---:|---|---|",
    ]
    for provider in document["config"]["providers"]:
        status = document["providerStatus"][provider]
        lines.append(
            f"| `{provider}` | {status['packageInstalled']} | "
            f"{status['externalMaskFileCount']} | `{status['status']}` | "
            f"{status['notes']} |"
        )

    lines.extend([
        "",
        "## Prompt Matrix",
        "",
        "| Key | Prompt | Role |",
        "|---|---|---|",
    ])
    for prompt in document["config"]["prompts"]:
        lines.append(f"| `{prompt['key']}` | {prompt['prompt']} | `{prompt['role']}` |")

    lines.extend([
        "",
        "## Metrics",
        "",
        "| Provider | Rows | Rows with masks | Rows with 3 face masks | Top-1 @10 | Top-3 @10 | Top-5 @10 | Oracle @10 | Top-3 @20 | Oracle @20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for provider in document["config"]["providers"]:
        item = summary["providerSummaries"][provider]
        lines.append(
            f"| `{provider}` | {item['rowCount']} | {item['rowsWithAnyMask']} | "
            f"{item['rowsWithThreeFaceMasks']} | "
            f"{item[f'top1HitCount@{primary:g}px']} | "
            f"{item[f'top3HitCount@{primary:g}px']} | "
            f"{item[f'top5HitCount@{primary:g}px']} | "
            f"{item[f'oracleHitCount@{primary:g}px']} | "
            f"{item[f'top3HitCount@{secondary:g}px']} | "
            f"{item[f'oracleHitCount@{secondary:g}px']} |"
        )

    lines.extend([
        "",
        "## Per-Row Readout",
        "",
        "| Set | Side | Provider | Mask prompts | Candidates | Best distance | Top-3 distance | Overlay | Notes |",
        "|---:|---|---|---|---:|---:|---:|---|---|",
    ])
    for row in document["rows"]:
        for provider in document["config"]["providers"]:
            result = row.get("providers", {}).get(provider, {})
            metrics = result.get("metrics") or {}
            overlay = result.get("overlayPath") or ""
            overlay_text = f"`{overlay}`" if overlay else ""
            lines.append(
                f"| {row.get('setId')} | {row.get('side')} | `{provider}` | "
                f"{', '.join(f'`{key}`' for key in result.get('maskPrompts', []))} | "
                f"{result.get('candidateCount', 0)} | {metrics.get('bestDistancePx', '')} | "
                f"{metrics.get('top3DistancePx', '')} | {overlay_text} | "
                f"{result.get('notes', '')} |"
            )

    lines.extend([
        "",
        "## External Mask Schema",
        "",
        "Place masks at:",
        "",
        "```text",
        "<mask-dir>/<provider>/set_<SET>_<SIDE>_<prompt>.png",
        "```",
        "",
        "Example:",
        "",
        "```text",
        "/tmp/foundation_masks/sam3/set_15_A_top_face.png",
        "/tmp/foundation_masks/sam3/set_15_A_left_face.png",
        "/tmp/foundation_masks/sam3/set_15_A_right_face.png",
        "```",
        "",
        "Mask pixels are read from alpha or grayscale values greater than 128.",
        "",
        "## Interpretation",
        "",
        "- The most important first metric is whether three visible-face masks generate vertex candidates near the human-labeled visible trihedral vertex.",
        "- Whole-cube masks are useful as a silhouette replacement/cross-check, but face masks are the key signal for rectification and vertex selection.",
        "- A future direct SAM3/Falcon adapter should write exactly this external-mask schema first, then graduate to in-process inference only if dependency/runtime cost is acceptable.",
        "",
    ])
    return "\n".join(lines)


def _headline(document: Dict[str, Any], threshold_px: float) -> str:
    provider_bits = []
    for provider, status in document["providerStatus"].items():
        metrics = document["summary"]["providerSummaries"][provider]
        label = f"@{threshold_px:g}px"
        provider_bits.append(
            f"`{provider}` status `{status['status']}`, top-3 {label} "
            f"{metrics[f'top3HitCount{label}']} / {metrics['rowCount']}"
        )
    if not provider_bits:
        return "No providers configured."
    return (
        "; ".join(provider_bits)
        + ". Direct model dependencies are intentionally optional; current committed output is a harness/scaffold unless external masks are supplied."
    )


def _evaluate_feedback_row(
    feedback_row: Dict[str, Any],
    *,
    external_mask_dir: Optional[Path],
    output_dir: Optional[Path],
    thresholds_px: Sequence[float],
    provider_specs: Sequence[ProviderSpec],
) -> Dict[str, Any]:
    human = _parse_point(feedback_row.get("humanVertexPoint"))
    base = {
        "setId": str(feedback_row.get("setId")),
        "side": str(feedback_row.get("side")),
        "evaluationTier": str(feedback_row.get("evaluationTier")),
        "imagePath": str(feedback_row.get("imagePath")),
        "humanVertexPoint": _round_point(human) if human is not None else None,
    }
    provider_results = {
        provider.key: _evaluate_provider_row(
            provider,
            feedback_row,
            human,
            external_mask_dir=external_mask_dir,
            output_dir=output_dir,
            thresholds_px=thresholds_px,
        )
        for provider in provider_specs
    }
    return {**base, "providers": provider_results}


def _evaluate_provider_row(
    provider: ProviderSpec,
    feedback_row: Dict[str, Any],
    human_vertex: Optional[Point],
    *,
    external_mask_dir: Optional[Path],
    output_dir: Optional[Path],
    thresholds_px: Sequence[float],
) -> Dict[str, Any]:
    if human_vertex is None:
        return {"status": "missing_label", "notes": "missing humanVertexPoint"}
    masks = _load_external_masks(
        external_mask_dir,
        provider.key,
        str(feedback_row.get("setId")),
        str(feedback_row.get("side")),
    )
    if not masks:
        return {
            "status": "no_external_masks",
            "maskPrompts": [],
            "candidateCount": 0,
            "metrics": _empty_metrics(thresholds_px),
            "notes": "direct model inference is not wired; provide external masks to score this provider",
        }

    candidates = _vertex_candidates_from_masks(masks)
    metrics = _evaluate_candidates(candidates, human_vertex, thresholds_px)
    overlay_path: Optional[Path] = None
    image_path = Path(str(feedback_row.get("imagePath") or ""))
    if output_dir is not None and image_path.exists():
        overlay_path = output_dir / provider.key / (
            f"set_{feedback_row.get('setId')}_{feedback_row.get('side')}_foundation_segmentation_v0.png"
        )
        _render_overlay(image_path, masks, candidates, human_vertex, overlay_path)

    return {
        "status": "evaluated",
        "maskPrompts": sorted(masks),
        "maskStats": {
            key: _mask_stats(mask)
            for key, mask in sorted(masks.items())
        },
        "candidateCount": len(candidates),
        "topCandidates": candidates[:5],
        "metrics": metrics,
        "overlayPath": str(overlay_path) if overlay_path else None,
        "notes": "",
    }


def _load_external_masks(
    external_mask_dir: Optional[Path],
    provider_key: str,
    set_id: str,
    side: str,
) -> MaskMap:
    if external_mask_dir is None:
        return {}
    masks: MaskMap = {}
    provider_dir = external_mask_dir / provider_key
    if not provider_dir.exists():
        return masks
    for prompt in PROMPT_SPECS:
        path = provider_dir / f"set_{set_id}_{side}_{prompt.key}.png"
        if path.exists():
            masks[prompt.key] = _read_mask_png(path)
    return masks


def _read_mask_png(path: Path) -> np.ndarray:
    image = Image.open(path)
    if image.mode in {"RGBA", "LA"}:
        alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
        return alpha > 128
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    return gray > 128


def _vertex_candidates_from_masks(masks: MaskMap) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    face_masks = {key: masks[key] for key in FACE_PROMPTS if key in masks}
    if len(face_masks) >= 2:
        candidates.extend(_face_boundary_vertex_candidates(face_masks))
    if "black_grid_lines" in masks and face_masks:
        candidates.extend(_grid_face_boundary_candidates(masks["black_grid_lines"], face_masks))
    return _dedupe_candidates(candidates, min_distance_px=4.0, limit=20)


def _face_boundary_vertex_candidates(
    face_masks: MaskMap,
    *,
    threshold_px: float = 7.0,
    max_points_per_face: int = 500,
) -> List[Dict[str, Any]]:
    boundaries = {
        key: _subsample_points(_boundary_points(mask), max_points_per_face)
        for key, mask in face_masks.items()
    }
    boundaries = {key: points for key, points in boundaries.items() if len(points)}
    if len(boundaries) < 2:
        return []

    all_points = np.vstack(list(boundaries.values()))
    sampled = _subsample_points(all_points, max_points_per_face * 2)
    candidates: List[Dict[str, Any]] = []
    for y, x in sampled:
        point = np.array([float(y), float(x)])
        distances = {
            key: _min_distance_to_points(point, points)
            for key, points in boundaries.items()
        }
        hit_count = sum(1 for distance in distances.values() if distance <= threshold_px)
        total_distance = sum(distances.values())
        score = hit_count * 1000.0 - total_distance
        candidates.append({
            "source": "face_boundary_intersection",
            "point": (float(x), float(y)),
            "score": round(float(score), 4),
            "details": {
                "faceBoundaryHitCount": hit_count,
                "faceBoundaryDistanceSumPx": round(float(total_distance), 2),
                "faceBoundaryDistancesPx": {
                    key: round(float(value), 2)
                    for key, value in sorted(distances.items())
                },
            },
        })
    return sorted(candidates, key=lambda item: float(item["score"]), reverse=True)


def _grid_face_boundary_candidates(grid_mask: np.ndarray, face_masks: MaskMap) -> List[Dict[str, Any]]:
    grid_points = _subsample_points(_boundary_points(grid_mask), 700)
    if len(grid_points) == 0:
        return []
    face_boundary_parts = []
    for mask in face_masks.values():
        points = _boundary_points(mask)
        if len(points):
            face_boundary_parts.append(_subsample_points(points, 400))
    if not face_boundary_parts:
        return []
    face_boundary = np.vstack(face_boundary_parts)
    candidates: List[Dict[str, Any]] = []
    for y, x in _subsample_points(grid_points, 300):
        point = np.array([float(y), float(x)])
        face_distance = _min_distance_to_points(point, face_boundary)
        score = 120.0 - face_distance
        if face_distance <= 18.0:
            candidates.append({
                "source": "grid_near_face_boundary",
                "point": (float(x), float(y)),
                "score": round(float(score), 4),
                "details": {"faceBoundaryDistancePx": round(float(face_distance), 2)},
            })
    return sorted(candidates, key=lambda item: float(item["score"]), reverse=True)


def _boundary_points(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    boundary = np.zeros(mask_bool.shape, dtype=bool)
    boundary[:-1, :] |= mask_bool[:-1, :] != mask_bool[1:, :]
    boundary[1:, :] |= mask_bool[:-1, :] != mask_bool[1:, :]
    boundary[:, :-1] |= mask_bool[:, :-1] != mask_bool[:, 1:]
    boundary[:, 1:] |= mask_bool[:, :-1] != mask_bool[:, 1:]
    boundary &= mask_bool
    return np.argwhere(boundary)


def _subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    stride = max(1, math.ceil(len(points) / max_points))
    return points[::stride][:max_points]


def _min_distance_to_points(point_yx: np.ndarray, points_yx: np.ndarray) -> float:
    if len(points_yx) == 0:
        return float("inf")
    deltas = points_yx.astype(float) - point_yx[None, :]
    return float(np.sqrt((deltas * deltas).sum(axis=1)).min())


def _dedupe_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    min_distance_px: float,
    limit: int,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        point = _parse_point(candidate.get("point"))
        if point is None:
            continue
        if any(_distance_px(point, existing["point"]) < min_distance_px for existing in result):
            continue
        result.append({**candidate, "point": point})
        if len(result) >= limit:
            break
    return result


def _evaluate_candidates(
    candidates: Sequence[Dict[str, Any]],
    human_vertex: Point,
    thresholds_px: Sequence[float],
) -> Dict[str, Any]:
    ranked = []
    for rank, candidate in enumerate(candidates, start=1):
        point = _parse_point(candidate.get("point"))
        if point is None:
            continue
        ranked.append({
            "rank": rank,
            "source": candidate.get("source"),
            "point": _round_point(point),
            "distancePx": round(_distance_px(human_vertex, point), 2),
            "details": candidate.get("details", {}),
        })
    best = min(ranked, key=lambda item: float(item["distancePx"]), default=None)
    metrics: Dict[str, Any] = {
        "candidateCount": len(ranked),
        "bestDistancePx": best["distancePx"] if best else None,
        "top1DistancePx": ranked[0]["distancePx"] if ranked else None,
        "top3DistancePx": _min_ranked_distance(ranked[:3]),
        "top5DistancePx": _min_ranked_distance(ranked[:5]),
    }
    for threshold in thresholds_px:
        label = f"@{threshold:g}px"
        metrics[f"top1Hit{label}"] = any(item["distancePx"] <= threshold for item in ranked[:1])
        metrics[f"top3Hit{label}"] = any(item["distancePx"] <= threshold for item in ranked[:3])
        metrics[f"top5Hit{label}"] = any(item["distancePx"] <= threshold for item in ranked[:5])
        metrics[f"oracleHit{label}"] = best is not None and best["distancePx"] <= threshold
    return metrics


def _empty_metrics(thresholds_px: Sequence[float]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "candidateCount": 0,
        "bestDistancePx": None,
        "top1DistancePx": None,
        "top3DistancePx": None,
        "top5DistancePx": None,
    }
    for threshold in thresholds_px:
        label = f"@{threshold:g}px"
        metrics[f"top1Hit{label}"] = False
        metrics[f"top3Hit{label}"] = False
        metrics[f"top5Hit{label}"] = False
        metrics[f"oracleHit{label}"] = False
    return metrics


def _summarize_provider(
    rows: Sequence[Dict[str, Any]],
    provider_key: str,
    thresholds_px: Sequence[float],
) -> Dict[str, Any]:
    provider_rows = [
        row.get("providers", {}).get(provider_key, {})
        for row in rows
    ]
    summary: Dict[str, Any] = {
        "rowCount": len(provider_rows),
        "rowsWithAnyMask": sum(1 for row in provider_rows if row.get("maskPrompts")),
        "rowsWithThreeFaceMasks": sum(
            1 for row in provider_rows
            if all(prompt in set(row.get("maskPrompts", [])) for prompt in FACE_PROMPTS)
        ),
        "rowsWithCandidates": sum(1 for row in provider_rows if int(row.get("candidateCount") or 0) > 0),
    }
    for threshold in thresholds_px:
        label = f"@{threshold:g}px"
        summary[f"top1HitCount{label}"] = sum(1 for row in provider_rows if (row.get("metrics") or {}).get(f"top1Hit{label}"))
        summary[f"top3HitCount{label}"] = sum(1 for row in provider_rows if (row.get("metrics") or {}).get(f"top3Hit{label}"))
        summary[f"top5HitCount{label}"] = sum(1 for row in provider_rows if (row.get("metrics") or {}).get(f"top5Hit{label}"))
        summary[f"oracleHitCount{label}"] = sum(1 for row in provider_rows if (row.get("metrics") or {}).get(f"oracleHit{label}"))
    return summary


def _provider_status(provider: ProviderSpec, external_mask_dir: Optional[Path]) -> Dict[str, Any]:
    installed_packages = [
        name for name in provider.package_names
        if importlib.util.find_spec(name) is not None
    ]
    external_count = 0
    if external_mask_dir is not None:
        provider_dir = external_mask_dir / provider.key
        if provider_dir.exists():
            external_count = sum(1 for _ in provider_dir.glob("set_*_*.png"))
    if external_count:
        status = "external_masks_available"
    elif installed_packages:
        status = "package_detected_external_masks_required"
    else:
        status = "package_missing_external_masks_required"
    return {
        "displayName": provider.display_name,
        "packageInstalled": bool(installed_packages),
        "installedPackages": installed_packages,
        "externalMaskFileCount": external_count,
        "status": status,
        "notes": provider.notes,
    }


def _mask_stats(mask: np.ndarray) -> Dict[str, Any]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {"pixelCount": 0, "areaFraction": 0.0, "bbox": None, "centroid": None}
    return {
        "pixelCount": int(mask.sum()),
        "areaFraction": round(float(mask.sum()) / float(mask.size), 6),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid": _round_point((float(xs.mean()), float(ys.mean()))),
    }


def _min_ranked_distance(ranked: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not ranked:
        return None
    return min(float(item["distancePx"]) for item in ranked)


def _render_overlay(
    image_path: Path,
    masks: MaskMap,
    candidates: Sequence[Dict[str, Any]],
    human_vertex: Point,
    output_path: Path,
) -> None:
    processing_image, _ = _load_processing_image(image_path)
    base = processing_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    colors = {
        "whole_cube": (0, 180, 220, 38),
        "top_face": (255, 215, 0, 70),
        "left_face": (255, 80, 80, 70),
        "right_face": (80, 130, 255, 70),
        "stickers": (80, 220, 120, 55),
        "black_grid_lines": (255, 0, 255, 90),
    }
    for key, mask in masks.items():
        layer = Image.new("RGBA", base.size, colors.get(key, (255, 255, 255, 40)))
        layer.putalpha(Image.fromarray((mask.astype(np.uint8) * colors.get(key, (0, 0, 0, 40))[3]), mode="L"))
        overlay = Image.alpha_composite(overlay, layer)
    draw = ImageDraw.Draw(overlay)
    _draw_candidates(draw, candidates[:5])
    hx, hy = human_vertex
    draw.line((hx - 18, hy, hx + 18, hy), fill=(255, 0, 0, 255), width=4)
    draw.line((hx, hy - 18, hx, hy + 18), fill=(255, 0, 0, 255), width=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, overlay).convert("RGB").save(output_path, quality=92)


def _draw_candidates(draw: ImageDraw.ImageDraw, candidates: Sequence[Dict[str, Any]]) -> None:
    colors = [
        (255, 255, 0, 255),
        (255, 255, 255, 245),
        (255, 150, 0, 245),
        (180, 110, 255, 245),
        (0, 255, 180, 245),
    ]
    for rank, candidate in enumerate(candidates, start=1):
        point = _parse_point(candidate.get("point"))
        if point is None:
            continue
        x, y = point
        radius = 14 if rank == 1 else 9
        color = colors[(rank - 1) % len(colors)]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0, 255), width=3)
        draw.text((x + radius + 4, y - radius), str(rank), fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))


def _mean(values: Iterable[Any]) -> float:
    normalized = [float(value) for value in values]
    return round(sum(normalized) / len(normalized), 2) if normalized else 0.0


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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--external-mask-dir", type=Path, default=None)
    parser.add_argument(
        "--thresholds-px",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS_PX),
    )
    args = parser.parse_args(argv)

    feedback = _read_json(args.feedback)
    document = generate_segmentation_bakeoff_artifacts(
        feedback,
        external_mask_dir=args.external_mask_dir,
        output_dir=args.out_dir,
        thresholds_px=args.thresholds_px,
    )
    _write_json(args.summary, document)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(document), encoding="utf-8")
    print(f"wrote {args.summary}")
    print(f"wrote {args.report}")
    if args.external_mask_dir:
        print(f"read external masks from {args.external_mask_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
