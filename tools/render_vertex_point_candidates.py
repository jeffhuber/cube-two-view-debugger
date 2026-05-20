#!/usr/bin/env python3
"""Render vertex-point candidate diagnostics.

The vertex point is the visible trihedral corner where the three visible cube
faces meet. This tool ranks candidate vertex points and writes top-N overlays
for human review. Diagnostics-only: it does not change recognizer behavior.
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

from tools.evaluate_hybrid_pipeline import _load_processing_image  # noqa: E402
from tools.global_cube_model_v0 import ProjectedCubeModel  # noqa: E402
from tools.interior_bezel_detection import (  # noqa: E402
    InteriorBezelDetection,
    detect_interior_bezel_lines,
)
from tools.render_global_cube_model_v0_overlays import (  # noqa: E402
    DEFAULT_CORPUS_MANIFEST,
    DEFAULT_HARD_MANIFEST,
    EASY_CORPUS_SET_IDS,
    compact_detection,
    diagnostic_disposition,
    evaluation_tier,
    load_pairs,
    missing_required_optional_dependencies,
    parse_side_filter,
    _compute_rembg_mask,
)
from tools.vertex_point_candidates import (  # noqa: E402
    VertexPointCandidateResult,
    fit_result_from_vertex_candidate,
    rank_vertex_point_candidates,
    serialize_vertex_candidate,
)


DEFAULT_OUTPUT_DIR = Path("/tmp/vertex_point_candidate_overlays")
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "vertex_point_candidates_easy_corpus_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "VERTEX_POINT_CANDIDATES_EASY_CORPUS_REPORT.md"


def generate_vertex_point_candidate_artifacts(
    *,
    set_ids: Sequence[str],
    output_dir: Path,
    hard_manifest_path: Path = DEFAULT_HARD_MANIFEST,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    edge_steps: int = 32,
    profile: str = "custom",
    side_filter: Optional[Sequence[Tuple[str, str]]] = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    pairs = load_pairs(hard_manifest_path, corpus_manifest_path)
    rows: List[Dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_side_filter = (
        {(str(set_id), side.upper()) for set_id, side in side_filter}
        if side_filter is not None
        else None
    )

    for set_id in [str(value) for value in set_ids]:
        pair = pairs.get(set_id)
        if pair is None:
            rows.append({
                "setId": set_id,
                "profile": profile,
                "evaluationTier": "unknown",
                "status": "set_not_found",
                "diagnosticDisposition": diagnostic_disposition("unknown", "set_not_found"),
            })
            continue
        for side, image_key in (("A", "imageAPath"), ("B", "imageBPath")):
            if normalized_side_filter is not None and (set_id, side) not in normalized_side_filter:
                continue
            image_path = Path(pair[image_key])
            row = {
                "setId": set_id,
                "side": side,
                "source": pair.get("source"),
                "profile": profile,
                "evaluationTier": evaluation_tier(pair),
                "imagePath": str(image_path),
                "expectedCategory": pair.get("expectedCategory"),
                "expectedScoreFloor": pair.get("expectedScoreFloor"),
                "currentScoreObserved": pair.get("currentScoreObserved"),
            }
            if not image_path.exists():
                rows.append({
                    **row,
                    "status": "image_missing",
                    "diagnosticDisposition": diagnostic_disposition(row["evaluationTier"], "image_missing"),
                })
                continue
            try:
                image, image_rgb = _load_processing_image(image_path)
                mask = _compute_rembg_mask(image)
                detection = detect_interior_bezel_lines(image_rgb, mask)
                result = rank_vertex_point_candidates(
                    detection,
                    mask,
                    edge_steps=edge_steps,
                    top_n=top_n,
                )
                overlay_path = output_dir / f"set_{set_id}_{side}_vertex_point_candidates.png"
                if result.top_candidate is not None:
                    render_overlay(image, mask, detection, result, overlay_path)
                status = result.status
                rows.append({
                    **row,
                    "status": status,
                    "diagnosticDisposition": diagnostic_disposition(row["evaluationTier"], status),
                    "overlayPath": str(overlay_path) if result.top_candidate is not None else None,
                    "detection": compact_detection(detection),
                    "candidateDiagnostics": result.diagnostics,
                    "topCandidates": [
                        serialize_vertex_candidate(candidate, rank)
                        for rank, candidate in enumerate(result.candidates, start=1)
                    ],
                    "manualReview": {
                        "status": "unlabeled",
                        "target": "top1 vertex point within 10 px of human label; top3 contains correct point",
                    },
                })
            except Exception as exc:  # pragma: no cover - local CLI/deps path
                rows.append({
                    **row,
                    "status": "error",
                    "diagnosticDisposition": diagnostic_disposition(row["evaluationTier"], "error"),
                    "error": f"{exc.__class__.__name__}: {exc}",
                })

    return {
        "schemaVersion": 1,
        "probe": "vertex_point_candidates_v0",
        "description": (
            "Diagnostics-only ranking of visible trihedral-corner candidates. "
            "Each candidate is scored by fitting the coherent projected cube "
            "model from that point."
        ),
        "config": {
            "setIds": list(set_ids),
            "edgeSteps": edge_steps,
            "outputDir": str(output_dir),
            "profile": profile,
            "sideFilter": (
                [f"{set_id}:{side}" for set_id, side in sorted(normalized_side_filter)]
                if normalized_side_filter is not None
                else None
            ),
            "topN": top_n,
        },
        "summary": summarize_rows(rows),
        "rows": rows,
    }


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    easy = [row for row in rows if row.get("evaluationTier") == "easy_corpus"]
    top_ok = [row for row in rows if row.get("status") == "ok"]
    return {
        "requestedPairCount": len({row.get("setId") for row in rows}),
        "imageRowCount": len(rows),
        "candidateRowCount": sum(1 for row in rows if row.get("topCandidates")),
        "topOkRowCount": len(top_ok),
        "topWeakRowCount": sum(
            1 for row in rows
            if row.get("topCandidates") and row.get("status") != "ok"
        ),
        "easyImageRowCount": len(easy),
        "easyTopOkRowCount": sum(1 for row in easy if row.get("status") == "ok"),
        "easyTopWeakRowCount": sum(
            1 for row in easy
            if row.get("topCandidates") and row.get("status") != "ok"
        ),
        "errorRowCount": sum(
            1 for row in rows
            if row.get("status") in {"error", "image_missing", "set_not_found"}
        ),
        "unlabeledManualReviewRowCount": sum(
            1 for row in rows
            if (row.get("manualReview") or {}).get("status") == "unlabeled"
        ),
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    lines = [
        "# Vertex Point Candidate Diagnostics",
        "",
        "Diagnostics-only first-principles scaffold. This does not alter recognition behavior.",
        "",
        "The vertex point is the visible trihedral corner where the three visible cube faces meet. This report ranks candidate vertex points before any production wiring.",
        "",
        "## Summary",
        "",
        f"- Requested pairs: {summary['requestedPairCount']}",
        f"- Image rows: {summary['imageRowCount']}",
        f"- Rows with candidates: {summary['candidateRowCount']}",
        f"- Top candidate OK rows: {summary['topOkRowCount']}",
        f"- Top candidate weak rows: {summary['topWeakRowCount']}",
        f"- Easy-corpus top OK rows: {summary['easyTopOkRowCount']} / {summary['easyImageRowCount']}",
        f"- Easy-corpus top weak rows: {summary['easyTopWeakRowCount']}",
        f"- Error/missing rows: {summary['errorRowCount']}",
        f"- Unlabeled manual-review rows: {summary['unlabeledManualReviewRowCount']}",
        "",
        "## Readout",
        "",
        "| Set | Side | Tier | Status | Top source | Top point | Score | IoU | Inside | Cell inside | Same-status gap | Detector signal | Candidates | Overlay |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in document["rows"]:
        candidates = row.get("topCandidates") or []
        top = candidates[0] if candidates else {}
        components = top.get("scoreComponents") or {}
        diagnostics = row.get("candidateDiagnostics") or {}
        overlay = row.get("overlayPath") or ""
        overlay_text = f"`{overlay}`" if overlay else ""
        top_point = top.get("vertexPoint") or ""
        lines.append(
            f"| {row.get('setId')} | {row.get('side', '')} | `{row.get('evaluationTier', '')}` | "
            f"`{row.get('status')}` | `{top.get('source', '')}` | `{top_point}` | "
            f"{top.get('modelScore', '')} | {components.get('silhouetteIoU', '')} | "
            f"{components.get('insideRatio', '')} | {components.get('cellInsideRatio', '')} | "
            f"{diagnostics.get('topSameStatusScoreGap', '')} | "
            f"{diagnostics.get('detectorSignalQuality', '')} | {len(candidates)} | {overlay_text} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Ranking prefers candidates whose coherent projected cube model clears the existing geometry thresholds, then sorts by model score.",
        "- Top-5 candidates are preserved so human labels can measure top-1 precision and top-3 recall instead of forcing a single unreviewed answer.",
        "- A weak top candidate on an easy-corpus row means the vertex point is still a geometry-model iteration target, not a production fallback target.",
        "- The next useful human input is marking the true vertex point on these overlays and recording whether top-1 is within 10 px and top-3 contains the correct point.",
        "",
    ])
    return "\n".join(lines)


def render_overlay(
    image: Image.Image,
    mask: np.ndarray,
    detection: InteriorBezelDetection,
    result: VertexPointCandidateResult,
    output_path: Path,
) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    mask_alpha = (mask.astype(np.uint8) * 38)
    mask_layer = Image.new("RGBA", base.size, (0, 180, 220, 0))
    mask_layer.putalpha(Image.fromarray(mask_alpha, mode="L"))
    base = Image.alpha_composite(base, mask_layer)

    _draw_detection_lines(draw, detection)
    top = result.top_candidate
    if top is not None:
        _draw_model(draw, top.model)
    _draw_candidates(draw, result)
    _draw_text_panel(draw, result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, overlay).convert("RGB").save(output_path, quality=92)


def _draw_detection_lines(draw: ImageDraw.ImageDraw, detection: InteriorBezelDetection) -> None:
    colors = [(255, 0, 255, 220), (0, 220, 255, 220), (255, 215, 0, 220)]
    for idx, line in enumerate(detection.boundary_lines[:3]):
        draw.line([line[0], line[1]], fill=colors[idx % len(colors)], width=3)
    if detection.cube_center is not None:
        x, y = detection.cube_center
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 255, 255, 240), width=2)


def _draw_model(draw: ImageDraw.ImageDraw, model: ProjectedCubeModel) -> None:
    face_colors = [(255, 80, 80, 46), (80, 220, 120, 46), (80, 130, 255, 46)]
    line_colors = [(255, 80, 80, 220), (80, 220, 120, 220), (80, 130, 255, 220)]
    for idx, face in enumerate(model.faces):
        draw.polygon(face.quad, fill=face_colors[idx % len(face_colors)])
        draw.line(face.quad + [face.quad[0]], fill=line_colors[idx % len(line_colors)], width=3)
    cx, cy = model.cube_center
    for ax, ay in model.axes:
        draw.line([(cx, cy), (cx + ax, cy + ay)], fill=(255, 255, 255, 220), width=2)


def _draw_candidates(draw: ImageDraw.ImageDraw, result: VertexPointCandidateResult) -> None:
    colors = [
        (255, 255, 0, 255),
        (255, 255, 255, 245),
        (255, 150, 0, 245),
        (180, 110, 255, 245),
        (0, 255, 180, 245),
    ]
    for rank, candidate in enumerate(result.candidates, start=1):
        x, y = candidate.vertex_point
        radius = 15 if rank == 1 else 10
        color = colors[(rank - 1) % len(colors)]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0, 255), width=3)
        draw.text((x + radius + 4, y - radius), str(rank), fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))


def _draw_text_panel(draw: ImageDraw.ImageDraw, result: VertexPointCandidateResult) -> None:
    top = result.top_candidate
    if top is None:
        return
    components = top.model.score_components
    lines = [
        f"vertex candidates: {result.status}",
        f"top={top.source} score={top.model.score:.3f} iou={components['silhouetteIoU']:.3f}",
        f"inside={components['insideRatio']:.3f} cell={components['cellInsideRatio']:.3f}",
    ]
    x0, y0 = 12, 12
    line_h = 18
    width = 500
    height = 18 + line_h * len(lines)
    draw.rectangle((x0 - 6, y0 - 6, x0 + width, y0 + height), fill=(0, 0, 0, 165))
    for idx, text in enumerate(lines):
        draw.text((x0, y0 + idx * line_h), text, fill=(255, 255, 255, 255))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", nargs="+", default=None)
    parser.add_argument(
        "--profile",
        choices=("easy-corpus", "custom"),
        default="easy-corpus",
        help="Named set selection. --sets overrides the named profile and uses custom.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--hard-manifest", type=Path, default=DEFAULT_HARD_MANIFEST)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--edge-steps", type=int, default=32)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--only-sides",
        nargs="+",
        default=None,
        help="Optional SET:SIDE filters, for example 15:A 26:B.",
    )
    args = parser.parse_args(argv)

    missing = missing_required_optional_dependencies()
    if missing:
        deps = ", ".join(missing)
        print(
            "error: render_vertex_point_candidates.py requires optional "
            f"diagnostic dependencies to regenerate outputs: {deps}.\n"
            "Install them in the repo venv, for example:\n"
            "  .venv/bin/pip install rembg scipy onnxruntime\n"
            "Refusing to write the summary/report because a dependency-light "
            "run would produce an all-error or empty artifact.",
            file=sys.stderr,
        )
        return 2

    profile = args.profile
    if args.sets is not None:
        set_ids = args.sets
        profile = "custom"
    else:
        set_ids = list(EASY_CORPUS_SET_IDS)

    try:
        side_filter = parse_side_filter(args.only_sides)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    document = generate_vertex_point_candidate_artifacts(
        set_ids=set_ids,
        output_dir=args.out_dir,
        hard_manifest_path=args.hard_manifest,
        corpus_manifest_path=args.corpus_manifest,
        edge_steps=args.edge_steps,
        profile=profile,
        side_filter=side_filter,
        top_n=args.top_n,
    )
    if document["summary"]["candidateRowCount"] <= 0:
        print(
            "error: vertex point candidate probe produced zero candidate rows; "
            "refusing to overwrite committed diagnostics artifacts.",
            file=sys.stderr,
        )
        return 2
    _write_json(args.summary, document)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(document), encoding="utf-8")
    print(f"wrote {args.summary}")
    print(f"wrote {args.report}")
    print(f"wrote overlays under {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
