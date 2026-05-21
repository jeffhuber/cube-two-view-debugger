#!/usr/bin/env python3
"""Render weak-projection 7-anchor cube mesh diagnostics.

Diagnostics-only: no recognizer behavior changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cube_mesh_anchor_fitter_v0 import (  # noqa: E402
    ANCHOR_NAMES,
    CubeMeshAnchorFit,
    CubeMeshAnchorResult,
    fit_cube_mesh_anchor_candidates,
    serialize_anchor_mesh,
)
from tools.evaluate_hybrid_pipeline import _load_processing_image  # noqa: E402
from tools.interior_bezel_detection import (  # noqa: E402
    InteriorBezelDetection,
    detect_interior_bezel_lines,
)
from tools.render_global_cube_model_v0_overlays import (  # noqa: E402
    DEFAULT_CORPUS_MANIFEST,
    DEFAULT_HARD_MANIFEST,
    EASY_CORPUS_SET_IDS,
    _compute_rembg_mask,
    compact_detection,
    diagnostic_disposition,
    evaluation_tier,
    load_pairs,
    missing_required_optional_dependencies,
    parse_side_filter,
)


DEFAULT_OUTPUT_DIR = Path("/tmp/cube_mesh_anchor_v0_overlays")
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "cube_mesh_anchor_v0_easy_corpus_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "CUBE_MESH_ANCHOR_V0_EASY_CORPUS_REPORT.md"


def generate_cube_mesh_anchor_artifacts(
    *,
    set_ids: Sequence[str],
    output_dir: Path,
    hard_manifest_path: Path = DEFAULT_HARD_MANIFEST,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    edge_steps: int = 32,
    profile: str = "custom",
    side_filter: Optional[Sequence[Tuple[str, str]]] = None,
    top_vertex_candidates: int = 5,
    top_meshes: int = 5,
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
                result = fit_cube_mesh_anchor_candidates(
                    detection,
                    mask,
                    edge_steps=edge_steps,
                    top_vertex_candidates=top_vertex_candidates,
                    top_meshes=top_meshes,
                )
                overlay_path = output_dir / f"set_{set_id}_{side}_cube_mesh_anchor_v0.png"
                if result.top_mesh is not None:
                    render_overlay(image, mask, detection, result, overlay_path)
                status = result.status
                rows.append({
                    **row,
                    "status": status,
                    "diagnosticDisposition": diagnostic_disposition(row["evaluationTier"], status),
                    "overlayPath": str(overlay_path) if result.top_mesh is not None else None,
                    "detection": compact_detection(detection),
                    "fitDiagnostics": result.diagnostics,
                    "anchorMeshes": [
                        serialize_anchor_mesh(mesh, rank)
                        for rank, mesh in enumerate(result.meshes, start=1)
                    ],
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
        "probe": "cube_mesh_anchor_v0",
        "description": (
            "Diagnostics-only weak-projection 7-anchor mesh seeded from ranked "
            "visible trihedral vertex-point candidates."
        ),
        "config": {
            "setIds": list(set_ids),
            "edgeSteps": edge_steps,
            "outputDir": str(output_dir),
            "profile": profile,
            "topVertexCandidates": top_vertex_candidates,
            "topMeshes": top_meshes,
            "sideFilter": (
                [f"{set_id}:{side}" for set_id, side in sorted(normalized_side_filter)]
                if normalized_side_filter is not None
                else None
            ),
        },
        "summary": summarize_rows(rows),
        "rows": rows,
    }


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    easy = [row for row in rows if row.get("evaluationTier") == "easy_corpus"]
    fitted = [row for row in rows if row.get("anchorMeshes")]
    return {
        "requestedPairCount": len({row.get("setId") for row in rows}),
        "imageRowCount": len(rows),
        "fittedRowCount": len(fitted),
        "okRowCount": sum(1 for row in rows if row.get("status") == "ok"),
        "weakRowCount": sum(
            1 for row in rows
            if row.get("anchorMeshes") and row.get("status") != "ok"
        ),
        "lowIouRowCount": sum(1 for row in rows if row.get("status") == "low_iou"),
        "lowAnchorSupportWarningRowCount": sum(
            1 for row in rows
            if any(
                "low_anchor_silhouette_support" in (mesh.get("warnings") or [])
                for mesh in row.get("anchorMeshes", [])
            )
        ),
        "errorRowCount": sum(
            1 for row in rows
            if row.get("status") in {"error", "image_missing", "set_not_found"}
        ),
        "easyImageRowCount": len(easy),
        "easyOkRowCount": sum(1 for row in easy if row.get("status") == "ok"),
        "easyWeakRowCount": sum(
            1 for row in easy
            if row.get("anchorMeshes") and row.get("status") != "ok"
        ),
        "modelIterationNeededRowCount": sum(
            1 for row in rows
            if row.get("diagnosticDisposition") == "model_iteration_needed"
        ),
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    lines = [
        "# Cube Mesh Anchor V0 Diagnostics",
        "",
        "Diagnostics-only first-principles scaffold. This does not alter recognition behavior.",
        "",
        "This probe fits a weak-projection 7-anchor cube mesh from the ranked visible trihedral vertex candidates.",
        "",
        "## Summary",
        "",
        f"- Requested pairs: {summary['requestedPairCount']}",
        f"- Image rows: {summary['imageRowCount']}",
        f"- Fitted rows: {summary['fittedRowCount']}",
        f"- OK rows: {summary['okRowCount']}",
        f"- Weak rows: {summary['weakRowCount']}",
        f"- Low-IoU rows: {summary['lowIouRowCount']}",
        f"- Low-anchor-support warning rows: {summary['lowAnchorSupportWarningRowCount']}",
        f"- Error/missing rows: {summary['errorRowCount']}",
        f"- Easy-corpus OK rows: {summary['easyOkRowCount']} / {summary['easyImageRowCount']}",
        f"- Easy-corpus weak rows: {summary['easyWeakRowCount']}",
        f"- Model-iteration-needed rows: {summary['modelIterationNeededRowCount']}",
        "",
        "## Readout",
        "",
        "| Set | Side | Tier | Status | Source vertex | Score | IoU | Anchor support | Face balance | Axis sep | Overlay |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in document["rows"]:
        meshes = row.get("anchorMeshes") or []
        top = meshes[0] if meshes else {}
        components = top.get("scoreComponents") or {}
        source = top.get("sourceVertexCandidate") or {}
        overlay = row.get("overlayPath") or ""
        overlay_text = f"`{overlay}`" if overlay else ""
        lines.append(
            f"| {row.get('setId')} | {row.get('side', '')} | `{row.get('evaluationTier', '')}` | "
            f"`{row.get('status')}` | `{source.get('source', '')}` r{top.get('sourceVertexRank', '')} | "
            f"{top.get('score', '')} | {components.get('silhouetteIoU', '')} | "
            f"{components.get('anchorNearSilhouetteRatio', '')} | "
            f"{components.get('faceAreaBalance', '')} | "
            f"{components.get('axisAngleSeparationScore', '')} | {overlay_text} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `V` is the visible trihedral vertex point; `X/Y/Z` are one shared edge length away; `XY/YZ/ZX` close the three visible face parallelograms.",
        "- This is a weak projected-image mesh, not a calibrated 3D pose solve. The report records `pnpStatus=not_run_no_calibrated_camera_or_human_anchor_labels` in row diagnostics.",
        "- Anchor-near-silhouette support is reported as a warning metric only; it is not a V0 status gate because hull-edge anchors sit on noisy rembg boundaries.",
        "- Easy-corpus weak rows should drive model iteration. Hard-background weak rows remain retake/segmentation candidates rather than forced recognizer wins.",
        "- The companion vertex human-feedback artifact should decide whether these meshes are anchored on the correct visible vertex before any downstream color sampling is considered.",
        "",
    ])
    return "\n".join(lines)


def render_overlay(
    image: Image.Image,
    mask: np.ndarray,
    detection: InteriorBezelDetection,
    result: CubeMeshAnchorResult,
    output_path: Path,
) -> None:
    top = result.top_mesh
    if top is None:
        raise ValueError("cannot render overlay without an anchor mesh")

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    mask_alpha = (mask.astype(np.uint8) * 34)
    mask_layer = Image.new("RGBA", base.size, (0, 180, 220, 0))
    mask_layer.putalpha(Image.fromarray(mask_alpha, mode="L"))
    base = Image.alpha_composite(base, mask_layer)

    _draw_detection_lines(draw, detection)
    _draw_mesh(draw, top)
    _draw_text_panel(draw, result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, overlay).convert("RGB").save(output_path, quality=92)


def _draw_detection_lines(draw: ImageDraw.ImageDraw, detection: InteriorBezelDetection) -> None:
    colors = [(255, 0, 255, 210), (0, 220, 255, 210), (255, 215, 0, 210)]
    for idx, line in enumerate(detection.boundary_lines[:3]):
        draw.line([line[0], line[1]], fill=colors[idx % len(colors)], width=3)


def _draw_mesh(draw: ImageDraw.ImageDraw, fit: CubeMeshAnchorFit) -> None:
    model = fit.model
    face_colors = [(255, 80, 80, 42), (80, 220, 120, 42), (80, 130, 255, 42)]
    line_colors = [(255, 80, 80, 230), (80, 220, 120, 230), (80, 130, 255, 230)]
    for idx, face in enumerate(model.faces):
        draw.polygon(face.quad, fill=face_colors[idx % len(face_colors)])
        draw.line(face.quad + [face.quad[0]], fill=line_colors[idx % len(line_colors)], width=4)
        for cell in face.cells:
            quad = cell["quad"]
            draw.line(quad + [quad[0]], fill=(255, 255, 255, 110), width=1)

    anchors = fit.anchors
    edges = (
        ("V", "X"), ("V", "Y"), ("V", "Z"),
        ("X", "XY"), ("Y", "XY"),
        ("Y", "YZ"), ("Z", "YZ"),
        ("Z", "ZX"), ("X", "ZX"),
    )
    for start, end in edges:
        draw.line([anchors[start], anchors[end]], fill=(255, 255, 255, 240), width=3)
    for name in ANCHOR_NAMES:
        x, y = anchors[name]
        radius = 10 if name == "V" else 7
        fill = (255, 255, 0, 255) if name == "V" else (255, 255, 255, 245)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(0, 0, 0, 255), width=3)
        draw.text((x + radius + 3, y - radius), name, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))


def _draw_text_panel(draw: ImageDraw.ImageDraw, result: CubeMeshAnchorResult) -> None:
    top = result.top_mesh
    if top is None:
        return
    components = top.score_components
    lines = [
        f"cube_mesh_anchor_v0: {top.status}",
        f"score={top.score:.3f} iou={components['silhouetteIoU']:.3f}",
        f"anchor={components['anchorNearSilhouetteRatio']:.3f} face={components['faceAreaBalance']:.3f}",
    ]
    x0, y0 = 12, 12
    line_h = 18
    width = 470
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
    parser.add_argument("--top-vertex-candidates", type=int, default=5)
    parser.add_argument("--top-meshes", type=int, default=5)
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
            "error: render_cube_mesh_anchor_overlays.py requires optional "
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

    document = generate_cube_mesh_anchor_artifacts(
        set_ids=set_ids,
        output_dir=args.out_dir,
        hard_manifest_path=args.hard_manifest,
        corpus_manifest_path=args.corpus_manifest,
        edge_steps=args.edge_steps,
        profile=profile,
        side_filter=side_filter,
        top_vertex_candidates=args.top_vertex_candidates,
        top_meshes=args.top_meshes,
    )
    if document["summary"]["fittedRowCount"] <= 0:
        print(
            "error: cube mesh anchor probe produced zero fitted rows; refusing "
            "to overwrite committed diagnostics artifacts.",
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
