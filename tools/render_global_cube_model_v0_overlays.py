#!/usr/bin/env python3
"""Render diagnostics for the projected cube model V0.

This is the first small implementation of the first-principles direction:
fit one coherent projected cube model, generate all visible face/cell quads
from that model, and inspect how well it agrees with the cube silhouette.

Diagnostics-only. No recognizer behavior changes.
"""

from __future__ import annotations

import argparse
import importlib.util
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
from tools.global_cube_model_v0 import (  # noqa: E402
    FitResult,
    ProjectedCubeModel,
    fit_projected_cube_model,
    serialize_model,
)
from tools.interior_bezel_detection import (  # noqa: E402
    InteriorBezelDetection,
    detect_interior_bezel_lines,
)


DEFAULT_HARD_MANIFEST = ROOT / "tests" / "fixtures" / "hard_case_manifest.json"
DEFAULT_CORPUS_MANIFEST = ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_OUTPUT_DIR = Path("/tmp/global_cube_model_v0_overlays")
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "global_cube_model_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "GLOBAL_CUBE_MODEL_V0_REPORT.md"
DEFAULT_SET_IDS = ("45", "17", "30", "31", "44", "47", "57", "58", "61")
REQUIRED_OPTIONAL_DEPENDENCIES = ("rembg", "scipy", "onnxruntime")

EXPLICIT_LOCAL_PAIRS: Dict[str, Dict[str, Any]] = {
    "45": {
        "setId": "45",
        "source": "explicit-local",
        "imageAPath": "/Users/jhuber/cube-corpus/Set 45 - A - white up IMG_7169.JPG",
        "imageBPath": "/Users/jhuber/cube-corpus/Set 45 - B - white up IMG_7170.JPG",
    }
}

Point = Tuple[float, float]


def missing_required_optional_dependencies() -> List[str]:
    return [
        name
        for name in REQUIRED_OPTIONAL_DEPENDENCIES
        if importlib.util.find_spec(name) is None
    ]


def generate_global_cube_model_v0_artifacts(
    *,
    set_ids: Sequence[str],
    output_dir: Path,
    hard_manifest_path: Path = DEFAULT_HARD_MANIFEST,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    edge_steps: int = 32,
) -> Dict[str, Any]:
    pairs = load_pairs(hard_manifest_path, corpus_manifest_path)
    rows: List[Dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for set_id in [str(value) for value in set_ids]:
        pair = pairs.get(set_id)
        if pair is None:
            rows.append({"setId": set_id, "status": "set_not_found"})
            continue
        for side, image_key in (("A", "imageAPath"), ("B", "imageBPath")):
            image_path = Path(pair[image_key])
            row = {
                "setId": set_id,
                "side": side,
                "source": pair.get("source"),
                "imagePath": str(image_path),
            }
            if not image_path.exists():
                rows.append({**row, "status": "image_missing"})
                continue
            try:
                image, image_rgb = _load_processing_image(image_path)
                mask = _compute_rembg_mask(image)
                detection = detect_interior_bezel_lines(image_rgb, mask)
                fit = fit_projected_cube_model(detection, mask, edge_steps=edge_steps)
                overlay_path = output_dir / f"set_{set_id}_{side}_global_model_v0.png"
                if fit.model is not None:
                    render_overlay(image, mask, detection, fit, overlay_path)
                rows.append({
                    **row,
                    "status": fit.status,
                    "overlayPath": str(overlay_path) if fit.model is not None else None,
                    "detection": compact_detection(detection),
                    "fitDiagnostics": fit.diagnostics,
                    "model": serialize_model(fit.model) if fit.model is not None else None,
                })
            except Exception as exc:  # pragma: no cover - exercised by local CLI/deps
                rows.append({**row, "status": "error", "error": f"{exc.__class__.__name__}: {exc}"})

    document = {
        "schemaVersion": 1,
        "probe": "global_cube_model_v0",
        "description": (
            "Diagnostics-only projected cube model scaffold. One cube center, "
            "three projected axes, one shared edge length; all visible face/cell "
            "quads are generated from that model."
        ),
        "config": {
            "setIds": list(set_ids),
            "edgeSteps": edge_steps,
            "outputDir": str(output_dir),
        },
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    return document


def render_overlay(
    image: Image.Image,
    mask: np.ndarray,
    detection: InteriorBezelDetection,
    fit: FitResult,
    output_path: Path,
) -> None:
    if fit.model is None:
        raise ValueError("cannot render overlay without a fitted model")

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_alpha = (mask.astype(np.uint8) * 38)
    mask_layer.putalpha(Image.fromarray(mask_alpha, mode="L"))
    mask_layer_rgb = Image.new("RGBA", base.size, (0, 180, 220, 0))
    mask_layer_rgb.putalpha(mask_layer.getchannel("A"))
    base = Image.alpha_composite(base, mask_layer_rgb)

    _draw_detection_lines(draw, detection)
    _draw_model(draw, fit.model)
    _draw_text_panel(draw, fit)

    out = Image.alpha_composite(base, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.convert("RGB").save(output_path, quality=92)


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    lines = [
        "# Global Cube Model V0 Diagnostics",
        "",
        "Diagnostics-only first-principles scaffold. This does not alter recognition behavior.",
        "",
        "The invariant under test: every sampled sticker cell should be generated by one coherent projected cube model, not by independent local grid guesses.",
        "",
        "## Summary",
        "",
        f"- Requested pairs: {summary['requestedPairCount']}",
        f"- Image rows: {summary['imageRowCount']}",
        f"- Fitted rows: {summary['fittedRowCount']}",
        f"- OK rows: {summary['okRowCount']}",
        f"- Low-IoU rows: {summary['lowIouRowCount']}",
        f"- Low-inside rows: {summary['lowInsideRowCount']}",
        f"- Low cell-inside rows: {summary['lowCellInsideRowCount']}",
        f"- Error/missing rows: {summary['errorRowCount']}",
        "",
        "## Readout",
        "",
        "| Set | Side | Status | Score | IoU | Inside | Coverage | Cell inside | Overlay |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in document["rows"]:
        model = row.get("model") or {}
        components = model.get("scoreComponents") or {}
        overlay = row.get("overlayPath") or ""
        overlay_text = f"`{overlay}`" if overlay else ""
        lines.append(
            f"| {row.get('setId')} | {row.get('side', '')} | `{row.get('status')}` | "
            f"{model.get('score', '')} | {components.get('silhouetteIoU', '')} | "
            f"{components.get('insideRatio', '')} | {components.get('maskCoverage', '')} | "
            f"{components.get('cellInsideRatio', '')} | {overlay_text} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This V0 fit is intentionally coarse: axes come from the interior-bezel detector and only axis signs plus one shared edge length are searched.",
        "- A good row means the generated model mostly sits inside the rembg cube silhouette and produces plausible model-derived cell quads.",
        "- A weak row is still useful: it tells us whether failure is in center/axis initialization, edge-length search, silhouette agreement, or the model shape itself.",
        "- The next useful human review artifact is the overlay PNG, not a production promotion threshold.",
        "",
    ])
    return "\n".join(lines)


def load_pairs(hard_manifest_path: Path, corpus_manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    pairs: Dict[str, Dict[str, Any]] = {key: dict(value) for key, value in EXPLICIT_LOCAL_PAIRS.items()}
    for source, path in (("hard-case", hard_manifest_path), ("corpus", corpus_manifest_path)):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for pair in data.get("pairs", []):
            set_id = str(pair.get("setId"))
            pairs.setdefault(set_id, {**pair, "source": source})
    return pairs


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    fitted = [row for row in rows if row.get("model")]
    return {
        "requestedPairCount": len({row.get("setId") for row in rows}),
        "imageRowCount": len(rows),
        "fittedRowCount": len(fitted),
        "okRowCount": sum(1 for row in rows if row.get("status") == "ok"),
        "lowIouRowCount": sum(1 for row in rows if row.get("status") == "low_iou"),
        "lowInsideRowCount": sum(1 for row in rows if row.get("status") == "low_inside_ratio"),
        "lowCellInsideRowCount": sum(1 for row in rows if row.get("status") == "low_cell_inside"),
        "errorRowCount": sum(
            1 for row in rows
            if row.get("status") in {"error", "image_missing", "set_not_found"}
        ),
    }


def compact_detection(detection: InteriorBezelDetection) -> Dict[str, Any]:
    return {
        "cubeCenter": (
            [round(float(detection.cube_center[0]), 1), round(float(detection.cube_center[1]), 1)]
            if detection.cube_center is not None else None
        ),
        "boundaryAnglesDeg": [round(math.degrees(float(angle)), 2) for angle in detection.boundary_angles],
        "lineQualities": [round(float(value), 4) for value in detection.line_qualities],
        "signalQuality": round(float(detection.signal_quality), 4),
        "error": detection.debug.get("error"),
    }


def _compute_rembg_mask(image: Image.Image) -> np.ndarray:
    from rembg import new_session, remove  # type: ignore

    session = getattr(_compute_rembg_mask, "_session", None)
    if session is None:
        session = new_session("u2net")
        _compute_rembg_mask._session = session  # type: ignore[attr-defined]
    rgba = remove(image, session=session)
    alpha = np.array(rgba.split()[-1], dtype=np.uint8)
    return alpha > 128


def _draw_detection_lines(draw: ImageDraw.ImageDraw, detection: InteriorBezelDetection) -> None:
    colors = [(255, 0, 255, 235), (0, 220, 255, 235), (255, 215, 0, 235)]
    for idx, line in enumerate(detection.boundary_lines[:3]):
        color = colors[idx % len(colors)]
        draw.line([line[0], line[1]], fill=color, width=4)
    if detection.cube_center is not None:
        x, y = detection.cube_center
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 255, 255, 255), width=3)


def _draw_model(draw: ImageDraw.ImageDraw, model: ProjectedCubeModel) -> None:
    face_colors = [
        (255, 80, 80, 52),
        (80, 220, 120, 52),
        (80, 130, 255, 52),
    ]
    line_colors = [
        (255, 80, 80, 235),
        (80, 220, 120, 235),
        (80, 130, 255, 235),
    ]
    for idx, face in enumerate(model.faces):
        fill = face_colors[idx % len(face_colors)]
        line = line_colors[idx % len(line_colors)]
        draw.polygon(face.quad, fill=fill, outline=line)
        draw.line(face.quad + [face.quad[0]], fill=line, width=4)
        for cell in face.cells:
            quad = cell["quad"]
            draw.line(quad + [quad[0]], fill=(255, 255, 255, 125), width=1)

    cx, cy = model.cube_center
    for axis in model.axes:
        ax, ay = axis
        draw.line([(cx, cy), (cx + ax, cy + ay)], fill=(255, 255, 255, 240), width=3)
    for x, y in model.outer_vertices():
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255, 230))


def _draw_text_panel(draw: ImageDraw.ImageDraw, fit: FitResult) -> None:
    model = fit.model
    if model is None:
        return
    components = model.score_components
    lines = [
        f"global_cube_model_v0: {fit.status}",
        f"score={model.score:.3f} iou={components['silhouetteIoU']:.3f}",
        f"inside={components['insideRatio']:.3f} coverage={components['maskCoverage']:.3f}",
    ]
    x0, y0 = 12, 12
    line_h = 18
    width = 430
    height = 18 + line_h * len(lines)
    draw.rectangle((x0 - 6, y0 - 6, x0 + width, y0 + height), fill=(0, 0, 0, 160))
    for idx, text in enumerate(lines):
        draw.text((x0, y0 + idx * line_h), text, fill=(255, 255, 255, 255))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", nargs="+", default=list(DEFAULT_SET_IDS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--hard-manifest", type=Path, default=DEFAULT_HARD_MANIFEST)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--edge-steps", type=int, default=32)
    args = parser.parse_args(argv)

    missing = missing_required_optional_dependencies()
    if missing:
        deps = ", ".join(missing)
        print(
            "error: render_global_cube_model_v0_overlays.py requires optional "
            f"diagnostic dependencies to regenerate outputs: {deps}.\n"
            "Install them in the repo venv, for example:\n"
            "  .venv/bin/pip install rembg scipy onnxruntime\n"
            "Refusing to write the summary/report because a dependency-light "
            "run would produce an all-error or empty artifact.",
            file=sys.stderr,
        )
        return 2

    document = generate_global_cube_model_v0_artifacts(
        set_ids=args.sets,
        output_dir=args.out_dir,
        hard_manifest_path=args.hard_manifest,
        corpus_manifest_path=args.corpus_manifest,
        edge_steps=args.edge_steps,
    )
    if document["summary"]["fittedRowCount"] <= 0:
        print(
            "error: global cube model V0 produced zero fitted rows; refusing "
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
