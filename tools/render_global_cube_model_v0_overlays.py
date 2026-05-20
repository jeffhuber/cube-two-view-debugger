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
    fit_projected_cube_model_v01,
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
EASY_CORPUS_SET_IDS = ("15", "23", "26", "29", "32", "36", "37", "42")
FIT_VERSION_CHOICES = ("v0", "v0.1")
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
    profile: str = "custom",
    fit_version: str = "v0",
    side_filter: Optional[Sequence[Tuple[str, str]]] = None,
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
                fit = fit_projected_cube_model_for_version(
                    detection,
                    mask,
                    edge_steps=edge_steps,
                    fit_version=fit_version,
                )
                overlay_path = output_dir / overlay_filename(set_id, side, fit_version)
                if fit.model is not None:
                    render_overlay(image, mask, detection, fit, overlay_path)
                status = fit.status
                rows.append({
                    **row,
                    "status": status,
                    "diagnosticDisposition": diagnostic_disposition(row["evaluationTier"], status),
                    "overlayPath": str(overlay_path) if fit.model is not None else None,
                    "detection": compact_detection(detection),
                    "fitDiagnostics": fit.diagnostics,
                    "model": serialize_model(fit.model) if fit.model is not None else None,
                })
            except Exception as exc:  # pragma: no cover - exercised by local CLI/deps
                rows.append({
                    **row,
                    "status": "error",
                    "diagnosticDisposition": diagnostic_disposition(row["evaluationTier"], "error"),
                    "error": f"{exc.__class__.__name__}: {exc}",
                })

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
            "profile": profile,
            "fitVersion": fit_version,
            "sideFilter": (
                [f"{set_id}:{side}" for set_id, side in sorted(normalized_side_filter)]
                if normalized_side_filter is not None
                else None
            ),
        },
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    return document


def fit_projected_cube_model_for_version(
    detection: InteriorBezelDetection,
    mask: np.ndarray,
    *,
    edge_steps: int,
    fit_version: str,
) -> FitResult:
    if fit_version == "v0":
        return fit_projected_cube_model(detection, mask, edge_steps=edge_steps)
    if fit_version == "v0.1":
        return fit_projected_cube_model_v01(detection, mask, edge_steps=edge_steps)
    raise ValueError(f"unsupported fit_version: {fit_version}")


def overlay_filename(set_id: str, side: str, fit_version: str) -> str:
    suffix = "v0" if fit_version == "v0" else fit_version.replace(".", "")
    return f"set_{set_id}_{side}_global_model_{suffix}.png"


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
    config = document.get("config") or {}
    fit_version = str(config.get("fitVersion") or "v0")
    title_version = fit_version.upper()
    fit_shape_note = (
        "This V0 fit is intentionally coarse: axes come from the interior-bezel "
        "detector and only axis signs plus one shared edge length are searched."
        if fit_version == "v0"
        else "This V0.1 fit keeps the same coarse model but adds bounded center "
        "refinement around the detector center before choosing the best "
        "threshold-passing candidate."
    )
    lines = [
        f"# Global Cube Model {title_version} Diagnostics",
        "",
        "Diagnostics-only first-principles scaffold. This does not alter recognition behavior.",
        "",
        "The invariant under test: every sampled sticker cell should be generated by one coherent projected cube model, not by independent local grid guesses.",
        "",
        "## Summary",
        "",
        f"- Fit version: {fit_version}",
        f"- Requested pairs: {summary['requestedPairCount']}",
        f"- Image rows: {summary['imageRowCount']}",
        f"- Fitted rows: {summary['fittedRowCount']}",
        f"- OK rows: {summary['okRowCount']}",
        f"- Low-IoU rows: {summary['lowIouRowCount']}",
        f"- Low-inside rows: {summary['lowInsideRowCount']}",
        f"- Low cell-inside rows: {summary['lowCellInsideRowCount']}",
        f"- Error/missing rows: {summary['errorRowCount']}",
        f"- Easy-corpus OK rows: {summary['easyOkRowCount']} / {summary['easyImageRowCount']}",
        f"- Easy-corpus weak rows: {summary['easyWeakRowCount']}",
        f"- Retake/segmentation-candidate rows: {summary['retakeCandidateRowCount']}",
        f"- Model-iteration-needed rows: {summary['modelIterationNeededRowCount']}",
        "",
        "## Readout",
        "",
        "| Set | Side | Tier | Status | Disposition | Score | IoU | Inside | Coverage | Cell inside | Overlay |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in document["rows"]:
        model = row.get("model") or {}
        components = model.get("scoreComponents") or {}
        overlay = row.get("overlayPath") or ""
        overlay_text = f"`{overlay}`" if overlay else ""
        lines.append(
            f"| {row.get('setId')} | {row.get('side', '')} | `{row.get('evaluationTier', '')}` | "
            f"`{row.get('status')}` | `{row.get('diagnosticDisposition', '')}` | "
            f"{model.get('score', '')} | {components.get('silhouetteIoU', '')} | "
            f"{components.get('insideRatio', '')} | {components.get('maskCoverage', '')} | "
            f"{components.get('cellInsideRatio', '')} | {overlay_text} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- {fit_shape_note}",
        "- Easy-corpus rows are clean success cases from `tests/fixtures/corpus_manifest.json`; the model should become boringly consistent there before harder backgrounds are used as success targets.",
        "- Weak rows outside the easy tier should be read as retake/segmentation candidates when the silhouette or background is confusing.",
        "- A weak easy-corpus row is more serious: it points at center/axis initialization, edge-length search, silhouette agreement, or the model shape itself.",
        "- The `diagnosticDisposition` field encodes that split directly: `model_iteration_needed` for easy-corpus weak rows, and `geometry_retake_or_segmentation_candidate` for weak non-easy rows.",
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


def evaluation_tier(pair: Dict[str, Any]) -> str:
    if (
        pair.get("source") == "corpus"
        and pair.get("expectedCategory") == "success_clean"
        and pair.get("expectedScoreFloor") == 54
        and pair.get("currentScoreObserved") == 54
    ):
        return "easy_corpus"
    if pair.get("source") == "hard-case":
        return "hard_case_stress"
    if pair.get("source") == "explicit-local":
        return "local_example"
    return "corpus_stress"


def diagnostic_disposition(evaluation_tier_value: str, status: str) -> str:
    if status == "ok":
        return "model_ok"
    if status in {"error", "image_missing", "set_not_found"}:
        return "input_or_dependency_error"
    if evaluation_tier_value == "easy_corpus":
        return "model_iteration_needed"
    return "geometry_retake_or_segmentation_candidate"


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    fitted = [row for row in rows if row.get("model")]
    easy = [row for row in rows if row.get("evaluationTier") == "easy_corpus"]
    easy_weak = [
        row for row in easy
        if row.get("status") not in {"ok"}
    ]
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
        "easyImageRowCount": len(easy),
        "easyOkRowCount": sum(1 for row in easy if row.get("status") == "ok"),
        "easyWeakRowCount": len(easy_weak),
        "retakeCandidateRowCount": sum(
            1 for row in rows
            if row.get("diagnosticDisposition") == "geometry_retake_or_segmentation_candidate"
        ),
        "modelIterationNeededRowCount": sum(
            1 for row in rows
            if row.get("diagnosticDisposition") == "model_iteration_needed"
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


def parse_side_filter(tokens: Optional[Sequence[str]]) -> Optional[Tuple[Tuple[str, str], ...]]:
    if tokens is None:
        return None
    parsed: List[Tuple[str, str]] = []
    for token in tokens:
        if ":" not in token:
            raise ValueError(f"side filter must use SET:SIDE form, got {token!r}")
        set_id, side = token.split(":", 1)
        side = side.upper()
        if side not in {"A", "B"} or not set_id:
            raise ValueError(f"side filter must use SET:A or SET:B form, got {token!r}")
        parsed.append((set_id, side))
    return tuple(parsed)


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
    fit_version = fit.diagnostics.get("fitVersion", "v0")
    lines = [
        f"global_cube_model_{fit_version}: {fit.status}",
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
    parser.add_argument("--sets", nargs="+", default=None)
    parser.add_argument(
        "--profile",
        choices=("v0-hard-mix", "easy-corpus", "custom"),
        default="v0-hard-mix",
        help="Named set selection. --sets overrides the named profile and uses custom.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--hard-manifest", type=Path, default=DEFAULT_HARD_MANIFEST)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--edge-steps", type=int, default=32)
    parser.add_argument(
        "--fit-version",
        choices=FIT_VERSION_CHOICES,
        default="v0",
        help="Projected cube fitter version to run.",
    )
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
            "error: render_global_cube_model_v0_overlays.py requires optional "
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
    elif profile == "easy-corpus":
        set_ids = list(EASY_CORPUS_SET_IDS)
    else:
        set_ids = list(DEFAULT_SET_IDS)

    try:
        side_filter = parse_side_filter(args.only_sides)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    document = generate_global_cube_model_v0_artifacts(
        set_ids=set_ids,
        output_dir=args.out_dir,
        hard_manifest_path=args.hard_manifest,
        corpus_manifest_path=args.corpus_manifest,
        edge_steps=args.edge_steps,
        profile=profile,
        fit_version=args.fit_version,
        side_filter=side_filter,
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
