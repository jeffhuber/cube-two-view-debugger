#!/usr/bin/env python3
"""Evaluate saved cube-geometry labels against current image analysis.

This is diagnostic-only. It does not change recognizer behavior.

The Geometry Labeler saves EXIF-corrected browser-natural image coordinates.
The recognizer analyzes a resized processing image. This tool bridges those
spaces, then reports how current sticker/ROI/grid evidence lines up with the
human-labelled cube hull and face quads.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubik_recognizer.image_pipeline import ImageAnalysis, analyze_image  # noqa: E402
from rubik_recognizer.recognizer import FACE_ORDER  # noqa: E402
from tools.audit_recognition_pair import file_sha256  # noqa: E402
from tools.inspect_cube_isolation import (  # noqa: E402
    FACE_COLORS,
    convex_hull,
    isolation_diagnostics_for_analysis,
    point_in_polygon,
    _draw_polygon,
    _load_process_image,
    _polygon_bbox,
    _tuple_points,
)
from tools.probe_hard_cases import load_manifest_document, normalize_path  # noqa: E402


Point = Tuple[float, float]
Polygon = List[Point]

DEFAULT_LABELS_DIR = ROOT / "runs" / "labels"
DEFAULT_MANIFESTS = (
    ROOT / "tests" / "fixtures" / "hard_case_manifest.json",
    ROOT / "tests" / "fixtures" / "corpus_manifest.json",
)
DEFAULT_IMAGE_ROOTS = (Path("/Users/jhuber/Downloads"),)


def evaluate_label_file(
    label_path: Path,
    *,
    image_path: Optional[Path] = None,
    manifests: Sequence[Path] = DEFAULT_MANIFESTS,
    image_roots: Sequence[Path] = DEFAULT_IMAGE_ROOTS,
    overlay_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    label_path = label_path.expanduser().resolve()
    document = load_label_document(label_path)
    resolved_image = image_path or resolve_image_path(document, label_path, manifests, image_roots)
    image_bytes = resolved_image.read_bytes()
    analysis = analyze_image(image_bytes)
    metrics = geometry_metrics_for_analysis(
        document,
        analysis,
        image_path=resolved_image,
        label_path=label_path,
    )
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"{label_path.stem}-geometry-metrics.png"
        write_geometry_overlay(resolved_image, metrics, overlay_path)
        metrics["artifacts"]["overlayPath"] = str(overlay_path)
    return metrics


def load_label_document(path: Path) -> Dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Label file {path} must contain a JSON object.")
    labels = document.get("labels")
    image = document.get("image")
    if not isinstance(labels, dict):
        raise ValueError(f"Label file {path} must include labels.")
    if not isinstance(image, dict):
        raise ValueError(f"Label file {path} must include image metadata.")
    return document


def geometry_metrics_for_analysis(
    document: Dict[str, Any],
    analysis: ImageAnalysis,
    *,
    image_path: Optional[Path] = None,
    label_path: Optional[Path] = None,
) -> Dict[str, Any]:
    processing_width, processing_height = processing_image_size(analysis.width, analysis.height)
    geometry = scaled_label_geometry(document, processing_width, processing_height)
    label_hull = label_cube_hull(geometry)
    if len(label_hull) < 3:
        raise ValueError("Label must include a cube hull or at least one face quad.")

    expected_anchor = expected_anchor_for_side(str(document.get("imageSide") or ""))
    isolation = isolation_diagnostics_for_analysis(
        analysis,
        image_path=image_path,
        anchor=expected_anchor,
    )
    sticker_rows = sticker_label_rows(analysis.stickers, label_hull, geometry["faceQuads"])
    summary = sticker_summary(sticker_rows)
    roi_metrics = roi_label_metrics(analysis.roi, label_hull)
    proposed = proposed_region_metrics(
        label_hull,
        isolation["proposedCubeRegion"],
        width=processing_width,
        height=processing_height,
    )
    image = document.get("image") if isinstance(document.get("image"), dict) else {}
    actual_sha = file_sha256(str(image_path)) if image_path else None

    return {
        "schemaVersion": 1,
        "tool": "rubik-two-view-recognizer-geometry-label-evaluator",
        "labelPath": str(label_path) if label_path else None,
        "imagePath": str(image_path) if image_path else None,
        "setId": document.get("setId"),
        "imageSide": document.get("imageSide"),
        "coordinateSpace": {
            "label": document.get("coordinateSpace"),
            "analysis": "processingImage",
            "naturalWidth": image.get("width"),
            "naturalHeight": image.get("height"),
            "processingWidth": processing_width,
            "processingHeight": processing_height,
            "scaleX": geometry["scaleX"],
            "scaleY": geometry["scaleY"],
        },
        "imageSha256": {
            "label": image.get("sha256"),
            "actual": actual_sha,
            "matches": image.get("sha256") in (None, "", actual_sha),
        },
        "analysis": {
            "width": analysis.width,
            "height": analysis.height,
            "roi": list(analysis.roi),
            "stickerCount": len(analysis.stickers),
            "gridCount": len(analysis.grids),
            "warnings": list(analysis.warnings),
        },
        "labels": {
            "faceLabels": sorted(geometry["faceQuads"], key=FACE_ORDER.index),
            "faceQuadCount": len(geometry["faceQuads"]),
            "cubeHullPointCount": len(geometry["cubeHull"]),
            "cubeHullSource": geometry["cubeHullSource"],
            "cubeHull": json_points(label_hull),
            "faceQuads": {
                face: json_points(points)
                for face, points in geometry["faceQuads"].items()
            },
        },
        "metrics": {
            "stickers": summary,
            "faceCoverage": face_coverage(sticker_rows, geometry["faceQuads"]),
            "roi": roi_metrics,
            "proposedCubeRegion": proposed,
        },
        "stickers": sticker_rows,
        "artifacts": {},
    }


def scaled_label_geometry(
    document: Dict[str, Any],
    processing_width: int,
    processing_height: int,
) -> Dict[str, Any]:
    image = document.get("image") if isinstance(document.get("image"), dict) else {}
    labels = document.get("labels") if isinstance(document.get("labels"), dict) else {}
    natural_width = float(image.get("width") or processing_width)
    natural_height = float(image.get("height") or processing_height)
    scale_x = processing_width / max(1.0, natural_width)
    scale_y = processing_height / max(1.0, natural_height)
    face_quads_raw = labels.get("faceQuads") if isinstance(labels.get("faceQuads"), dict) else {}
    face_quads: Dict[str, Polygon] = {}
    for face in FACE_ORDER:
        points = face_quads_raw.get(face)
        if isinstance(points, list) and len(points) == 4:
            face_quads[face] = [scale_point(point, scale_x, scale_y) for point in points]
    cube_hull_raw = labels.get("cubeHull") if isinstance(labels.get("cubeHull"), list) else []
    cube_hull = [
        scale_point(point, scale_x, scale_y)
        for point in cube_hull_raw
        if isinstance(point, dict)
    ]
    return {
        "scaleX": round(scale_x, 6),
        "scaleY": round(scale_y, 6),
        "faceQuads": face_quads,
        "cubeHull": cube_hull,
        "cubeHullSource": "cubeHull" if len(cube_hull) >= 3 else "faceQuadConvexHull",
    }


def scale_point(point: Dict[str, Any], scale_x: float, scale_y: float) -> Point:
    return (float(point["x"]) * scale_x, float(point["y"]) * scale_y)


def label_cube_hull(geometry: Dict[str, Any]) -> Polygon:
    cube_hull = geometry.get("cubeHull") or []
    if len(cube_hull) >= 3:
        return list(cube_hull)
    points: List[Point] = []
    for face_points in geometry.get("faceQuads", {}).values():
        points.extend(face_points)
    return convex_hull(points)


def expected_anchor_for_side(image_side: str) -> str:
    side = image_side.upper()
    if side == "A":
        return "U"
    if side == "B":
        return "D"
    return "auto"


def sticker_label_rows(
    stickers: Sequence[Any],
    label_hull: Sequence[Point],
    face_quads: Dict[str, Sequence[Point]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sticker in stickers:
        center = (float(sticker.center[0]), float(sticker.center[1]))
        matched_faces = [
            face
            for face in FACE_ORDER
            if face in face_quads and point_in_polygon(center, face_quads[face])
        ]
        match = getattr(sticker, "match", None)
        inside_cube = point_in_polygon(center, label_hull)
        rows.append(
            {
                "id": getattr(sticker, "id", None),
                "center": json_point(center),
                "source": getattr(sticker, "source", "unknown") or "unknown",
                "detectedFace": getattr(match, "face", None),
                "detectedColor": getattr(match, "color", None),
                "confidence": round(float(getattr(match, "confidence", 0.0)), 4),
                "insideLabeledCubeHull": inside_cube,
                "insideAnyFaceQuad": bool(matched_faces),
                "labelFace": matched_faces[0] if matched_faces else None,
                "labelFaceHits": matched_faces,
            }
        )
    return rows


def sticker_summary(stickers: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    inside_cube = [row for row in stickers if row["insideLabeledCubeHull"]]
    outside_cube = [row for row in stickers if not row["insideLabeledCubeHull"]]
    inside_face = [row for row in stickers if row["insideAnyFaceQuad"]]
    return {
        "detected": len(stickers),
        "insideLabeledCubeHull": len(inside_cube),
        "outsideLabeledCubeHull": len(outside_cube),
        "insideAnyFaceQuad": len(inside_face),
        "outsideLabeledCubeByDetectedFace": ordered_counts(row.get("detectedFace") for row in outside_cube),
        "outsideLabeledCubeBySource": ordered_counts(row.get("source") for row in outside_cube),
    }


def face_coverage(
    stickers: Sequence[Dict[str, Any]],
    face_quads: Dict[str, Sequence[Point]],
) -> Dict[str, Dict[str, Any]]:
    coverage: Dict[str, Dict[str, Any]] = {}
    for face in FACE_ORDER:
        if face not in face_quads:
            continue
        rows = [row for row in stickers if row.get("labelFace") == face]
        coverage[face] = {
            "detectedCenters": len(rows),
            "coverageVsNine": round(len(rows) / 9.0, 4),
            "detectedFaceCounts": ordered_counts(row.get("detectedFace") for row in rows),
            "detectedSourceCounts": ordered_counts(row.get("source") for row in rows),
        }
    return coverage


def roi_label_metrics(roi: Sequence[int], label_hull: Sequence[Point]) -> Dict[str, Any]:
    roi_poly = [
        (float(roi[0]), float(roi[1])),
        (float(roi[2]), float(roi[1])),
        (float(roi[2]), float(roi[3])),
        (float(roi[0]), float(roi[3])),
    ]
    return {
        "roi": list(roi),
        "labelCubeHullBbox": _polygon_bbox(label_hull),
        "containsAllLabelHullVertices": all(point_in_polygon(point, roi_poly) for point in label_hull),
        "labelHullVerticesOutsideRoi": [
            json_point(point) for point in label_hull if not point_in_polygon(point, roi_poly)
        ],
    }


def proposed_region_metrics(
    label_hull: Sequence[Point],
    proposed: Dict[str, Any],
    *,
    width: int,
    height: int,
) -> Dict[str, Any]:
    hull = _tuple_points(proposed.get("hull") or [])
    padded = _tuple_points(proposed.get("paddedHull") or [])
    return {
        "hullSource": proposed.get("hullSource"),
        "selectedGridPointCount": proposed.get("selectedGridPointCount"),
        "sourcePointCount": proposed.get("sourcePointCount"),
        "iouWithLabelCubeHull": polygon_iou(label_hull, hull, width, height),
        "paddedIouWithLabelCubeHull": polygon_iou(label_hull, padded, width, height),
        "bbox": proposed.get("bbox"),
    }


def polygon_iou(
    first: Sequence[Point],
    second: Sequence[Point],
    width: int,
    height: int,
) -> Optional[float]:
    if len(first) < 3 or len(second) < 3:
        return None
    first_mask = polygon_mask(first, width, height)
    second_mask = polygon_mask(second, width, height)
    intersection = np.logical_and(first_mask, second_mask).sum()
    union = np.logical_or(first_mask, second_mask).sum()
    if not union:
        return None
    return round(float(intersection / union), 4)


def polygon_mask(points: Sequence[Point], width: int, height: int) -> np.ndarray:
    image = Image.new("1", (max(1, int(width)), max(1, int(height))), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon([(float(x), float(y)) for x, y in points], fill=1)
    return np.asarray(image, dtype=bool)


def resolve_image_path(
    document: Dict[str, Any],
    label_path: Path,
    manifests: Sequence[Path],
    image_roots: Sequence[Path],
) -> Path:
    image = document.get("image") if isinstance(document.get("image"), dict) else {}
    label_sha = image.get("sha256")
    candidates = image_candidates(document, label_path, manifests, image_roots)
    if not candidates:
        raise FileNotFoundError(
            f"Could not resolve image for label {label_path}; pass --image explicitly."
        )
    if label_sha:
        for candidate in candidates:
            if candidate.exists() and file_sha256(str(candidate)) == label_sha:
                return candidate
    existing = [candidate for candidate in candidates if candidate.exists()]
    if len(existing) == 1:
        return existing[0]
    if existing:
        names = ", ".join(str(path) for path in existing[:5])
        raise ValueError(f"Multiple possible images for {label_path}: {names}")
    raise FileNotFoundError(f"No candidate image path exists for label {label_path}.")


def image_candidates(
    document: Dict[str, Any],
    label_path: Path,
    manifests: Sequence[Path],
    image_roots: Sequence[Path],
) -> List[Path]:
    image = document.get("image") if isinstance(document.get("image"), dict) else {}
    set_id = normalize_set_id(document.get("setId"))
    side = str(document.get("imageSide") or image.get("side") or "").upper()
    name = image.get("name")
    candidates: List[Path] = []
    for manifest in manifests:
        if not manifest.exists():
            continue
        doc = load_manifest_document(manifest)
        for row in doc.get("pairs", []):
            if set_id and normalize_set_id(row.get("setId") or row.get("id")) != set_id:
                continue
            for key in image_keys_for_side(side):
                value = row.get(key)
                if value:
                    candidates.append(normalize_path(str(value), manifest))
    if name:
        for root in [label_path.parent, *image_roots]:
            if root:
                candidates.append((root / str(name)).expanduser())
    return unique_paths(candidates)


def image_keys_for_side(side: str) -> List[str]:
    if side == "A":
        return ["imageAPath"]
    if side == "B":
        return ["imageBPath"]
    return ["imageAPath", "imageBPath", "imagePath"]


def normalize_set_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("set "):
        return text[4:].strip()
    return text


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for path in paths:
        resolved = path.expanduser()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def write_geometry_overlay(
    image_path: Path,
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    image = _load_process_image(image_path)
    draw = ImageDraw.Draw(image, "RGBA")
    roi = metrics["analysis"]["roi"]
    draw.rectangle(roi, outline=(48, 120, 255, 210), width=3)
    labels = metrics["labels"]
    label_hull = _tuple_points(labels["cubeHull"])
    _draw_polygon(draw, label_hull, fill=(0, 200, 220, 42), outline=(0, 120, 160, 235), width=4)
    for face, points in labels["faceQuads"].items():
        color = FACE_COLORS.get(face, (255, 0, 255))
        rgba = (*color, 225)
        _draw_polygon(draw, _tuple_points(points), fill=(*color, 35), outline=rgba, width=3)
    for sticker in metrics["stickers"]:
        x, y = sticker["center"]
        if sticker["insideAnyFaceQuad"]:
            fill = (30, 190, 90, 220)
            outline = (0, 90, 30, 255)
        elif sticker["insideLabeledCubeHull"]:
            fill = (250, 190, 40, 210)
            outline = (120, 80, 0, 255)
        else:
            fill = (240, 60, 60, 210)
            outline = (120, 0, 0, 255)
        radius = 5 if sticker.get("source") == "component" else 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_json(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    payload = {
        "schemaVersion": 1,
        "results": list(results),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_table(results: Sequence[Dict[str, Any]]) -> None:
    print("Set Side Stk InCube OutCube FacePts ROI HullIoU PadIoU Label")
    print("--- ---- --- ------ ------- ------- --- ------- ------ ----------------")
    for result in results:
        metrics = result["metrics"]
        stickers = metrics["stickers"]
        roi = metrics["roi"]
        proposed = metrics["proposedCubeRegion"]
        print(
            f"{str(result.get('setId') or '-'):>3} "
            f"{str(result.get('imageSide') or '-'):>4} "
            f"{stickers['detected']:3d} "
            f"{stickers['insideLabeledCubeHull']:6d} "
            f"{stickers['outsideLabeledCubeHull']:7d} "
            f"{stickers['insideAnyFaceQuad']:7d} "
            f"{'yes' if roi['containsAllLabelHullVertices'] else 'no':>3} "
            f"{format_optional(proposed['iouWithLabelCubeHull']):>7} "
            f"{format_optional(proposed['paddedIouWithLabelCubeHull']):>6} "
            f"{Path(str(result.get('labelPath') or '-')).name}"
        )


def ordered_counts(values: Iterable[Any]) -> Dict[str, int]:
    counts = Counter(str(value or "unknown") for value in values)
    ordered = {face: counts.pop(face) for face in FACE_ORDER if counts.get(face)}
    if counts.get("unknown"):
        ordered["unknown"] = counts.pop("unknown")
    ordered.update({key: counts[key] for key in sorted(counts)})
    return ordered


def processing_image_size(width: int, height: int, max_side: int = 1150) -> Tuple[int, int]:
    side = max(width, height)
    if side <= max_side:
        return width, height
    scale = max_side / float(side)
    return int(width * scale), int(height * scale)


def json_point(point: Point) -> List[float]:
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def json_points(points: Sequence[Point]) -> List[List[float]]:
    return [json_point(point) for point in points]


def format_optional(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def default_label_paths() -> List[Path]:
    if not DEFAULT_LABELS_DIR.exists():
        return []
    return sorted(DEFAULT_LABELS_DIR.glob("*.json"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "labels",
        nargs="*",
        type=Path,
        help="Saved label JSON files. Defaults to runs/labels/*.json.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Source image path. Only valid when evaluating one label file.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        type=Path,
        help="Manifest path used to resolve label images. Defaults to hard-case + corpus manifests.",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        type=Path,
        help="Directory to check for the labelled image filename.",
    )
    parser.add_argument("--json-output", type=Path, help="Optional path for full JSON output.")
    parser.add_argument("--overlay-dir", type=Path, help="Optional directory for PNG overlays.")
    parser.add_argument("--quiet", action="store_true", help="Do not print the readable table.")
    args = parser.parse_args(argv)

    label_paths = args.labels or default_label_paths()
    if not label_paths:
        raise SystemExit("No label JSON files found. Pass one or save labels under runs/labels/.")
    if args.image and len(label_paths) != 1:
        raise SystemExit("--image can only be used with one label JSON file.")

    manifests = tuple(args.manifest) if args.manifest else DEFAULT_MANIFESTS
    image_roots = tuple(args.image_root) if args.image_root else DEFAULT_IMAGE_ROOTS
    results = [
        evaluate_label_file(
            path,
            image_path=args.image,
            manifests=manifests,
            image_roots=image_roots,
            overlay_dir=args.overlay_dir,
        )
        for path in label_paths
    ]
    if args.json_output:
        write_json(args.json_output, results)
    if not args.quiet:
        print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
