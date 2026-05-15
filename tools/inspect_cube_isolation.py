#!/usr/bin/env python3
"""Inspect cube/background separation signals for one image.

This is a diagnostic tool only. It does not change recognizer behavior.

The tool reuses `analyze_image(...)`, selects the most plausible U- or D-
anchored visible-grid assignment, builds a padded convex hull from the selected
grid points, and reports which detected sticker candidates fall inside or
outside that proposed cube region. Use the optional overlay to inspect the
proposed mask visually before turning any cube-isolation idea into recognizer
behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubik_recognizer.image_pipeline import ImageAnalysis, analyze_image  # noqa: E402
from rubik_recognizer.recognizer import (  # noqa: E402
    FACE_ORDER,
    _assigned_grid_by_face,
    _grid_quality_score,
)


Point = Tuple[float, float]
Polygon = List[Point]

FACE_COLORS = {
    "U": (245, 245, 245),
    "R": (230, 60, 50),
    "F": (40, 180, 95),
    "D": (245, 220, 35),
    "L": (255, 140, 30),
    "B": (60, 120, 230),
}

MAX_PROCESS_IMAGE_SIDE = 1150


def convex_hull(points: Sequence[Point]) -> Polygon:
    """Return the convex hull of `points` using the monotonic chain method."""
    unique = sorted({(round(float(x), 4), round(float(y), 4)) for x, y in points})
    if len(unique) <= 1:
        return unique

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: Polygon = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: Polygon = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def expand_polygon(poly: Sequence[Point], padding_fraction: float) -> Polygon:
    """Expand a polygon around its centroid by `padding_fraction`."""
    if not poly:
        return []
    cx = sum(x for x, _ in poly) / len(poly)
    cy = sum(y for _, y in poly) / len(poly)
    scale = 1.0 + padding_fraction
    return [(cx + (x - cx) * scale, cy + (y - cy) * scale) for x, y in poly]


def point_in_polygon(point: Point, poly: Sequence[Point]) -> bool:
    """Return True when point is inside or on the polygon boundary."""
    if len(poly) < 3:
        return False
    x, y = point
    inside = False
    previous = poly[-1]
    for current in poly:
        x1, y1 = previous
        x2, y2 = current
        if _point_on_segment(point, previous, current):
            return True
        if (y1 > y) != (y2 > y):
            x_at_y = (x2 - x1) * (y - y1) / max(1e-9, y2 - y1) + x1
            if x <= x_at_y:
                inside = not inside
        previous = current
    return inside


def isolation_diagnostics_for_analysis(
    analysis: ImageAnalysis,
    *,
    image_path: Optional[Path] = None,
    anchor: str = "auto",
    padding_fraction: float = 0.18,
) -> Dict[str, Any]:
    anchor_choice = _choose_anchor(analysis, anchor)
    assignments = anchor_choice["assignments"]
    hull_points = _selected_grid_points(assignments)
    hull_source = "selectedGridPoints"
    if len(hull_points) < 3:
        hull_points = [tuple(sticker.center) for sticker in analysis.stickers]
        hull_source = "allStickerCenters"
    hull = convex_hull(hull_points)
    padded_hull = expand_polygon(hull, padding_fraction)
    stickers = [_sticker_diagnostic(sticker, padded_hull) for sticker in analysis.stickers]
    keep_count = sum(1 for sticker in stickers if sticker["insideProposedCubeHull"])
    drop_count = len(stickers) - keep_count
    return {
        "schemaVersion": 1,
        "imagePath": str(image_path) if image_path else None,
        "analysis": {
            "width": analysis.width,
            "height": analysis.height,
            "coordinateSpace": "processingImage",
            "processingWidth": _processing_image_size(analysis.width, analysis.height)[0],
            "processingHeight": _processing_image_size(analysis.width, analysis.height)[1],
            "roi": list(analysis.roi),
            "stickerCount": len(analysis.stickers),
            "gridCount": len(analysis.grids),
            "warnings": list(analysis.warnings),
        },
        "anchorRequested": anchor,
        "anchorUsed": anchor_choice["anchor"],
        "anchorCandidates": anchor_choice["candidates"],
        "selectedGridFaces": sorted(assignments),
        "selectedGridIds": {
            face: getattr(grid, "id", None)
            for face, grid in assignments.items()
        },
        "proposedCubeRegion": {
            "hullSource": hull_source,
            "paddingFraction": padding_fraction,
            "hull": _json_points(hull),
            "paddedHull": _json_points(padded_hull),
            "bbox": _polygon_bbox(padded_hull),
            "selectedGridPointCount": len(_selected_grid_points(assignments)),
            "sourcePointCount": len(hull_points),
        },
        "classificationSummary": _classification_summary(stickers),
        "stickers": stickers,
    }


def write_overlay(
    image_path: Path,
    diagnostics: Dict[str, Any],
    output_path: Path,
) -> None:
    """Draw diagnostics in the same resized coordinate space as analyze_image."""
    image = _load_process_image(image_path)
    draw = ImageDraw.Draw(image, "RGBA")
    roi = diagnostics["analysis"]["roi"]
    draw.rectangle(roi, outline=(48, 120, 255, 210), width=3)
    region = diagnostics["proposedCubeRegion"]
    hull = _tuple_points(region["hull"])
    padded = _tuple_points(region["paddedHull"])
    _draw_polygon(draw, hull, fill=(30, 190, 90, 40), outline=(30, 190, 90, 230), width=3)
    _draw_polygon(draw, padded, fill=(255, 190, 40, 30), outline=(255, 190, 40, 230), width=3)
    for sticker in diagnostics["stickers"]:
        x, y = sticker["center"]
        face = sticker.get("face")
        base = FACE_COLORS.get(face, (180, 180, 180))
        if sticker["insideProposedCubeHull"]:
            fill = (*base, 210)
            outline = (20, 120, 60, 255)
        else:
            fill = (240, 60, 60, 200)
            outline = (120, 0, 0, 255)
        radius = 4 if sticker.get("source") == "component" else 3
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _choose_anchor(analysis: ImageAnalysis, requested: str) -> Dict[str, Any]:
    candidates = {
        anchor: _anchor_assignment_summary(analysis, anchor)
        for anchor in ("U", "D")
    }
    if requested in ("U", "D"):
        chosen = requested
    else:
        chosen = max(candidates, key=lambda item: candidates[item]["score"])
    return {
        "anchor": chosen,
        "assignments": _assigned_grid_by_face(analysis, chosen),
        "candidates": {
            anchor: {key: value for key, value in summary.items() if key != "assignments"}
            for anchor, summary in candidates.items()
        },
    }


def _anchor_assignment_summary(analysis: ImageAnalysis, anchor: str) -> Dict[str, Any]:
    assignments = _assigned_grid_by_face(analysis, anchor)
    quality = sum(_grid_quality_score(grid) for grid in assignments.values())
    score = len(assignments) * 100.0 + quality
    if anchor in assignments:
        score += 500.0
    return {
        "assignments": assignments,
        "assignmentCount": len(assignments),
        "anchorPresent": anchor in assignments,
        "score": round(score, 3),
        "faces": sorted(assignments),
    }


def _selected_grid_points(assignments: Dict[str, Any]) -> List[Point]:
    points: List[Point] = []
    for grid in assignments.values():
        for row in getattr(grid, "points", []):
            for point in row:
                points.append((float(point[0]), float(point[1])))
    return points


def _sticker_diagnostic(sticker: Any, hull: Sequence[Point]) -> Dict[str, Any]:
    center = (float(sticker.center[0]), float(sticker.center[1]))
    match = getattr(sticker, "match", None)
    inside = point_in_polygon(center, hull)
    return {
        "id": getattr(sticker, "id", None),
        "center": [round(center[0], 3), round(center[1], 3)],
        "source": getattr(sticker, "source", "unknown") or "unknown",
        "face": getattr(match, "face", None),
        "color": getattr(match, "color", None),
        "confidence": round(float(getattr(match, "confidence", 0.0)), 4),
        "insideProposedCubeHull": inside,
        "proposedAction": "keep" if inside else "drop",
    }


def _classification_summary(stickers: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    kept = [sticker for sticker in stickers if sticker["insideProposedCubeHull"]]
    dropped = [sticker for sticker in stickers if not sticker["insideProposedCubeHull"]]
    return {
        "kept": len(kept),
        "dropped": len(dropped),
        "keptByFace": _count_by(kept, "face"),
        "droppedByFace": _count_by(dropped, "face"),
        "keptBySource": _count_by(kept, "source"),
        "droppedBySource": _count_by(dropped, "source"),
    }


def _count_by(items: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts = Counter(str(item.get(key) or "unknown") for item in items)
    face_order = [*FACE_ORDER, "unknown"]
    ordered = {face: counts.pop(face) for face in face_order if counts.get(face)}
    ordered.update({key: counts[key] for key in sorted(counts)})
    return ordered


def _json_points(points: Sequence[Point]) -> List[List[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in points]


def _tuple_points(points: Sequence[Sequence[float]]) -> Polygon:
    return [(float(x), float(y)) for x, y in points]


def _polygon_bbox(points: Sequence[Point]) -> Optional[List[float]]:
    if not points:
        return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return [round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3)]


def _point_on_segment(point: Point, a: Point, b: Point) -> bool:
    px, py = point
    ax, ay = a
    bx, by = b
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > 1e-6:
        return False
    return (
        min(ax, bx) - 1e-6 <= px <= max(ax, bx) + 1e-6
        and min(ay, by) - 1e-6 <= py <= max(ay, by) + 1e-6
    )


def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Point],
    *,
    fill: Tuple[int, int, int, int],
    outline: Tuple[int, int, int, int],
    width: int,
) -> None:
    if len(points) < 3:
        return
    draw.polygon(points, fill=fill)
    draw.line([*points, points[0]], fill=outline, width=width, joint="curve")


def _load_process_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        image = ImageOps.exif_transpose(img).convert("RGB")
    size = _processing_image_size(image.width, image.height)
    if image.size == size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


def _processing_image_size(width: int, height: int) -> Tuple[int, int]:
    side = max(width, height)
    if side <= MAX_PROCESS_IMAGE_SIDE:
        return width, height
    scale = MAX_PROCESS_IMAGE_SIDE / float(side)
    return int(width * scale), int(height * scale)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect proposed cube/background isolation signals for one image."
    )
    parser.add_argument("image", type=Path, help="Path to a cube photo.")
    parser.add_argument(
        "--anchor",
        choices=("auto", "U", "D"),
        default="auto",
        help="Visible anchor to use for selected-grid hull construction.",
    )
    parser.add_argument(
        "--padding-fraction",
        type=float,
        default=0.18,
        help="Scale the selected-grid convex hull outward by this fraction.",
    )
    parser.add_argument("--json-output", type=Path, help="Optional path to write the diagnostic JSON.")
    parser.add_argument("--overlay-output", type=Path, help="Optional path to write a PNG overlay.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the one-line summary.")
    args = parser.parse_args(argv)

    analysis = analyze_image(args.image.read_bytes())
    diagnostics = isolation_diagnostics_for_analysis(
        analysis,
        image_path=args.image,
        anchor=args.anchor,
        padding_fraction=args.padding_fraction,
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.overlay_output:
        write_overlay(args.image, diagnostics, args.overlay_output)
    if not args.quiet:
        summary = diagnostics["classificationSummary"]
        print(
            "anchor={anchor} stickers={stickers} kept={kept} dropped={dropped} hullSource={source}".format(
                anchor=diagnostics["anchorUsed"],
                stickers=diagnostics["analysis"]["stickerCount"],
                kept=summary["kept"],
                dropped=summary["dropped"],
                source=diagnostics["proposedCubeRegion"]["hullSource"],
            )
        )
    if not args.json_output and args.quiet:
        print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
