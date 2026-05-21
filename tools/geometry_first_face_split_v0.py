#!/usr/bin/env python3
"""Evaluate geometry-first face splitting from fitted cube models.

Diagnostics/data-only. This module does not alter recognizer behavior.

The premise is deliberately first-principles: once a whole-cube silhouette,
visible trihedral vertex, and three projected cube axes are trusted, the three
visible faces should be split by geometry, not by semantic segmentation
prompts. This probe measures whether the fitted models already contain usable
face quads and whether vertex accuracy is the gating failure.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEEDBACK = Path("/Users/jhuber/Downloads/gcm_refined_feedback.json")
DEFAULT_REMBG_DIR = Path("/tmp/gcm_refined")
DEFAULT_SAM3_DIR = Path("/tmp/gcm_sam3")
DEFAULT_SUMMARY = ROOT / "tests" / "fixtures" / "geometry_first_face_split_v0_summary.json"
DEFAULT_REPORT = ROOT / "tools" / "GEOMETRY_FIRST_FACE_SPLIT_V0_REPORT.md"
STRICT_VERTEX_PX = 30.0
PLAUSIBLE_VERTEX_PX = 50.0
MIN_FACE_AREA_RATIO = 0.005
MIN_CELL_AREA_RATIO = 0.0002
MAX_FACE_OVERLAP_RATIO = 0.03
RASTER_MAX_DIM = 600

Point = Tuple[float, float]
Quad = List[Point]


def generate_geometry_split_bakeoff(
    *,
    feedback_path: Path = DEFAULT_FEEDBACK,
    rembg_dir: Path = DEFAULT_REMBG_DIR,
    sam3_dir: Path = DEFAULT_SAM3_DIR,
) -> Dict[str, Any]:
    feedback = _read_json(feedback_path)
    source_dirs = {"rembg": rembg_dir, "sam3": sam3_dir}
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for key, feedback_row in sorted(feedback.items()):
        truth = _point_or_none(feedback_row.get("true_vertex"))
        if truth is None:
            skipped.append({"key": key, "reason": "missing_true_vertex"})
            continue
        try:
            set_id, side = key.split("_", 1)
        except ValueError:
            skipped.append({"key": key, "reason": "bad_feedback_key"})
            continue

        source_results: Dict[str, Dict[str, Any]] = {}
        missing_source = False
        for source, directory in source_dirs.items():
            path = directory / f"set_{set_id}_{side}_data.json"
            if not path.exists():
                source_results[source] = {
                    "source": source,
                    "status": "missing_fit_output",
                    "path": str(path),
                }
                missing_source = True
                continue
            data = _read_json(path)
            source_results[source] = _evaluate_source(source, data, truth, path)

        if missing_source:
            skipped.append({
                "key": key,
                "reason": "missing_fit_output",
                "sources": source_results,
            })
            continue

        best_source = _best_source(source_results)
        rows.append({
            "key": key,
            "setId": set_id,
            "side": side,
            "trueVertex": _round_point(truth),
            "sources": source_results,
            "bestSourceByVertexError": best_source,
            "bestSourceStatus": source_results[best_source]["status"] if best_source else "none",
        })

    return {
        "schemaVersion": 1,
        "probe": "geometry_first_face_split_v0",
        "description": (
            "Diagnostics/data-only geometry-first face split evaluation. "
            "Face quads and 3x3 cells are generated from fitted cube model "
            "geometry, not semantic face masks."
        ),
        "source": {
            "feedbackPath": str(feedback_path),
            "rembgFitDir": str(rembg_dir),
            "sam3FitDir": str(sam3_dir),
        },
        "config": {
            "strictVertexPx": STRICT_VERTEX_PX,
            "plausibleVertexPx": PLAUSIBLE_VERTEX_PX,
            "minFaceAreaRatio": MIN_FACE_AREA_RATIO,
            "minCellAreaRatio": MIN_CELL_AREA_RATIO,
            "maxFaceOverlapRatio": MAX_FACE_OVERLAP_RATIO,
            "rasterMaxDim": RASTER_MAX_DIM,
        },
        "summary": _summarize(rows, skipped),
        "rows": rows,
        "skipped": skipped,
    }


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    lines = [
        "# Geometry-First Face Split V0",
        "",
        "Diagnostics/data-only artifact. This does not alter recognition behavior.",
        "",
        "This report evaluates the face-splitting step that should replace semantic `top_face`/`left_face`/`right_face` masks: use one fitted cube model to generate the three visible face quads and their 3x3 cell grids.",
        "",
        "## Summary",
        "",
        f"- Compared rows: {summary['rowCount']}",
        f"- Skipped rows: {summary['skippedCount']}",
        f"- Strict vertex threshold: {STRICT_VERTEX_PX:.0f} px",
        f"- Plausible vertex threshold: {PLAUSIBLE_VERTEX_PX:.0f} px",
        "",
        "| Source | Rows | Nondegenerate splits | Strict-ready | Plausible | Vertex-blocked | Mean vertex error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source in ("rembg", "sam3", "oracle_best_source"):
        source_summary = summary["sources"][source]
        lines.append(
            f"| `{source}` | {source_summary['rowCount']} | {source_summary['nondegenerateCount']} | "
            f"{source_summary['strictReadyCount']} | {source_summary['plausibleCount']} | "
            f"{source_summary['vertexBlockedCount']} | {source_summary['meanVertexErrorPx']:.1f} px |"
        )

    lines.extend([
        "",
        "## Rows",
        "",
        "| Row | Best source | rembg status/error | SAM3 status/error |",
        "|---|---|---|---|",
    ])
    for row in document["rows"]:
        rembg = row["sources"]["rembg"]
        sam3 = row["sources"]["sam3"]
        lines.append(
            f"| `{row['key']}` | `{row['bestSourceByVertexError']}` | "
            f"`{rembg['status']}` / {rembg.get('vertexErrorPx', ''):.0f} px | "
            f"`{sam3['status']}` / {sam3.get('vertexErrorPx', ''):.0f} px |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Geometry-first splitting is mechanically viable: the fitted models usually produce nondegenerate face quads and cells.",
        "- The blocker is still upstream vertex/axis selection. When the vertex is off by more than roughly 50 px, the face split is not trustworthy even if the quads are well-formed.",
        "- SAM3 whole-cube masks improve the strict-ready count versus rembg in this local bakeoff, but the oracle across the two sources is still far below a production threshold.",
        "- Next work should improve source selection / vertex confidence before any rectified color read path uses these generated face grids.",
        "",
    ])
    return "\n".join(lines)


def _evaluate_source(source: str, data: Dict[str, Any], truth: Point, path: Path) -> Dict[str, Any]:
    vertex = _point_or_none(data.get("cube_center_screen"))
    image_size = _image_size(data)
    quads = _derive_face_quads(data)
    if vertex is None:
        return {"source": source, "status": "missing_vertex", "path": str(path)}
    if not quads:
        return {
            "source": source,
            "status": "missing_face_quads",
            "path": str(path),
            "vertexErrorPx": round(_distance(vertex, truth), 2),
        }
    if image_size is None:
        return {
            "source": source,
            "status": "missing_image_size",
            "path": str(path),
            "vertexErrorPx": round(_distance(vertex, truth), 2),
        }

    width, height = image_size
    image_area = max(1.0, float(width * height))
    face_areas = [_polygon_area(quad) for quad in quads.values()]
    cells = [cell for quad in quads.values() for cell in _subdivide_quad(quad)]
    cell_areas = [_polygon_area(cell) for cell in cells]
    min_face_area_ratio = min(face_areas) / image_area if face_areas else 0.0
    min_cell_area_ratio = min(cell_areas) / image_area if cell_areas else 0.0
    overlap_ratio = _face_overlap_ratio(quads, image_size)
    vertex_error = _distance(vertex, truth)
    nondegenerate = (
        len(quads) == 3
        and len(cells) == 27
        and min_face_area_ratio >= MIN_FACE_AREA_RATIO
        and min_cell_area_ratio >= MIN_CELL_AREA_RATIO
        and overlap_ratio <= MAX_FACE_OVERLAP_RATIO
    )
    status = _split_status(vertex_error, nondegenerate)
    debug = data.get("debug") or {}
    return {
        "source": source,
        "status": status,
        "path": str(path),
        "vertex": _round_point(vertex),
        "vertexErrorPx": round(vertex_error, 2),
        "faceCount": len(quads),
        "cellCount": len(cells),
        "nondegenerate": nondegenerate,
        "minFaceAreaRatio": round(min_face_area_ratio, 6),
        "minCellAreaRatio": round(min_cell_area_ratio, 6),
        "faceOverlapRatio": round(overlap_ratio, 6),
        "fitQuality": data.get("fit_quality"),
        "fitResidualRmsPx": debug.get("fit_residual_rms_px"),
        "refinement": debug.get("refinement"),
        "cubeCenterSource": debug.get("cube_center_source"),
    }


def _summarize(rows: Sequence[Dict[str, Any]], skipped: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    source_rows: Dict[str, List[Dict[str, Any]]] = {"rembg": [], "sam3": [], "oracle_best_source": []}
    for row in rows:
        for source in ("rembg", "sam3"):
            source_rows[source].append(row["sources"][source])
        best = row.get("bestSourceByVertexError")
        if best:
            source_rows["oracle_best_source"].append(row["sources"][best])

    return {
        "rowCount": len(rows),
        "skippedCount": len(skipped),
        "sources": {
            source: _summarize_source(values)
            for source, values in source_rows.items()
        },
    }


def _summarize_source(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    vertex_errors = [
        float(row["vertexErrorPx"])
        for row in rows
        if isinstance(row.get("vertexErrorPx"), (int, float))
    ]
    return {
        "rowCount": len(rows),
        "nondegenerateCount": sum(1 for row in rows if row.get("nondegenerate")),
        "strictReadyCount": sum(1 for row in rows if row.get("status") == "split_ready_strict"),
        "plausibleCount": sum(1 for row in rows if row.get("status") in {"split_ready_strict", "split_plausible"}),
        "vertexBlockedCount": sum(1 for row in rows if row.get("status") == "vertex_error_blocked"),
        "degenerateCount": sum(1 for row in rows if row.get("status") == "degenerate_split"),
        "meanVertexErrorPx": round(float(statistics.fmean(vertex_errors)), 4) if vertex_errors else None,
        "medianVertexErrorPx": round(float(statistics.median(vertex_errors)), 4) if vertex_errors else None,
    }


def _best_source(source_results: Dict[str, Dict[str, Any]]) -> Optional[str]:
    candidates = [
        (source, result)
        for source, result in source_results.items()
        if isinstance(result.get("vertexErrorPx"), (int, float))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item[1]["vertexErrorPx"]))[0]


def _split_status(vertex_error: float, nondegenerate: bool) -> str:
    if not nondegenerate:
        return "degenerate_split"
    if vertex_error <= STRICT_VERTEX_PX:
        return "split_ready_strict"
    if vertex_error <= PLAUSIBLE_VERTEX_PX:
        return "split_plausible"
    return "vertex_error_blocked"


def _derive_face_quads(data: Dict[str, Any]) -> Dict[str, Quad]:
    existing = data.get("face_quads")
    if isinstance(existing, dict) and existing:
        return {
            str(name): [_as_point(point) for point in quad]
            for name, quad in existing.items()
            if isinstance(quad, list) and len(quad) == 4 and all(_point_or_none(point) for point in quad)
        }

    corners = data.get("visible_corners") or {}
    required = ("front", "h_x", "h_y", "h_z", "h_xy", "h_xz", "h_yz")
    if not all(_point_or_none(corners.get(name)) for name in required):
        return {}
    front = _as_point(corners["front"])
    h_x = _as_point(corners["h_x"])
    h_y = _as_point(corners["h_y"])
    h_z = _as_point(corners["h_z"])
    h_xy = _as_point(corners["h_xy"])
    h_xz = _as_point(corners["h_xz"])
    h_yz = _as_point(corners["h_yz"])
    return {
        "face_xy": [front, h_x, h_xy, h_y],
        "face_xz": [front, h_x, h_xz, h_z],
        "face_yz": [front, h_y, h_yz, h_z],
    }


def _subdivide_quad(quad: Sequence[Point]) -> List[Quad]:
    cells: List[Quad] = []
    for row in range(3):
        v0 = row / 3.0
        v1 = (row + 1) / 3.0
        for col in range(3):
            u0 = col / 3.0
            u1 = (col + 1) / 3.0
            cells.append([
                _bilinear(quad, u0, v0),
                _bilinear(quad, u1, v0),
                _bilinear(quad, u1, v1),
                _bilinear(quad, u0, v1),
            ])
    return cells


def _bilinear(quad: Sequence[Point], u: float, v: float) -> Point:
    a, b, c, d = quad
    return (
        (1 - u) * (1 - v) * a[0] + u * (1 - v) * b[0] + u * v * c[0] + (1 - u) * v * d[0],
        (1 - u) * (1 - v) * a[1] + u * (1 - v) * b[1] + u * v * c[1] + (1 - u) * v * d[1],
    )


def _polygon_area(points: Sequence[Point]) -> float:
    area = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def _face_overlap_ratio(quads: Dict[str, Quad], image_size: Tuple[int, int]) -> float:
    width, height = image_size
    scale = min(1.0, RASTER_MAX_DIM / max(width, height))
    raster_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    stack = np.zeros((raster_size[1], raster_size[0]), dtype=np.uint8)
    for quad in quads.values():
        image = Image.new("L", raster_size, 0)
        draw = ImageDraw.Draw(image)
        draw.polygon([(x * scale, y * scale) for x, y in quad], fill=1)
        stack += np.asarray(image, dtype=np.uint8)
    union = int((stack > 0).sum())
    overlap = int((stack > 1).sum())
    return overlap / max(1, union)


def _image_size(data: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    value = data.get("imageSize")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _point_or_none(value: Any) -> Optional[Point]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def _as_point(value: Any) -> Point:
    point = _point_or_none(value)
    if point is None:
        raise ValueError(f"not a point: {value!r}")
    return point


def _distance(left: Point, right: Point) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _round_point(point: Point) -> List[float]:
    return [round(float(point[0]), 2), round(float(point[1]), 2)]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--rembg-dir", type=Path, default=DEFAULT_REMBG_DIR)
    parser.add_argument("--sam3-dir", type=Path, default=DEFAULT_SAM3_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    document = generate_geometry_split_bakeoff(
        feedback_path=args.feedback,
        rembg_dir=args.rembg_dir,
        sam3_dir=args.sam3_dir,
    )
    _write_json(args.summary, document)
    _write_text(args.report, render_report(document))
    print(f"wrote {args.summary}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
