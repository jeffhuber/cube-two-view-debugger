#!/usr/bin/env python3
"""Join interior-bezel cell diagnostics with #175 overlay feedback.

Diagnostics-only. This builds the slot/cell-level mining table requested
after #178:

* canonical quad source: the hybrid overlay quads humans reviewed in #175
* cell quads: each face quad subdivided into a 3x3 grid by bilinear interpolation
* bezel signal: tools.interior_bezel_detection.cell_line_diagnostics
* discontinuity signal: tools.probe_overlay_discontinuity.cell_discontinuity_metrics
* human label: tests/fixtures/hard_case_visual_feedback.json failureModes

No recognizer behavior changes are made or implied.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_hybrid_pipeline import _load_processing_image, _proposer_face_quads  # noqa: E402
from tools.interior_bezel_detection import (  # noqa: E402
    DEFAULT_DISTANCE_THRESHOLD_PX,
    DEFAULT_LINE_QUALITY_THRESHOLD,
    InteriorBezelDetection,
    cell_line_diagnostics,
    detect_interior_bezel_lines,
)
from tools.overlay_feedback import DEFAULT_OUTPUT as DEFAULT_LABELS  # noqa: E402
from tools.probe_overlay_discontinuity import (  # noqa: E402
    HIGH_HALF_DELTA_THRESHOLD,
    HIGH_STD_THRESHOLD,
    cell_discontinuity_metrics,
)
from tools.rectify_faces import DEFAULT_FACE_SIZE, rectify_face  # noqa: E402


DEFAULT_MANIFEST = ROOT / "tests" / "fixtures" / "hard_case_manifest.json"
DEFAULT_OUTPUT = ROOT / "tests" / "fixtures" / "hard_case_visual_feedback_bezel_join.json"
DEFAULT_REPORT = ROOT / "tools" / "BEZEL_DISCONTINUITY_JOIN_REPORT.md"
DEFAULT_DETECTOR_VERSION = "iterative-v1"
REQUIRED_OPTIONAL_DEPENDENCIES = ("rembg", "scipy", "onnxruntime")

Point = Tuple[float, float]
Quad = List[Point]


def subdivide_face_quad(face_quad: Sequence[Point]) -> List[Dict[str, Any]]:
    """Return the 9 image-space cell quads for a face quad.

    The face quad is treated as a bilinear patch with canonical corners
    A/B/C/D corresponding to normalized coordinates:

    * A = (0, 0)
    * B = (1, 0)
    * C = (1, 1)
    * D = (0, 1)

    This matches the rectified face sampling convention: each sticker
    occupies one third of the normalized face in row-major order.
    """
    if len(face_quad) != 4:
        raise ValueError("face_quad must contain exactly 4 points")
    quad = [(float(x), float(y)) for x, y in face_quad]
    cells: List[Dict[str, Any]] = []
    for row in range(3):
        v0 = row / 3.0
        v1 = (row + 1) / 3.0
        for col in range(3):
            u0 = col / 3.0
            u1 = (col + 1) / 3.0
            cell_quad = [
                _bilinear_point(quad, u0, v0),
                _bilinear_point(quad, u1, v0),
                _bilinear_point(quad, u1, v1),
                _bilinear_point(quad, u0, v1),
            ]
            cells.append({"row": row, "col": col, "quad": cell_quad})
    return cells


def discontinuity_cell_flag(cell: Dict[str, Any]) -> bool:
    """Cell-level version of #175's discontinuity thresholds."""
    return (
        float(cell.get("internalStd", 0.0)) >= HIGH_STD_THRESHOLD
        or float(cell.get("maxHalfDelta", 0.0)) >= HIGH_HALF_DELTA_THRESHOLD
    )


def cross_tab_axis(bezel_flag: bool, discontinuity_flag: bool) -> str:
    if bezel_flag and discontinuity_flag:
        return "both_hit"
    if bezel_flag:
        return "bezel_only"
    if discontinuity_flag:
        return "discontinuity_only"
    return "both_miss"


def probe_bezel_discontinuity_join(
    labels: Dict[str, Any],
    manifest: Dict[str, Any],
    *,
    high_quality_threshold: float = DEFAULT_LINE_QUALITY_THRESHOLD,
    max_distance_px: float = DEFAULT_DISTANCE_THRESHOLD_PX,
    hull_guard: bool = False,
    fit_error_fallback: bool = False,
    detector_version: str = DEFAULT_DETECTOR_VERSION,
) -> Dict[str, Any]:
    manifest_by_set = {str(row["setId"]): row for row in manifest.get("pairs", [])}
    rows: List[Dict[str, Any]] = []
    slot_rows: List[Dict[str, Any]] = []
    detections: Dict[str, Dict[str, Any]] = {}
    pair_cache: Dict[Tuple[str, str], Tuple[Image.Image, Dict[str, Quad], Dict[str, Any], InteriorBezelDetection]] = {}

    for item in labels.get("sets", []):
        set_id = str(item.get("setId"))
        manifest_row = manifest_by_set.get(set_id)
        if not manifest_row:
            slot_rows.append({"setId": set_id, "error": "set_not_in_hard_case_manifest"})
            continue

        image_by_side = {
            "A": Path(manifest_row["imageAPath"]),
            "B": Path(manifest_row["imageBPath"]),
        }
        slots_by_side: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": []}
        for slot in item.get("slots", []):
            slots_by_side.setdefault(slot["side"], []).append(slot)

        for side, slots in slots_by_side.items():
            if not slots:
                continue
            image_path = image_by_side[side]
            cache_key = (set_id, side)
            if cache_key not in pair_cache:
                try:
                    image, image_rgb = _load_processing_image(image_path)
                    quads, debug = _proposer_face_quads(
                        image_path,
                        side,
                        hull_guard=hull_guard,
                        fit_error_fallback=fit_error_fallback,
                        processing_image=image,
                    )
                    mask = _compute_rembg_mask(image)
                    detection = detect_interior_bezel_lines(image_rgb, mask)
                    pair_cache[cache_key] = (image, quads, debug, detection)
                    detections[_pair_key(set_id, side)] = _compact_detection(detection)
                except Exception as exc:  # pragma: no cover - exercised by CLI/local deps
                    error = f"{exc.__class__.__name__}: {exc}"
                    detections[_pair_key(set_id, side)] = {"error": error}
                    for slot in slots:
                        slot_rows.append({**_label_fields(set_id, slot, {}), "error": error})
                    continue

            image, quads, debug, detection = pair_cache[cache_key]
            for slot in slots:
                slot_label = slot["slot"]
                quad = quads.get(slot_label)
                selected = debug.get("selectedPerFace", {}).get(slot_label, {})
                slot_base = _label_fields(set_id, slot, selected)
                if quad is None:
                    slot_rows.append({**slot_base, "error": "quad_not_found"})
                    continue

                try:
                    rectified = rectify_face(image, quad, output_size=DEFAULT_FACE_SIZE)
                    discontinuity = cell_discontinuity_metrics(rectified)
                    discontinuity_cells = {
                        (int(cell["row"]), int(cell["col"])): cell
                        for cell in discontinuity.get("cells", [])
                    }
                    cell_quads = subdivide_face_quad(quad)
                except Exception as exc:  # pragma: no cover - exercised by CLI/local deps
                    slot_rows.append({**slot_base, "error": f"{exc.__class__.__name__}: {exc}"})
                    continue

                slot_cell_rows = []
                for cell in cell_quads:
                    row = int(cell["row"])
                    col = int(cell["col"])
                    disc_cell = discontinuity_cells.get((row, col), {})
                    disc_flag = discontinuity_cell_flag(disc_cell)
                    bezel = cell_line_diagnostics(
                        detection,
                        cell["quad"],
                        min_line_quality=high_quality_threshold,
                        high_quality_threshold=high_quality_threshold,
                        max_distance_px=max_distance_px,
                        detector_version=detector_version,
                    )
                    bezel_flag = bool(bezel.get("crosses_high_quality_bezel"))
                    axis = cross_tab_axis(bezel_flag, disc_flag)
                    joined = {
                        **slot_base,
                        "cell": {"row": row, "col": col},
                        "cellQuad": _round_quad(cell["quad"]),
                        "humanLabelScope": "slot_label_repeated_per_cell",
                        "bezel": bezel,
                        "bezelFlag": bezel_flag,
                        "discontinuity": {
                            "flag": disc_flag,
                            "internalStd": disc_cell.get("internalStd"),
                            "maxHalfDelta": disc_cell.get("maxHalfDelta"),
                            "meanRgb": disc_cell.get("meanRgb"),
                            "thresholds": {
                                "highInternalStd": HIGH_STD_THRESHOLD,
                                "highHalfDelta": HIGH_HALF_DELTA_THRESHOLD,
                            },
                        },
                        "crossTabAxis": axis,
                    }
                    rows.append(joined)
                    slot_cell_rows.append(joined)

                slot_rows.append(_slot_summary(slot_base, discontinuity, slot_cell_rows))

    return {
        "schemaVersion": 1,
        "policy": "diagnostics_only_no_behavior_change",
        "labelsSource": labels.get("source", {}),
        "quadSource": (
            "tools.evaluate_hybrid_pipeline._proposer_face_quads, matching "
            "tools.render_hybrid_overlays.py / #175 human-reviewed overlays"
        ),
        "probeConfig": {
            "hullGuard": hull_guard,
            "fitErrorFallback": fit_error_fallback,
            "detectorVersion": detector_version,
            "bezelThresholds": {
                "lineQuality": high_quality_threshold,
                "distancePx": max_distance_px,
            },
            "discontinuityThresholds": {
                "highInternalStd": HIGH_STD_THRESHOLD,
                "highHalfDelta": HIGH_HALF_DELTA_THRESHOLD,
            },
        },
        "detections": detections,
        "summary": summarize_rows(rows, slot_rows),
        "slotRows": slot_rows,
        "rows": rows,
    }


def missing_required_optional_dependencies() -> List[str]:
    """Return optional research dependencies required to regenerate outputs."""
    return [
        name
        for name in REQUIRED_OPTIONAL_DEPENDENCIES
        if importlib.util.find_spec(name) is None
    ]


def summarize_rows(rows: Sequence[Dict[str, Any]], slot_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [row for row in rows if not row.get("error")]
    human_bad = [row for row in scored if row.get("humanBad")]
    human_good = [row for row in scored if not row.get("humanBad")]
    axis_counts = Counter(row.get("crossTabAxis") for row in scored)
    axis_by_human = {
        "humanBad": dict(sorted(Counter(row.get("crossTabAxis") for row in human_bad).items())),
        "humanGood": dict(sorted(Counter(row.get("crossTabAxis") for row in human_good).items())),
    }
    both_hit_good = [
        row for row in human_good
        if row.get("crossTabAxis") == "both_hit"
    ]
    both_miss_bad = [
        row for row in human_bad
        if row.get("crossTabAxis") == "both_miss"
    ]
    both_hit_bad = [
        row for row in human_bad
        if row.get("crossTabAxis") == "both_hit"
    ]
    return {
        "cellRowCount": len(rows),
        "scoredCellRowCount": len(scored),
        "slotRowCount": len(slot_rows),
        "scoredSlotRowCount": sum(1 for row in slot_rows if not row.get("error")),
        "humanBadCellCount": len(human_bad),
        "humanGoodCellCount": len(human_good),
        "axisCounts": dict(sorted(axis_counts.items())),
        "axisCountsByHumanLabel": axis_by_human,
        "candidateGuardBothHitCells": len([row for row in scored if row.get("crossTabAxis") == "both_hit"]),
        "candidateGuardBothHitHumanBadCells": len(both_hit_bad),
        "candidateGuardBothHitHumanGoodCells": len(both_hit_good),
        "bothMissHumanBadCells": len(both_miss_bad),
        "bezelOnlyCells": axis_counts.get("bezel_only", 0),
        "discontinuityOnlyCells": axis_counts.get("discontinuity_only", 0),
        "topBothHitHumanGoodCells": [_compact_cell(row) for row in _rank_cells(both_hit_good)[:12]],
        "topBothHitHumanBadCells": [_compact_cell(row) for row in _rank_cells(both_hit_bad)[:12]],
        "topBothMissHumanBadCells": [_compact_cell(row) for row in _rank_cells(both_miss_bad)[:12]],
        "thresholdSweep": threshold_sweep(scored),
    }


def threshold_sweep(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Recompute the bezel+discontinuity conjunction from raw per-line fields.

    The detector is comparatively expensive, so this keeps a small grid of
    candidate thresholds in the artifact for quick zero-FP inspection without
    re-running rembg/Hough.
    """
    out: List[Dict[str, Any]] = []
    for line_q in (0.3, 0.4, 0.5, 0.6, 0.8):
        for dist_px in (20.0, 30.0, 40.0, 60.0):
            both_hit = []
            bezel_hit = []
            for row in rows:
                disc = bool(row.get("discontinuity", {}).get("flag"))
                bezel = _bezel_flag_at_threshold(row, line_q, dist_px)
                if bezel:
                    bezel_hit.append(row)
                if bezel and disc:
                    both_hit.append(row)
            out.append({
                "lineQuality": line_q,
                "distancePx": dist_px,
                "bezelHitCells": len(bezel_hit),
                "bothHitCells": len(both_hit),
                "bothHitHumanBadCells": sum(1 for row in both_hit if row.get("humanBad")),
                "bothHitHumanGoodCells": sum(1 for row in both_hit if not row.get("humanBad")),
            })
    out.sort(key=lambda row: (
        row["bothHitHumanGoodCells"],
        -row["bothHitHumanBadCells"],
        row["lineQuality"],
        row["distancePx"],
    ))
    return out


def render_report(document: Dict[str, Any]) -> str:
    summary = document["summary"]
    config = document["probeConfig"]
    lines = [
        "# Bezel + Discontinuity Join Probe",
        "",
        "Diagnostics-only slot/cell mining over the same hybrid overlay quads that received human visual feedback in #175.",
        "The candidate guard shape under inspection is `crosses_high_quality_bezel AND discontinuity_flag`; neither signal alone is proposed for behavior wiring.",
        "",
        "## Summary",
        "",
        f"- Cells scored: {summary['scoredCellRowCount']} / {summary['cellRowCount']}",
        f"- Slots scored: {summary['scoredSlotRowCount']} / {summary['slotRowCount']}",
        f"- Human-bad cells: {summary['humanBadCellCount']} (slot label repeated per cell)",
        f"- Human-good cells: {summary['humanGoodCellCount']} (slot label repeated per cell)",
        f"- Bezel thresholds: line quality >= {config['bezelThresholds']['lineQuality']}, distance <= {config['bezelThresholds']['distancePx']} px",
        f"- Discontinuity thresholds: internal std >= {config['discontinuityThresholds']['highInternalStd']} or half delta >= {config['discontinuityThresholds']['highHalfDelta']}",
        "",
        "## Cross-Tab Axes",
        "",
        "| Axis | Count | Human-bad | Human-good |",
        "|---|---:|---:|---:|",
    ]
    axis_counts = summary["axisCounts"]
    bad_counts = summary["axisCountsByHumanLabel"]["humanBad"]
    good_counts = summary["axisCountsByHumanLabel"]["humanGood"]
    for axis in ("both_hit", "bezel_only", "discontinuity_only", "both_miss"):
        lines.append(
            f"| `{axis}` | {axis_counts.get(axis, 0)} | "
            f"{bad_counts.get(axis, 0)} | {good_counts.get(axis, 0)} |"
        )

    lines.extend([
        "",
        "## Guard-Candidate Readout",
        "",
        f"- Both-hit cells: {summary['candidateGuardBothHitCells']}",
        f"- Both-hit human-bad cells: {summary['candidateGuardBothHitHumanBadCells']}",
        f"- Both-hit human-good cells: {summary['candidateGuardBothHitHumanGoodCells']}",
        f"- Both-miss human-bad cells: {summary['bothMissHumanBadCells']}",
        "",
        "## Threshold Sweep",
        "",
        "| Rank | Line q | Distance px | Bezel-hit cells | Both-hit cells | Both-hit bad | Both-hit good |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for idx, row in enumerate(summary["thresholdSweep"][:12], 1):
        lines.append(
            f"| {idx} | {row['lineQuality']} | {row['distancePx']} | "
            f"{row['bezelHitCells']} | {row['bothHitCells']} | "
            f"{row['bothHitHumanBadCells']} | {row['bothHitHumanGoodCells']} |"
        )

    _append_ranked_section(
        lines,
        "Both-Hit Human-Good Cells",
        summary["topBothHitHumanGoodCells"],
        "These are the potential false-positive rows for the conjunction.",
    )
    _append_ranked_section(
        lines,
        "Both-Hit Human-Bad Cells",
        summary["topBothHitHumanBadCells"],
        "These are the rows the conjunction explains.",
    )
    _append_ranked_section(
        lines,
        "Both-Miss Human-Bad Cells",
        summary["topBothMissHumanBadCells"],
        "These are known-bad slot cells neither signal explains.",
    )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- This report is not a production guard and does not change recognizer behavior.",
        "- The human label is slot-level and repeated across that slot's 9 cells; cell counts should be read as diagnostic evidence, not independent labels.",
        "- The canonical quad source is the #175 hybrid overlay quads, not production recognizer quads. Production selected-grid transfer is a later check.",
        "- The only plausible future behavior shape remains a conservative manual-review guard where independent bezel and discontinuity signals agree, after broader zero-FP mining.",
        "",
    ])
    return "\n".join(lines)


def _compute_rembg_mask(image: Image.Image) -> np.ndarray:
    from rembg import new_session, remove  # type: ignore

    session = getattr(_compute_rembg_mask, "_session", None)
    if session is None:
        session = new_session("u2net")
        _compute_rembg_mask._session = session  # type: ignore[attr-defined]
    rgba = remove(image, session=session)
    alpha = np.array(rgba.split()[-1], dtype=np.uint8)
    return alpha > 128


def _bilinear_point(quad: Sequence[Point], u: float, v: float) -> Point:
    a, b, c, d = quad
    x = (
        (1 - u) * (1 - v) * a[0]
        + u * (1 - v) * b[0]
        + u * v * c[0]
        + (1 - u) * v * d[0]
    )
    y = (
        (1 - u) * (1 - v) * a[1]
        + u * (1 - v) * b[1]
        + u * v * c[1]
        + (1 - u) * v * d[1]
    )
    return (float(x), float(y))


def _label_fields(set_id: str, slot: Dict[str, Any], selected: Dict[str, Any]) -> Dict[str, Any]:
    modes = [mode for mode in slot.get("failureModes", []) if mode != "ok"]
    return {
        "setId": set_id,
        "image": slot.get("image"),
        "side": slot.get("side"),
        "slot": slot.get("slot"),
        "quadQuality": slot.get("quadQuality"),
        "rectifiedQuality": slot.get("rectifiedQuality"),
        "humanRectifiedSourceFace": slot.get("rectifiedSourceFace"),
        "selectedSourceFace": selected.get("sourceCenterFace"),
        "selectedSourcePosition": selected.get("sourcePosition"),
        "failureModes": modes,
        "humanBad": bool(modes),
    }


def _slot_summary(
    slot_base: Dict[str, Any],
    discontinuity: Dict[str, Any],
    cell_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    axis_counts = Counter(row.get("crossTabAxis") for row in cell_rows)
    return {
        **slot_base,
        "cellCount": len(cell_rows),
        "axisCounts": dict(sorted(axis_counts.items())),
        "bezelHitCells": sum(1 for row in cell_rows if row.get("bezelFlag")),
        "discontinuityHitCells": sum(
            1 for row in cell_rows if row.get("discontinuity", {}).get("flag")
        ),
        "bothHitCells": axis_counts.get("both_hit", 0),
        "bothMissCells": axis_counts.get("both_miss", 0),
        "slotDiscontinuityScore": discontinuity.get("score"),
        "maxInternalStd": discontinuity.get("maxInternalStd"),
        "maxHalfDelta": discontinuity.get("maxHalfDelta"),
    }


def _compact_detection(detection: InteriorBezelDetection) -> Dict[str, Any]:
    return {
        "cubeCenter": (
            [round(detection.cube_center[0], 1), round(detection.cube_center[1], 1)]
            if detection.cube_center else None
        ),
        "lineQualities": [round(float(q), 3) for q in detection.line_qualities],
        "boundaryAnglesDeg": [round(math.degrees(a), 2) for a in detection.boundary_angles],
        "signalQuality": round(float(detection.signal_quality), 3),
        "detectorVersion": DEFAULT_DETECTOR_VERSION,
        "debug": detection.debug,
    }


def _compact_cell(row: Dict[str, Any]) -> Dict[str, Any]:
    per_line = row.get("bezel", {}).get("per_line", [])
    best_line = None
    crossing_lines = [line for line in per_line if line.get("crosses_cell")]
    if crossing_lines:
        best_line = max(crossing_lines, key=lambda line: float(line.get("quality", 0.0)))
    return {
        "setId": row.get("setId"),
        "side": row.get("side"),
        "slot": row.get("slot"),
        "cell": row.get("cell"),
        "humanBad": row.get("humanBad"),
        "failureModes": row.get("failureModes", []),
        "axis": row.get("crossTabAxis"),
        "discontinuityInternalStd": row.get("discontinuity", {}).get("internalStd"),
        "discontinuityMaxHalfDelta": row.get("discontinuity", {}).get("maxHalfDelta"),
        "bezelMinDistancePx": row.get("bezel", {}).get("min_distance_from_centroid_px"),
        "bestCrossingLine": best_line,
    }


def _rank_cells(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("discontinuity", {}).get("internalStd") or 0.0),
            -float(row.get("discontinuity", {}).get("maxHalfDelta") or 0.0),
            str(row.get("setId")),
            str(row.get("side")),
            str(row.get("slot")),
            int(row.get("cell", {}).get("row", 0)),
            int(row.get("cell", {}).get("col", 0)),
        ),
    )


def _bezel_flag_at_threshold(row: Dict[str, Any], line_q: float, dist_px: float) -> bool:
    for line in row.get("bezel", {}).get("per_line", []):
        if (
            line.get("crosses_cell")
            and float(line.get("quality", 0.0)) >= line_q
            and float(line.get("distance_from_centroid_px", float("inf"))) <= dist_px
        ):
            return True
    return False


def _round_quad(quad: Sequence[Point]) -> List[List[float]]:
    return [[round(float(x), 1), round(float(y), 1)] for x, y in quad]


def _pair_key(set_id: str, side: str) -> str:
    return f"{set_id}:{side}"


def _append_ranked_section(
    lines: List[str],
    title: str,
    rows: Sequence[Dict[str, Any]],
    note: str,
) -> None:
    lines.extend([
        "",
        f"## {title}",
        "",
        note,
        "",
    ])
    if not rows:
        lines.append("_None._")
        return
    lines.extend([
        "| Set | Slot | Cell | Axis | Std | Half delta | Best crossing line | Failure modes |",
        "|---:|---|---|---|---:|---:|---|---|",
    ])
    for row in rows:
        cell = row.get("cell") or {}
        best_line = row.get("bestCrossingLine")
        if best_line:
            line_text = (
                f"q={best_line.get('quality')}, "
                f"d={best_line.get('distance_from_centroid_px')}px, "
                f"angle={best_line.get('angle_deg')}"
            )
        else:
            line_text = ""
        modes = ", ".join(f"`{mode}`" for mode in row.get("failureModes", [])) or "`ok`"
        lines.append(
            f"| {row.get('setId')} | `{row.get('side')}:{row.get('slot')}` | "
            f"{cell.get('row')},{cell.get('col')} | `{row.get('axis')}` | "
            f"{row.get('discontinuityInternalStd')} | {row.get('discontinuityMaxHalfDelta')} | "
            f"{line_text} | {modes} |"
        )


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--line-quality-threshold", type=float, default=DEFAULT_LINE_QUALITY_THRESHOLD)
    parser.add_argument("--distance-threshold-px", type=float, default=DEFAULT_DISTANCE_THRESHOLD_PX)
    parser.add_argument("--hull-guard", action="store_true")
    parser.add_argument("--fit-error-fallback", action="store_true")
    args = parser.parse_args(argv)

    missing = missing_required_optional_dependencies()
    if missing:
        deps = ", ".join(missing)
        print(
            "error: probe_bezel_discontinuity_join.py requires optional "
            f"diagnostic dependencies to regenerate outputs: {deps}.\n"
            "Install them in the repo venv, for example:\n"
            "  .venv/bin/pip install rembg scipy onnxruntime\n"
            "Refusing to write the output/report because a dependency-light "
            "run would produce an all-error or empty artifact.",
            file=sys.stderr,
        )
        return 2

    labels = _load_json(args.labels)
    manifest = _load_json(args.manifest)
    document = probe_bezel_discontinuity_join(
        labels,
        manifest,
        high_quality_threshold=args.line_quality_threshold,
        max_distance_px=args.distance_threshold_px,
        hull_guard=args.hull_guard,
        fit_error_fallback=args.fit_error_fallback,
    )
    summary = document.get("summary", {})
    if summary.get("scoredCellRowCount", 0) <= 0:
        print(
            "error: probe produced zero scored cells; refusing to overwrite "
            "the output/report with an empty mining artifact.",
            file=sys.stderr,
        )
        return 2
    _write_json(args.output, document)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(document), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
