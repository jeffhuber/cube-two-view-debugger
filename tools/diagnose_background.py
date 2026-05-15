#!/usr/bin/env python3
"""Compare cube/background diagnostic signals across local photo pairs.

This is diagnostic-only. It does not change recognizer behavior.

The default row set is the hard-case manifest's `background_sticker_noise`
pairs. Add `--include-control-set 15` to compare the hard cases against a
known clean corpus pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.inspect_cube_isolation import isolation_diagnostics_for_analysis  # noqa: E402
from tools.probe_hard_cases import load_manifest_document, normalize_path  # noqa: E402
from rubik_recognizer.image_pipeline import (  # noqa: E402
    ImageAnalysis,
    _load_image,
    _resize_for_processing,
    _rgb_to_hsv_arrays,
    analyze_image,
)
from rubik_recognizer.recognizer import FACE_ORDER, _assigned_grid_by_face  # noqa: E402


DEFAULT_HARD_CASE_MANIFEST = ROOT / "tests" / "fixtures" / "hard_case_manifest.json"
DEFAULT_CORPUS_MANIFEST = ROOT / "tests" / "fixtures" / "corpus_manifest.json"
BACKGROUND_FAILURE_CLASS = "background_sticker_noise"
SATURATION_PIXEL_THRESHOLD = 0.23
SATURATION_VALUE_FLOOR = 0.20
MIN_VIABLE_KEPT_STICKERS = 18


def diagnose_pair(
    row: Dict[str, Any],
    manifest_path: Path,
    *,
    source: str,
    padding_fraction: float = 0.18,
) -> Dict[str, Any]:
    return {
        "setId": str(row.get("setId") or row.get("id")),
        "source": source,
        "linkedIssue": row.get("linkedIssue"),
        "failureClass": row.get("failureClass"),
        "imageA": diagnose_image(
            normalize_path(row["imageAPath"], manifest_path),
            expected_anchor="U",
            padding_fraction=padding_fraction,
        ),
        "imageB": diagnose_image(
            normalize_path(row["imageBPath"], manifest_path),
            expected_anchor="D",
            padding_fraction=padding_fraction,
        ),
    }


def diagnose_image(
    image_path: Path,
    *,
    expected_anchor: str,
    padding_fraction: float = 0.18,
) -> Dict[str, Any]:
    image_bytes = image_path.read_bytes()
    analysis = analyze_image(image_bytes)
    isolation = isolation_diagnostics_for_analysis(
        analysis,
        image_path=image_path,
        anchor=expected_anchor,
        padding_fraction=padding_fraction,
    )
    processing_width = isolation["analysis"]["processingWidth"]
    processing_height = isolation["analysis"]["processingHeight"]
    return {
        "imagePath": str(image_path),
        "expectedAnchor": expected_anchor,
        "roiFraction": roi_fraction(analysis.roi, processing_width, processing_height),
        "saturatedPixelFraction": saturated_pixel_fraction(image_bytes),
        "stickerCount": len(analysis.stickers),
        "gridCount": len(analysis.grids),
        "dominantGridCenterFace": dominant_grid_center_face(analysis),
        "selectedAnchor": selected_anchor_detail(analysis, expected_anchor),
        "isolation": {
            "anchorUsed": isolation["anchorUsed"],
            "selectedGridFaces": isolation["selectedGridFaces"],
            "selectedGridIds": isolation["selectedGridIds"],
            "proposedCubeRegion": isolation["proposedCubeRegion"],
            "classificationSummary": isolation["classificationSummary"],
            "filterPreview": filter_preview(isolation["classificationSummary"], analysis, expected_anchor),
        },
    }


def roi_fraction(roi: Sequence[int], width: int, height: int) -> float:
    x0, y0, x1, y1 = roi
    area = max(0, x1 - x0) * max(0, y1 - y0)
    return round(area / float(max(1, width * height)), 4)


def saturated_pixel_fraction(image_bytes: bytes) -> float:
    image = _load_image(image_bytes)
    process_image, _ = _resize_for_processing(image, max_side=1150)
    arr = np.asarray(process_image).astype(np.uint8)
    hsv = _rgb_to_hsv_arrays(arr)
    saturated = (hsv[:, :, 1] > SATURATION_PIXEL_THRESHOLD) & (
        hsv[:, :, 2] > SATURATION_VALUE_FLOOR
    )
    return round(float(saturated.mean()), 4)


def dominant_grid_center_face(analysis: ImageAnalysis) -> Dict[str, Any]:
    counts = Counter(grid.center_face for grid in analysis.grids)
    if not counts:
        return {"face": None, "count": 0, "counts": {}}
    order = {face: index for index, face in enumerate(FACE_ORDER)}
    face, count = min(counts.items(), key=lambda item: (-item[1], order.get(item[0], 99)))
    return {
        "face": face,
        "count": count,
        "counts": ordered_face_counts(counts),
    }


def selected_anchor_detail(analysis: ImageAnalysis, expected_anchor: str) -> Dict[str, Any]:
    assignments = _assigned_grid_by_face(analysis, expected_anchor)
    grid = assignments.get(expected_anchor)
    if grid is None:
        return {
            "present": False,
            "gridId": None,
            "matchedCount": 0,
            "fitError": None,
            "assignedFaces": sorted(assignments),
        }
    return {
        "present": True,
        "gridId": grid.id,
        "centerFace": grid.center_face,
        "matchedCount": grid.matched_count,
        "fitError": round(float(grid.fit_error), 3),
        "assignedFaces": sorted(assignments),
    }


def filter_preview(
    classification_summary: Dict[str, Any],
    analysis: ImageAnalysis,
    expected_anchor: str,
) -> Dict[str, Any]:
    kept = int(classification_summary.get("kept") or 0)
    dropped = int(classification_summary.get("dropped") or 0)
    total = max(1, kept + dropped)
    anchor = selected_anchor_detail(analysis, expected_anchor)
    return {
        "wouldKeep": kept,
        "wouldDrop": dropped,
        "keptFraction": round(kept / float(total), 4),
        "viability": hull_viability(kept, dropped, anchor_present=bool(anchor["present"])),
    }


def hull_viability(kept: int, dropped: int, *, anchor_present: bool) -> str:
    if not anchor_present:
        return "anchor_missing"
    if kept < MIN_VIABLE_KEPT_STICKERS:
        return "too_few_kept"
    if dropped == 0:
        return "no_partition"
    return "candidate"


def ordered_face_counts(counts: Counter[str]) -> Dict[str, int]:
    ordered = {face: counts[face] for face in FACE_ORDER if counts.get(face)}
    for face in sorted(set(counts) - set(ordered)):
        ordered[face] = counts[face]
    return ordered


def collect_rows(
    *,
    hard_case_manifest: Path,
    corpus_manifest: Path,
    set_ids: Sequence[str],
    control_set_ids: Sequence[str],
) -> List[Tuple[Dict[str, Any], Path, str]]:
    hard_doc = load_manifest_document(hard_case_manifest)
    selected = {str(item) for item in set_ids}
    rows: List[Tuple[Dict[str, Any], Path, str]] = []
    for row in hard_doc["pairs"]:
        set_id = str(row.get("setId") or row.get("id"))
        if selected:
            if set_id in selected:
                rows.append((dict(row), hard_case_manifest, "hard-case"))
        elif row.get("failureClass") == BACKGROUND_FAILURE_CLASS:
            rows.append((dict(row), hard_case_manifest, "hard-case"))

    control_ids = {str(item) for item in control_set_ids}
    if control_ids:
        corpus_doc = load_manifest_document(corpus_manifest)
        for row in corpus_doc["pairs"]:
            set_id = str(row.get("setId") or row.get("id"))
            if set_id in control_ids:
                rows.append((dict(row), corpus_manifest, "control"))
    return rows


def write_json(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    payload = {
        "schemaVersion": 1,
        "results": list(results),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_table(results: Sequence[Dict[str, Any]]) -> None:
    print("Set Source    Side ROI%  Sat% Stickers Grids Dom  Anchor Match Kept Drop Viability")
    print("--- --------- ---- ----- ----- -------- ----- ---- ------ ----- ---- ---- --------------")
    for result in results:
        for side in ("imageA", "imageB"):
            image = result[side]
            source = str(result["source"])
            dominant = image["dominantGridCenterFace"]
            anchor = image["selectedAnchor"]
            preview = image["isolation"]["filterPreview"]
            print(
                f"{result['setId']:>3} "
                f"{source:<9.9} "
                f"{side[-1]:>4} "
                f"{image['roiFraction'] * 100:5.1f} "
                f"{image['saturatedPixelFraction'] * 100:5.1f} "
                f"{image['stickerCount']:8d} "
                f"{image['gridCount']:5d} "
                f"{_dominant_cell(dominant):<4.4} "
                f"{image['expectedAnchor']:<6.6} "
                f"{anchor['matchedCount']:5d} "
                f"{preview['wouldKeep']:4d} "
                f"{preview['wouldDrop']:4d} "
                f"{preview['viability']:<14.14}"
            )


def _dominant_cell(value: Dict[str, Any]) -> str:
    face = value.get("face")
    if not face:
        return "-"
    return f"{face}/{value.get('count', 0)}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_HARD_CASE_MANIFEST),
        help="Hard-case manifest JSON path.",
    )
    parser.add_argument(
        "--corpus-manifest",
        default=str(DEFAULT_CORPUS_MANIFEST),
        help="Labelled corpus manifest JSON path for optional control sets.",
    )
    parser.add_argument(
        "--set-id",
        action="append",
        default=[],
        help="Only diagnose one or more hard-case set ids. Defaults to all background-noise rows.",
    )
    parser.add_argument(
        "--include-control-set",
        action="append",
        default=[],
        help="Add one or more labelled corpus set ids as controls, e.g. --include-control-set 15.",
    )
    parser.add_argument(
        "--padding-fraction",
        type=float,
        default=0.18,
        help="Selected-grid hull padding fraction to preview.",
    )
    parser.add_argument("--json-output", type=Path, help="Optional path for full JSON output.")
    parser.add_argument("--quiet", action="store_true", help="Do not print the readable table.")
    args = parser.parse_args(argv)

    hard_manifest = Path(args.manifest).expanduser()
    if not hard_manifest.is_absolute():
        hard_manifest = (Path.cwd() / hard_manifest).resolve()
    corpus_manifest = Path(args.corpus_manifest).expanduser()
    if not corpus_manifest.is_absolute():
        corpus_manifest = (Path.cwd() / corpus_manifest).resolve()

    rows = collect_rows(
        hard_case_manifest=hard_manifest,
        corpus_manifest=corpus_manifest,
        set_ids=args.set_id,
        control_set_ids=args.include_control_set,
    )
    results = [
        diagnose_pair(
            row,
            manifest,
            source=source,
            padding_fraction=args.padding_fraction,
        )
        for row, manifest, source in rows
    ]
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.json_output, results)
    if not args.quiet:
        print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
