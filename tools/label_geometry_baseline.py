#!/usr/bin/env python3
"""Summarize the latest human geometry labels for recognizer diagnostics.

This tool is intentionally label-aware and diagnostic-only. It discovers the
latest A/B geometry label JSON per set, runs the existing geometry evaluator,
and writes a compact baseline that can be compared before/after recognizer
changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_geometry_labels import (  # noqa: E402
    DEFAULT_IMAGE_ROOTS,
    DEFAULT_LABELS_DIR,
    DEFAULT_MANIFESTS,
    evaluate_label_file,
    load_label_document,
)


def normalize_set_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("set "):
        text = text[4:]
    elif lowered.startswith("set-"):
        text = text[4:]
    return text.strip()


def latest_label_paths(labels_dir: Path, selected_set_ids: Optional[Iterable[str]] = None) -> List[Path]:
    selected = {normalize_set_id(item) for item in selected_set_ids or []}
    selected.discard("")
    latest: Dict[Tuple[str, str], Tuple[str, Path]] = {}
    for label_path in sorted(labels_dir.glob("*geometry-label.json")):
        try:
            document = load_label_document(label_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        set_id = normalize_set_id(document.get("setId"))
        side = str(document.get("imageSide") or "").upper()
        if not set_id or side not in {"A", "B"}:
            continue
        if selected and set_id not in selected:
            continue
        saved_at = str(document.get("savedAt") or document.get("createdAt") or "")
        key = (set_id, side)
        current = latest.get(key)
        if current is None or (saved_at, str(label_path)) > (current[0], str(current[1])):
            latest[key] = (saved_at, label_path)
    return [path for _, path in sorted(latest.values(), key=lambda item: label_sort_key(item[1]))]


def label_sort_key(path: Path) -> Tuple[int, str, str]:
    try:
        document = load_label_document(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return (10**9, "", str(path))
    set_id = normalize_set_id(document.get("setId"))
    try:
        set_number = int(set_id)
    except ValueError:
        set_number = 10**9
    return (set_number, str(document.get("imageSide") or ""), str(path))


def summarize_metric(result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    stickers = metrics.get("stickers") if isinstance(metrics.get("stickers"), dict) else {}
    selected = metrics.get("selectedGridCells") if isinstance(metrics.get("selectedGridCells"), dict) else {}
    selected_overall = selected.get("overall") if isinstance(selected.get("overall"), dict) else {}
    top = metrics.get("topVisibleTripleGridCells") if isinstance(metrics.get("topVisibleTripleGridCells"), dict) else {}
    top_overall = top.get("overall") if isinstance(top.get("overall"), dict) else {}
    proposed = metrics.get("proposedCubeRegion") if isinstance(metrics.get("proposedCubeRegion"), dict) else {}
    labels = result.get("labels") if isinstance(result.get("labels"), dict) else {}
    face_quads = labels.get("faceQuads") if isinstance(labels.get("faceQuads"), dict) else {}
    sha = result.get("imageSha256") if isinstance(result.get("imageSha256"), dict) else {}
    return {
        "setId": result.get("setId"),
        "imageSide": result.get("imageSide"),
        "labelPath": result.get("labelPath"),
        "imagePath": result.get("imagePath"),
        "imageSha256Matches": sha.get("matches"),
        "faceLabels": sorted(face_quads),
        "cubeHullPointCount": len(labels.get("cubeHull") or []),
        "stickersDetected": stickers.get("detected", 0),
        "stickersOutsideHull": stickers.get("outsideLabeledCubeHull", 0),
        "selectedGridCells": selected_overall.get("cells", 0),
        "selectedGridCellsOutsideHull": selected_overall.get("outsideLabeledCubeHull", 0),
        "selectedGridSampleCells": selected_overall.get("gridSampleCells", 0),
        "selectedGridSampleCellsOutsideHull": selected_overall.get("gridSampleCellsOutsideLabeledCubeHull", 0),
        "topVisibleTripleCellsOutsideHull": top_overall.get("outsideLabeledCubeHull", 0),
        "hullIou": proposed.get("iouWithLabelCubeHull"),
        "paddedHullIou": proposed.get("paddedIouWithLabelCubeHull"),
        "overlayPath": (result.get("artifacts") or {}).get("overlayPath"),
    }


def summarize_sets(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_set: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_set.setdefault(normalize_set_id(row.get("setId")), []).append(row)

    summaries = []
    for set_id, set_rows in sorted(by_set.items(), key=lambda item: numeric_sort(item[0])):
        sides = {str(row.get("imageSide")) for row in set_rows}
        summaries.append(
            {
                "setId": f"Set {set_id}",
                "sides": sorted(sides),
                "missingSides": [side for side in ("A", "B") if side not in sides],
                "stickersOutsideHull": sum(int(row.get("stickersOutsideHull") or 0) for row in set_rows),
                "selectedGridCellsOutsideHull": sum(int(row.get("selectedGridCellsOutsideHull") or 0) for row in set_rows),
                "topVisibleTripleCellsOutsideHull": sum(int(row.get("topVisibleTripleCellsOutsideHull") or 0) for row in set_rows),
                "minHullIou": min(
                    (float(row["hullIou"]) for row in set_rows if row.get("hullIou") is not None),
                    default=None,
                ),
                "imageSha256Matches": all(row.get("imageSha256Matches") is True for row in set_rows),
            }
        )
    return summaries


def numeric_sort(value: str) -> Tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10**9, value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_table(rows: Sequence[Dict[str, Any]]) -> None:
    print("Set Side Stk Out Grid GridOut SampleOut TopOut HullIoU Label")
    print("--- ---- --- --- ---- ------- --------- ------ ------- ----------------")
    for row in rows:
        set_id = normalize_set_id(row.get("setId"))
        label = Path(str(row.get("labelPath") or "")).name
        hull_iou = row.get("hullIou")
        hull = f"{float(hull_iou):.3f}" if hull_iou is not None else "-"
        print(
            f"{set_id:>3} "
            f"{str(row.get('imageSide') or '-'):>4} "
            f"{int(row.get('stickersDetected') or 0):>3} "
            f"{int(row.get('stickersOutsideHull') or 0):>3} "
            f"{int(row.get('selectedGridCells') or 0):>4} "
            f"{int(row.get('selectedGridCellsOutsideHull') or 0):>7} "
            f"{int(row.get('selectedGridSampleCellsOutsideHull') or 0):>9} "
            f"{int(row.get('topVisibleTripleCellsOutsideHull') or 0):>6} "
            f"{hull:>7} "
            f"{label}"
        )


def build_baseline(
    label_paths: Sequence[Path],
    *,
    manifests: Sequence[Path],
    image_roots: Sequence[Path],
    overlay_dir: Optional[Path],
) -> Dict[str, Any]:
    results = [
        evaluate_label_file(
            path,
            manifests=manifests,
            image_roots=image_roots,
            overlay_dir=overlay_dir,
        )
        for path in label_paths
    ]
    rows = [summarize_metric(result) for result in results]
    return {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "labelPaths": [str(path) for path in label_paths],
        "rows": rows,
        "sets": summarize_sets(rows),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", default=str(DEFAULT_LABELS_DIR), help="Directory containing geometry-label JSON files.")
    parser.add_argument("--set-id", action="append", help="Only evaluate the latest labels for one or more set ids.")
    parser.add_argument("--manifest", action="append", type=Path, help="Manifest path used to resolve label images.")
    parser.add_argument("--image-root", action="append", type=Path, help="Additional directory for resolving label images.")
    parser.add_argument("--overlay-dir", type=Path, help="Optional directory for evaluator overlays.")
    parser.add_argument("--json-output", type=Path, help="Optional output path for full JSON baseline.")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir).expanduser()
    label_paths = latest_label_paths(labels_dir, args.set_id)
    if not label_paths:
        selected = ", ".join(args.set_id or []) or "all sets"
        print(f"No latest geometry labels found for {selected}.", file=sys.stderr)
        return 1

    manifests = tuple(args.manifest) if args.manifest else DEFAULT_MANIFESTS
    image_roots = tuple(args.image_root) if args.image_root else DEFAULT_IMAGE_ROOTS
    payload = build_baseline(
        label_paths,
        manifests=manifests,
        image_roots=image_roots,
        overlay_dir=args.overlay_dir,
    )
    print_table(payload["rows"])
    if args.json_output:
        write_json(args.json_output.expanduser(), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
