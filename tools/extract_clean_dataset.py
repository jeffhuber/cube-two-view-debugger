#!/usr/bin/env python3
"""Extract clean per-sticker (RGB → ground-truth color) training samples
across every hull-labeled corpus + discovered set, using the geometric
pipeline from `sample_stickers_from_hull.py` (4-point homography +
multiset face-ID + adaptive palette).

Output: JSONL at runs/color_samples_geom.jsonl, one line per sticker:
    {"setId": "15", "side": "A", "face": "U", "row": 0, "col": 0,
     "rgb": [233, 232, 99], "gtColor": "yellow",
     "calibratedClassifier": "yellow", "defaultClassifier": "yellow",
     "labelName": "U", "yawDetected": false}

`gtColor` is the ground-truth color from the corrected state at the
discovered true face's (row, col) position — no classifier in the loop
for the labels. The `calibratedClassifier` column is what `classify_rgb`
returns with the per-image adaptive palette; useful as a baseline
comparator. The `defaultClassifier` column is the un-calibrated baseline.

Drop-in replacement for `runs/color_samples_v0.jsonl` (which came from
the earlier greedy-multiset-Hungarian path with classifier-derived
labels). Feed into `train_color_classifier.py` for the definitive
bake-off."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import build_adaptive_palette, classify_rgb  # noqa: E402
from tools.extract_color_samples import (  # noqa: E402
    CORPUS_MANIFEST,
    PairTask,
    discover_additional_tasks,
    load_corpus_tasks,
    parse_ground_truth,
)
from tools.sample_stickers_from_hull import (  # noqa: E402
    collect_anchor_rgbs,
    finish_side,
    identify_faces_from_multisets,
    latest_hull_label,
    load_hull_label,
    process_side,
)

DEFAULT_OUTPUT = REPO_ROOT / "runs" / "color_samples_geom.jsonl"


def process_pair(task: PairTask, inset: float) -> Tuple[List[Dict], Dict]:
    """Run the geometric pipeline on one labeled pair. Returns (samples,
    metadata). Skips silently if hull labels are missing on either side."""
    try:
        gt_state = parse_ground_truth(task.ground_truth)
    except Exception as e:
        print(f"  set {task.set_id}: GT parse failed ({e})", file=sys.stderr)
        return [], {"setId": task.set_id, "skipped": "gt_parse_error"}

    # Step 1: prepare both sides
    prepared_sides: Dict[str, Dict] = {}
    for side, image_path in (("A", task.image_a), ("B", task.image_b)):
        hull_path = latest_hull_label(task.set_id, side)
        if hull_path is None:
            continue
        try:
            hull_doc = load_hull_label(hull_path)
            prepared_sides[side] = process_side(image_path, hull_doc, gt_state, side, inset=inset)
        except FileNotFoundError as e:
            print(f"  set {task.set_id} side {side}: missing image ({e})", file=sys.stderr)

    if not prepared_sides:
        return [], {"setId": task.set_id, "skipped": "no_hull_labels"}

    # Step 2: face identification via multisets
    label_maps: Dict[str, Dict[str, str]] = {}
    yaw_detected = False
    for side, prepared in prepared_sides.items():
        label_to_true, _ = identify_faces_from_multisets(prepared, gt_state, inset)
        label_maps[side] = label_to_true
        for label, true in label_to_true.items():
            # Anchor-letter mismatch indicates a yaw or flip-swap.
            anchor = "U" if side == "A" else "D"
            if label != anchor and label != true:
                yaw_detected = True

    # Step 3: combined adaptive palette
    combined_anchors: Dict[str, List[Tuple[int, int, int]]] = {}
    combined_samples: List[Tuple[int, int, int]] = []
    for side, prepared in prepared_sides.items():
        anchors, all_rgbs = collect_anchor_rgbs(prepared, inset, label_maps[side])
        for color, rgbs in anchors.items():
            combined_anchors.setdefault(color, []).extend(rgbs)
        combined_samples.extend(all_rgbs)
    palette = build_adaptive_palette(combined_samples, anchors=combined_anchors)

    # Step 4: per-side sampling with calibrated palette + emit samples
    samples: List[Dict] = []
    for side, prepared in prepared_sides.items():
        result = finish_side(prepared, gt_state, inset, palette, label_maps[side])
        for fs in result["face_samples"]:
            for s in fs["stickers"]:
                rgb = tuple(s["rgb"])
                samples.append({
                    "setId": task.set_id,
                    "side": side,
                    "labelName": fs["labelName"],
                    "face": s["face"],
                    "row": s["row"],
                    "col": s["col"],
                    "isCenter": (s["row"] == 1 and s["col"] == 1),
                    "rgb": list(rgb),
                    "calibratedClassifier": s["classifier"],
                    "defaultClassifier": classify_rgb(rgb).color,
                    "gtColor": s["gtColor"],
                })
    return samples, {
        "setId": task.set_id,
        "source": task.source,
        "sidesProcessed": list(prepared_sides.keys()),
        "yawDetected": yaw_detected,
        "sampleCount": len(samples),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inset", type=float, default=0.167)
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--corpus-only", action="store_true",
                    help="Only the 15 corpus_manifest pairs (skip discovered).")
    args = ap.parse_args()

    tasks = load_corpus_tasks(CORPUS_MANIFEST)
    if not args.corpus_only:
        tasks.extend(discover_additional_tasks({t.set_id for t in tasks}))

    print(f"processing {len(tasks)} tasks", file=sys.stderr)

    all_samples: List[Dict] = []
    metas: List[Dict] = []
    for task in tasks:
        print(f"set {task.set_id} ({task.source}):", file=sys.stderr)
        samples, meta = process_pair(task, args.inset)
        all_samples.extend(samples)
        metas.append(meta)
        print(f"  → {meta.get('sampleCount', 0)} samples"
              + (f"   ⚠ yaw/flip detected" if meta.get("yawDetected") else "")
              + (f"   SKIPPED: {meta['skipped']}" if "skipped" in meta else ""),
              file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")

    # Summary
    by_gt = Counter(s["gtColor"] for s in all_samples)
    cal_correct = sum(1 for s in all_samples if s["calibratedClassifier"] == s["gtColor"])
    def_correct = sum(1 for s in all_samples if s["defaultClassifier"] == s["gtColor"])
    n_yaw = sum(1 for m in metas if m.get("yawDetected"))
    n_skipped = sum(1 for m in metas if "skipped" in m)

    print("", file=sys.stderr)
    print(f"wrote {len(all_samples)} samples to {output_path}", file=sys.stderr)
    print(f"  per-color: {dict(by_gt)}", file=sys.stderr)
    print(f"  {n_yaw} sets had yaw/flip detected; {n_skipped} skipped", file=sys.stderr)
    print(
        f"  baseline default classifier:    {def_correct}/{len(all_samples)} = {def_correct / max(1, len(all_samples)):.1%}",
        file=sys.stderr,
    )
    print(
        f"  per-image calibrated classifier: {cal_correct}/{len(all_samples)} = {cal_correct / max(1, len(all_samples)):.1%}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
