#!/usr/bin/env python3
"""Measure per-sticker classification lift from per-face equalization.

For each (setId, side) with hand-labeled hull + ground-truth state:

  1. Load image + hull label → 3 face quads (HUMAN geometry, to isolate
     the equalize signal from proposer noise)
  2. Use joint A+B multiset face-ID (from PR #126) to map label → true face
  3. For each face:
       a. rectify → 9 sticker patches → classify  → baseline accuracy
       b. rectify → equalize → 9 patches → classify → equalized accuracy
  4. Report per-pair + aggregate lift

If equalize gives ≥3-5pp aggregate lift, particularly on OOD sets
(57/58/61/62 with varied lighting), it confirms per-face image
processing closes the OOD gap WITHOUT needing better geometry.

Per the COORDINATION.md sweep-logging convention: per-pair progress to
stderr with flush=True; log file should be redirected with
`> log 2>&1` (not `2>&1 > log`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.equalize_faces import equalize_face  # noqa: E402
from tools.extract_color_samples import (  # noqa: E402
    discover_additional_tasks,
    face_colors_from_state,
    load_corpus_tasks,
    parse_ground_truth,
)
from tools.rectify_faces import (  # noqa: E402
    DEFAULT_FACE_SIZE,
    extract_stickers_from_rectified,
    rectify_face,
)
from tools.sample_stickers_from_hull import (  # noqa: E402
    apply_orientation,
    discover_orientation,
    identify_faces_jointly,
    latest_hull_label,
    load_hull_label,
    scaled_face_quads,
)

CORPUS_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_REPORT = REPO_ROOT / "runs" / "equalize_lift_report.json"
DEFAULT_SUMMARY = REPO_ROOT / "runs" / "equalize_lift_summary.txt"
PROCESSING_MAX = 1150

EXPECTED_FACES_BY_SIDE = {"A": ("U", "R", "F"), "B": ("D", "L", "B")}


def _classify_face_chunk(
    face_img: Image.Image,
    gt_colors: List[str],
) -> Tuple[int, List[str]]:
    """Sample 9 stickers, classify, align via discover_orientation against
    gt_colors. Returns (correct_count, classified_aligned)."""
    samples = extract_stickers_from_rectified(face_img)
    rgbs = [s.rgb for row in samples for s in row]
    classified = [s.classified_color for row in samples for s in row]
    mirror, rot, _ = discover_orientation(rgbs, gt_colors)
    aligned = apply_orientation(classified, mirror, rot)
    correct = sum(1 for c, g in zip(aligned, gt_colors) if c == g)
    return correct, aligned


def evaluate_pair(
    set_id: str,
    image_a: Path,
    image_b: Path,
    gt_state: str,
    do_white_balance: bool = True,
    do_glare: bool = True,
) -> Dict:
    """Compare baseline vs equalized classification on a (set, A+B) pair.
    Uses HUMAN face quads from hull labels — no proposer in the loop."""
    # Load both images at processing resolution
    arrs = {}
    images = {}
    raw_quads_per_side = {}
    for side, image_path in (("A", image_a), ("B", image_b)):
        hull_path = latest_hull_label(set_id, side)
        if hull_path is None:
            return {"setId": set_id, "error": f"missing hull label for side {side}"}
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        natural_max = max(image.size)
        if natural_max > PROCESSING_MAX:
            scale = PROCESSING_MAX / float(natural_max)
            image = image.resize(
                (int(image.width * scale), int(image.height * scale)),
                Image.Resampling.LANCZOS,
            )
        images[side] = image
        arrs[side] = np.asarray(image)
        doc = load_hull_label(hull_path)
        raw_quads_per_side[side] = scaled_face_quads(doc, image.width, image.height)

    # Joint face-ID across both sides (uses GT state for multiset matching)
    prepared = {
        side: {"arr": arrs[side], "quads": raw_quads_per_side[side], "expected": EXPECTED_FACES_BY_SIDE[side]}
        for side in ("A", "B")
    }
    label_to_true, joint_score, joint_status = identify_faces_jointly(prepared, gt_state, inset=0.20)

    # Per face: rectify → classify (baseline) AND rectify → equalize → classify
    per_face: Dict[str, Dict] = {}
    baseline_sampled = baseline_correct = 0
    equalized_sampled = equalized_correct = 0
    for side in ("A", "B"):
        mapping = label_to_true.get(side, {})
        for label_face, true_face in mapping.items():
            quad = raw_quads_per_side[side].get(label_face)
            if quad is None or len(quad) != 4:
                continue
            gt_colors = face_colors_from_state(gt_state, true_face)
            # Baseline: rectify only
            try:
                rectified = rectify_face(images[side], quad, output_size=DEFAULT_FACE_SIZE)
            except Exception as e:
                per_face[true_face] = {"side": side, "error": f"rectify: {e}"}
                continue
            base_correct, base_aligned = _classify_face_chunk(rectified, gt_colors)
            # Equalized
            try:
                eq = equalize_face(rectified, true_face, do_white_balance=do_white_balance, do_glare=do_glare)
            except Exception as e:
                per_face[true_face] = {"side": side, "error": f"equalize: {e}"}
                continue
            eq_correct, eq_aligned = _classify_face_chunk(eq, gt_colors)
            per_face[true_face] = {
                "side": side,
                "labelFace": label_face,
                "trueFace": true_face,
                "baselineCorrect": base_correct,
                "equalizedCorrect": eq_correct,
                "lift": eq_correct - base_correct,
            }
            baseline_sampled += 9
            baseline_correct += base_correct
            equalized_sampled += 9
            equalized_correct += eq_correct

    return {
        "setId": set_id,
        "jointFaceID": {"score": joint_score, "status": joint_status},
        "perFace": per_face,
        "baselineAccuracy": round(baseline_correct / baseline_sampled, 4) if baseline_sampled else None,
        "equalizedAccuracy": round(equalized_correct / equalized_sampled, 4) if equalized_sampled else None,
        "stickersSampled": baseline_sampled,
        "lift": equalized_correct - baseline_correct,
        "liftFraction": round((equalized_correct - baseline_correct) / max(1, baseline_sampled), 4),
    }


def discover_pairs() -> List[Tuple[str, Path, Path, str]]:
    tasks = load_corpus_tasks(CORPUS_MANIFEST)
    tasks.extend(discover_additional_tasks({t.set_id for t in tasks}))
    out: List[Tuple[str, Path, Path, str]] = []
    for task in tasks:
        if not (task.image_a.exists() and task.image_b.exists()):
            continue
        try:
            gt = parse_ground_truth(task.ground_truth)
        except Exception:
            continue
        out.append((task.set_id, task.image_a, task.image_b, gt))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only-sets", nargs="*", default=None)
    ap.add_argument("--no-white-balance", action="store_true")
    ap.add_argument("--no-glare", action="store_true")
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = ap.parse_args()

    pairs = discover_pairs()
    if args.only_sets:
        wanted = set(args.only_sets)
        pairs = [p for p in pairs if p[0] in wanted]

    do_wb = not args.no_white_balance
    do_glare = not args.no_glare
    print(f"evaluating equalize lift on {len(pairs)} pairs (WB={do_wb} glare={do_glare})",
          file=sys.stderr)
    print("", file=sys.stderr)

    rows: List[Dict] = []
    for i, (set_id, image_a, image_b, gt) in enumerate(pairs, 1):
        try:
            row = evaluate_pair(set_id, image_a, image_b, gt, do_wb, do_glare)
        except Exception as e:
            row = {"setId": set_id, "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
        if "error" in row:
            print(f"  [{i:>2}/{len(pairs)}] set {set_id}: ERROR {row['error']}", file=sys.stderr, flush=True)
        else:
            b = row["baselineAccuracy"]
            e = row["equalizedAccuracy"]
            d = row["liftFraction"]
            sign = "+" if d >= 0 else ""
            print(
                f"  [{i:>2}/{len(pairs)}] set {set_id}: baseline={b}  equalized={e}  "
                f"lift={sign}{d:.4f} ({row['lift']:+d} stickers)",
                file=sys.stderr, flush=True,
            )

    # Aggregate
    valid = [r for r in rows if "error" not in r]
    baseline_total = sum(r["stickersSampled"] for r in valid)
    baseline_correct_total = sum(int(r["baselineAccuracy"] * r["stickersSampled"]) for r in valid if r.get("baselineAccuracy") is not None)
    equalized_correct_total = sum(int(r["equalizedAccuracy"] * r["stickersSampled"]) for r in valid if r.get("equalizedAccuracy") is not None)

    aggregate_baseline = baseline_correct_total / max(1, baseline_total)
    aggregate_equalized = equalized_correct_total / max(1, baseline_total)
    aggregate_lift = aggregate_equalized - aggregate_baseline

    summary_lines: List[str] = []
    summary_lines.append(f"Equalize-faces lift measurement: {len(pairs)} pairs")
    summary_lines.append(f"  white_balance={do_wb} glare={do_glare}")
    summary_lines.append("")
    summary_lines.append(f"Aggregate per-sticker accuracy ({baseline_total} stickers across {len(valid)} pairs):")
    summary_lines.append(f"  baseline:   {aggregate_baseline:.4f}")
    summary_lines.append(f"  equalized:  {aggregate_equalized:.4f}")
    summary_lines.append(f"  lift:       {aggregate_lift:+.4f} ({(equalized_correct_total - baseline_correct_total):+d} stickers)")
    summary_lines.append("")

    # OOD-set breakdown (Sets 57/58/61/62 specifically)
    ood_ids = {"57", "58", "61", "62"}
    ood = [r for r in valid if r["setId"] in ood_ids]
    if ood:
        ood_base = sum(int(r["baselineAccuracy"] * r["stickersSampled"]) for r in ood) / max(1, sum(r["stickersSampled"] for r in ood))
        ood_eq = sum(int(r["equalizedAccuracy"] * r["stickersSampled"]) for r in ood) / max(1, sum(r["stickersSampled"] for r in ood))
        summary_lines.append(f"OOD-set breakdown ({len(ood)} pairs from sets 57/58/61/62):")
        summary_lines.append(f"  baseline:   {ood_base:.4f}")
        summary_lines.append(f"  equalized:  {ood_eq:.4f}")
        summary_lines.append(f"  lift:       {ood_eq - ood_base:+.4f}")
        summary_lines.append("")

    # Biggest lift / regression
    sorted_by_lift = sorted(valid, key=lambda r: r["liftFraction"], reverse=True)
    summary_lines.append("Biggest 5 lifts:")
    for r in sorted_by_lift[:5]:
        summary_lines.append(f"  set {r['setId']}: baseline={r['baselineAccuracy']} → equalized={r['equalizedAccuracy']} ({r['lift']:+d} stickers)")
    summary_lines.append("")
    summary_lines.append("Biggest 5 regressions:")
    for r in sorted_by_lift[-5:]:
        summary_lines.append(f"  set {r['setId']}: baseline={r['baselineAccuracy']} → equalized={r['equalizedAccuracy']} ({r['lift']:+d} stickers)")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rows, indent=2))
    Path(args.summary).write_text("\n".join(summary_lines) + "\n")

    print("", file=sys.stderr)
    print("\n".join(summary_lines), file=sys.stderr)
    print(f"\nwrote {args.report}", file=sys.stderr)
    print(f"wrote {args.summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
