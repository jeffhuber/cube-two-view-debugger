#!/usr/bin/env python3
"""Diagnose the 50ish per-sticker errors in the canonical-classifier
rectified-from-human-quads baseline (aggregate ~97.28% on the 34-pair
corpus from PR #140's measurement).

Same pipeline as `tools/evaluate_equalize_lift.py` (HUMAN face quads
→ joint A+B face-ID → rectify → 9 sticker samples → canonical
classifier), but instead of aggregating, dump every wrong sticker
with its position, RGB, predicted vs GT color. Cluster by
confusion pair and by set. Render the worst pairs as overlay
images so we can SEE the failures.

Output:
  * runs/per_sticker_errors_report.json  — every wrong sticker
  * runs/per_sticker_errors_summary.txt  — clusters and breakdowns
  * runs/per_sticker_errors_overlays/    — PNG per face with errors

The goal is to answer: of the ~50 wrong stickers on the canonical
baseline, what are they? Which confusion pairs dominate? Which sets
contribute most? Are they OOD-lighting-clustered or spread evenly?

Per the COORDINATION.md sweep-logging convention: per-pair progress
to stderr with flush=True; log file should be redirected with
`> log 2>&1` (not `2>&1 > log`).

Usage:
  .venv/bin/python tools/diagnose_per_sticker_errors.py
  .venv/bin/python tools/diagnose_per_sticker_errors.py --only-sets 57 58 61 62
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import CANONICAL_RGB, FACE_TO_COLOR  # noqa: E402
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
DEFAULT_REPORT = REPO_ROOT / "runs" / "per_sticker_errors_report.json"
DEFAULT_SUMMARY = REPO_ROOT / "runs" / "per_sticker_errors_summary.txt"
DEFAULT_OVERLAYS = REPO_ROOT / "runs" / "per_sticker_errors_overlays"
PROCESSING_MAX = 1150

EXPECTED_FACES_BY_SIDE = {"A": ("U", "R", "F"), "B": ("D", "L", "B")}
OOD_SETS = {"57", "58", "61", "62"}


def _classify_face_chunk(face_img: Image.Image, gt_colors: List[str]):
    """Sample 9 stickers, classify, align to GT via orientation discovery.
    Returns (samples, aligned_classified, mirror, rot, correct_count).
    `samples` is the raw row-major 3x3 of StickerSample objects (NOT
    rotated); `aligned_classified` is the 9-element list aligned to
    `gt_colors` after orientation discovery."""
    samples = extract_stickers_from_rectified(face_img)
    rgbs = [s.rgb for row in samples for s in row]
    classified = [s.classified_color for row in samples for s in row]
    mirror, rot, _ = discover_orientation(rgbs, gt_colors)
    aligned_classified = apply_orientation(classified, mirror, rot)
    aligned_rgbs = apply_orientation(rgbs, mirror, rot)
    correct = sum(1 for c, g in zip(aligned_classified, gt_colors) if c == g)
    return samples, aligned_classified, aligned_rgbs, mirror, rot, correct


def _render_overlay(face_img: Image.Image, samples, aligned_classified,
                    aligned_rgbs, gt_colors, mirror: bool, rot: int,
                    out_path: Path, title: str) -> None:
    """Render a rectified face with each wrong sticker circled in red and
    labeled with `pred → gt`."""
    overlay = face_img.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = face_img.size
    cell = w / 3.0
    # We need to map the ALIGNED 0-8 index back to the raw row-major
    # position. That's the inverse of apply_orientation. Easier: render
    # using the aligned 3x3 grid (treat the aligned face as if it's
    # already in GT-orientation; the visual won't match raw sample
    # positions but the labels will be informative). Actually, for
    # inspection we want to show on the ACTUAL face image with the wrong
    # samples in their ACTUAL positions. Walk samples row-major.
    for r in range(3):
        for c in range(3):
            # Index in the raw row-major flatten
            raw_idx = r * 3 + c
            # Find where this raw_idx ended up in the aligned vector
            # so we know which gt position it corresponds to.
            # apply_orientation: aligned[i] = raw[mapping(i)].
            # We want the inverse: given raw_idx, what aligned index?
            # Easiest: invert by brute search since it's only 9 elements.
            aligned_idx = _aligned_index_for_raw(raw_idx, mirror, rot)
            pred = aligned_classified[aligned_idx]
            gt = gt_colors[aligned_idx]
            cx, cy = samples[r][c].center_xy
            if pred != gt:
                radius = int(cell * 0.22)
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    outline=(255, 0, 0), width=4,
                )
                label = f"{pred}->{gt}"
                draw.text((cx - radius, cy + radius + 2), label, fill=(255, 0, 0))
    # Title strip
    title_h = 28
    canvas = Image.new("RGB", (w, h + title_h), (250, 250, 250))
    canvas.paste(overlay, (0, title_h))
    draw2 = ImageDraw.Draw(canvas)
    draw2.text((6, 6), title, fill=(20, 20, 20))
    canvas.save(out_path, quality=88)


def _aligned_index_for_raw(raw_idx: int, mirror: bool, rot: int) -> int:
    """Invert apply_orientation: given a raw row-major index 0-8, return
    the aligned index it ends up at. Brute-force via a permutation."""
    raw_marker = [i for i in range(9)]
    aligned_marker = apply_orientation(raw_marker, mirror, rot)
    # aligned_marker[aligned_idx] == raw_idx → return aligned_idx
    return aligned_marker.index(raw_idx)


def _color_to_face_letter(color: str) -> Optional[str]:
    """Reverse FACE_TO_COLOR. Returns the face letter for a canonical
    color name. Defensive — returns None if unknown."""
    for face, c in FACE_TO_COLOR.items():
        if c == color:
            return face
    return None


def diagnose_pair(set_id: str, image_a: Path, image_b: Path, gt_state: str,
                  overlay_dir: Optional[Path]) -> Dict:
    """Run the baseline pipeline on one pair, dump every wrong sticker."""
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

    prepared = {
        side: {"arr": arrs[side], "quads": raw_quads_per_side[side],
               "expected": EXPECTED_FACES_BY_SIDE[side]}
        for side in ("A", "B")
    }
    label_to_true, _, _ = identify_faces_jointly(prepared, gt_state, inset=0.20)

    errors: List[Dict] = []
    stickers_sampled = 0
    stickers_correct = 0
    per_face_summary: List[Dict] = []
    for side in ("A", "B"):
        mapping = label_to_true.get(side, {})
        for label_face, true_face in mapping.items():
            quad = raw_quads_per_side[side].get(label_face)
            if quad is None or len(quad) != 4:
                continue
            gt_colors = face_colors_from_state(gt_state, true_face)
            try:
                rectified = rectify_face(images[side], quad,
                                         output_size=DEFAULT_FACE_SIZE)
            except Exception as e:
                per_face_summary.append({"side": side, "face": true_face,
                                         "error": f"rectify: {e}"})
                continue
            samples, aligned, aligned_rgbs, mirror, rot, correct = \
                _classify_face_chunk(rectified, gt_colors)
            stickers_sampled += 9
            stickers_correct += correct
            per_face_summary.append({
                "side": side, "face": true_face,
                "correct": correct, "ofTotal": 9,
            })

            face_has_errors = False
            for aligned_idx in range(9):
                pred = aligned[aligned_idx]
                gt = gt_colors[aligned_idx]
                if pred == gt:
                    continue
                face_has_errors = True
                rgb = aligned_rgbs[aligned_idx]
                errors.append({
                    "setId": set_id,
                    "side": side,
                    "trueFace": true_face,
                    "alignedIndex": aligned_idx,
                    "alignedRow": aligned_idx // 3,
                    "alignedCol": aligned_idx % 3,
                    "predicted": pred,
                    "groundTruth": gt,
                    "confusion": f"{pred}->{gt}",
                    "rgb": list(rgb),
                    "isOOD": set_id in OOD_SETS,
                })

            if face_has_errors and overlay_dir is not None:
                overlay_dir.mkdir(parents=True, exist_ok=True)
                out_path = overlay_dir / f"set{set_id}_{side}_{true_face}.png"
                title = f"set {set_id} side {side} face {true_face}  ({correct}/9)"
                _render_overlay(rectified, samples, aligned, aligned_rgbs,
                                gt_colors, mirror, rot, out_path, title)

    return {
        "setId": set_id,
        "isOOD": set_id in OOD_SETS,
        "stickersSampled": stickers_sampled,
        "stickersCorrect": stickers_correct,
        "accuracy": round(stickers_correct / stickers_sampled, 4)
        if stickers_sampled else None,
        "errorCount": len(errors),
        "perFace": per_face_summary,
        "errors": errors,
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


def _format_clusters(errors: List[Dict]) -> List[str]:
    lines: List[str] = []
    confusions = Counter(e["confusion"] for e in errors)
    lines.append(f"Confusion-pair frequency ({len(errors)} total errors):")
    for conf, n in confusions.most_common():
        lines.append(f"  {conf:>20}  {n:>3}  ({n*100/max(1,len(errors)):.1f}%)")
    lines.append("")

    by_face = Counter(e["trueFace"] for e in errors)
    lines.append("Errors by face:")
    for face in ("U", "R", "F", "D", "L", "B"):
        if by_face[face]:
            lines.append(f"  {face}  {by_face[face]}")
    lines.append("")

    by_set = Counter(e["setId"] for e in errors)
    lines.append("Errors by set (top 10):")
    for set_id, n in by_set.most_common(10):
        flag = "  [OOD]" if set_id in OOD_SETS else ""
        lines.append(f"  set {set_id}: {n} errors{flag}")
    lines.append("")

    ood_count = sum(1 for e in errors if e["isOOD"])
    non_ood = len(errors) - ood_count
    lines.append("OOD vs non-OOD split:")
    lines.append(f"  OOD     (Sets 57/58/61/62): {ood_count} errors")
    lines.append(f"  non-OOD (28 other pairs):   {non_ood} errors")
    lines.append("")

    # Position heatmap: which of the 9 positions go wrong most?
    pos = Counter((e["alignedRow"], e["alignedCol"]) for e in errors)
    lines.append("Error position heatmap (aligned to GT orientation):")
    lines.append("  (row=0 is top, col=0 is left; center is (1,1))")
    for r in range(3):
        cells = []
        for c in range(3):
            n = pos[(r, c)]
            mark = "·" if n == 0 else str(n)
            cells.append(f"{mark:>3}")
        lines.append("  " + " ".join(cells))
    lines.append("")

    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only-sets", nargs="*", default=None)
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--overlays", default=str(DEFAULT_OVERLAYS))
    ap.add_argument("--no-overlays", action="store_true")
    args = ap.parse_args()

    pairs = discover_pairs()
    if args.only_sets:
        wanted = set(args.only_sets)
        pairs = [p for p in pairs if p[0] in wanted]

    overlay_dir = None if args.no_overlays else Path(args.overlays)

    print(f"diagnosing per-sticker errors on {len(pairs)} pairs",
          file=sys.stderr)
    print("", file=sys.stderr)

    rows: List[Dict] = []
    all_errors: List[Dict] = []
    for i, (set_id, image_a, image_b, gt) in enumerate(pairs, 1):
        try:
            row = diagnose_pair(set_id, image_a, image_b, gt, overlay_dir)
        except Exception as e:
            row = {"setId": set_id, "error": f"{type(e).__name__}: {e}"}
        rows.append(row)
        if "error" in row:
            print(f"  [{i:>2}/{len(pairs)}] set {set_id}: ERROR {row['error']}",
                  file=sys.stderr, flush=True)
        else:
            all_errors.extend(row["errors"])
            flag = " [OOD]" if row.get("isOOD") else ""
            print(
                f"  [{i:>2}/{len(pairs)}] set {set_id}: "
                f"{row['stickersCorrect']}/{row['stickersSampled']} "
                f"(acc={row['accuracy']}, {row['errorCount']} errors){flag}",
                file=sys.stderr, flush=True,
            )

    valid = [r for r in rows if "error" not in r]
    total_sampled = sum(r["stickersSampled"] for r in valid)
    total_correct = sum(r["stickersCorrect"] for r in valid)
    aggregate = total_correct / max(1, total_sampled)

    summary_lines: List[str] = []
    summary_lines.append(
        f"Per-sticker error diagnostic: {len(pairs)} pairs, "
        f"{total_sampled} stickers"
    )
    summary_lines.append("")
    summary_lines.append("Aggregate per-sticker accuracy (baseline canonical "
                         "classifier, HUMAN face quads):")
    summary_lines.append(f"  correct:  {total_correct}/{total_sampled}")
    summary_lines.append(f"  accuracy: {aggregate:.4f}")
    summary_lines.append(f"  errors:   {len(all_errors)}")
    summary_lines.append("")
    summary_lines.extend(_format_clusters(all_errors))

    # Per-pair error counts (worst 10)
    sorted_pairs = sorted(valid, key=lambda r: r["errorCount"], reverse=True)
    summary_lines.append("Worst 10 pairs by error count:")
    for r in sorted_pairs[:10]:
        flag = "  [OOD]" if r.get("isOOD") else ""
        summary_lines.append(
            f"  set {r['setId']}: {r['errorCount']} errors "
            f"(acc={r['accuracy']}){flag}"
        )

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rows, indent=2))
    Path(args.summary).write_text("\n".join(summary_lines) + "\n")

    print("", file=sys.stderr)
    print("\n".join(summary_lines), file=sys.stderr)
    print(f"\nwrote {args.report}", file=sys.stderr)
    print(f"wrote {args.summary}", file=sys.stderr)
    if overlay_dir is not None and any(r.get("errorCount", 0) > 0
                                       for r in valid):
        print(f"wrote overlays to {overlay_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
