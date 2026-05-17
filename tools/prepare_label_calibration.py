#!/usr/bin/env python3
"""Prepare a single-set hand-labeling calibration packet.

For one chosen setId from the corpus, this produces:
  /tmp/cal-set-<id>-A.png  — image A with every detected sticker numbered
  /tmp/cal-set-<id>-B.png  — image B with every detected sticker numbered
  /tmp/cal-set-<id>-labels.txt — table with one row per numbered sticker:
      number, side, face, multiset_label, classifier_label, RGB
      (plus a blank column for the user's hand label and a fill-in-here block)

After hand-labeling, run ``score_label_calibration.py`` on the filled file
to compute multiset-vs-hand and classifier-vs-hand accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import FACE_TO_COLOR, classify_rgb  # noqa: E402
from rubik_recognizer.image_pipeline import analyze_image  # noqa: E402
from rubik_recognizer.validation import FACE_ORDER  # noqa: E402
from tools.extract_color_samples import (  # noqa: E402
    EXPECTED_FACES_BY_SIDE,
    assign_multiset,
    discover_additional_tasks,
    face_colors_from_state,
    latest_hull_label,
    load_corpus_tasks,
    load_hull_label,
    parse_ground_truth,
    scaled_face_quads,
    stickers_by_face_via_hull,
)

CORPUS_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
OUT_DIR = Path("/tmp")


def render_annotated(
    image_path: Path,
    centers_with_meta: List[Tuple[int, float, float, str]],
    out_path: Path,
) -> Tuple[int, int]:
    """Render the EXIF-corrected image with each sticker numbered.
    centers_with_meta items: (number, x, y, face). Returns (proc_w, proc_h) in
    the same resolution analyze_image used internally so the numbering aligns."""
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    # Match analyze_image's max-side=1150 resize so coordinates align.
    side = max(image.size)
    if side > 1150:
        scale = 1150 / float(side)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    face_color = {
        "U": (0, 0, 0),
        "R": (220, 35, 35),
        "F": (40, 160, 80),
        "D": (200, 180, 0),
        "L": (245, 130, 35),
        "B": (60, 90, 220),
    }
    for num, x, y, face in centers_with_meta:
        col = face_color.get(face, (255, 255, 255))
        r = 16
        # white halo + colored ring
        draw.ellipse((x - r - 2, y - r - 2, x + r + 2, y + r + 2), outline=(255, 255, 255), width=4)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=col, width=3)
        text = str(num)
        # Center text
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x - tw / 2 - bbox[0]
        ty = y - th / 2 - bbox[1]
        # White outline behind text
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                if dx == 0 and dy == 0:
                    continue
                draw.text((tx + dx, ty + dy), text, font=font, fill=(255, 255, 255))
        draw.text((tx, ty), text, font=font, fill=col)
    canvas.save(out_path, "PNG", optimize=True)
    return canvas.size


def prepare(set_id: str) -> int:
    tasks = load_corpus_tasks(CORPUS_MANIFEST)
    task = next((t for t in tasks if t.set_id == set_id), None)
    if task is None:
        extra = discover_additional_tasks({t.set_id for t in tasks})
        task = next((t for t in extra if t.set_id == set_id), None)
    if task is None:
        print(f"set {set_id} not found in corpus manifest or Downloads", file=sys.stderr)
        return 2
    gt_state = parse_ground_truth(task.ground_truth)

    counter = 0
    table_rows: List[Dict] = []
    for side, image_path in (("A", task.image_a), ("B", task.image_b)):
        image_bytes = image_path.read_bytes()
        analysis = analyze_image(image_bytes)

        hull_path = latest_hull_label(set_id, side)
        if hull_path is None:
            print(f"  set {set_id} side {side}: no hull label — skipping (calibration requires hull labels)", file=sys.stderr)
            continue
        document = load_hull_label(hull_path)
        proc_max = 1150
        natural_max = max(analysis.width, analysis.height)
        if natural_max <= proc_max:
            proc_w, proc_h = analysis.width, analysis.height
        else:
            scale = proc_max / float(natural_max)
            proc_w = int(analysis.width * scale)
            proc_h = int(analysis.height * scale)
        face_quads = scaled_face_quads(document, proc_w, proc_h)
        expected = EXPECTED_FACES_BY_SIDE[side]
        by_face = stickers_by_face_via_hull(analysis.stickers, face_quads, expected)

        centers_with_meta: List[Tuple[int, float, float, str]] = []
        for face in expected:
            stickers = by_face.get(face, [])
            if not stickers:
                continue
            gt_colors = face_colors_from_state(gt_state, face)
            rgbs = [tuple(int(v) for v in s.rgb) for s in stickers]
            assignments = assign_multiset(rgbs, gt_colors)
            for sticker, rgb, multi in zip(stickers, rgbs, assignments):
                counter += 1
                cur = sticker.match.color
                table_rows.append({
                    "n": counter,
                    "side": side,
                    "face": face,
                    "rgb": list(rgb),
                    "multiset": multi,
                    "classifier": cur,
                })
                centers_with_meta.append((counter, float(sticker.center[0]), float(sticker.center[1]), face))

        out_png = OUT_DIR / f"cal-set-{set_id}-{side}.png"
        render_annotated(image_path, centers_with_meta, out_png)
        print(f"wrote {out_png}", file=sys.stderr)

    # Per-face ground-truth color multiset (for reference)
    side_a_faces = "URF"
    side_b_faces = "DLB"
    gt_multiset_lines = []
    for side, faces in (("A", side_a_faces), ("B", side_b_faces)):
        for face in faces:
            cols = face_colors_from_state(gt_state, face)
            gt_multiset_lines.append(f"  side {side} face {face}: " + ", ".join(sorted(cols)))

    # Write label sheet
    sheet_path = OUT_DIR / f"cal-set-{set_id}-labels.txt"
    with sheet_path.open("w") as f:
        f.write(f"# Hand-label calibration: set {set_id}\n#\n")
        f.write(f"# Images:\n")
        f.write(f"#   /tmp/cal-set-{set_id}-A.png  (image A: U/R/F faces)\n")
        f.write(f"#   /tmp/cal-set-{set_id}-B.png  (image B: D/L/B faces)\n#\n")
        f.write("# For each numbered sticker, write the TRUE color you see in the photo\n")
        f.write("# into the `hand` column. Use one of: white, yellow, red, orange, green, blue.\n")
        f.write("# (Or paste the whole face block at the bottom; see HAND_LABELS block below.)\n#\n")
        f.write("# Per-face ground-truth color MULTISETS (these counts must match what you label):\n")
        for line in gt_multiset_lines:
            f.write(f"# {line}\n")
        f.write("#\n")
        f.write("# Per-sticker table (for reference):\n")
        f.write("#   n  side face   multiset    classifier  rgb\n")
        for row in table_rows:
            multi = row['multiset'] or "(extra)"
            f.write(
                f"#  {row['n']:>2}   {row['side']}    {row['face']}    "
                f"{multi:<10s}  {row['classifier']:<10s}  "
                f"{tuple(row['rgb'])}\n"
            )
        f.write("\n")
        f.write("# ===== FILL IN BELOW =====\n")
        f.write("# Format: one line per sticker as `N=color`. Order doesn't matter.\n")
        f.write("# Examples:\n#   1=white\n#   2=red\n#   ...\n\n")
        f.write("HAND_LABELS:\n")
        for row in table_rows:
            f.write(f"{row['n']}=\n")
    print(f"wrote {sheet_path}", file=sys.stderr)

    # Also dump the table as JSON so the scoring script can read it without re-running the pipeline.
    json_path = OUT_DIR / f"cal-set-{set_id}-table.json"
    json_path.write_text(json.dumps({"setId": set_id, "rows": table_rows}, indent=2))
    print(f"wrote {json_path}", file=sys.stderr)

    print(f"\n{counter} stickers to label across both images of set {set_id}", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("set_id", help="setId from corpus manifest, e.g. 17")
    args = ap.parse_args()
    return prepare(args.set_id)


if __name__ == "__main__":
    sys.exit(main())
