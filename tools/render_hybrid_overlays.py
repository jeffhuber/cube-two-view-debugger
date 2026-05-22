#!/usr/bin/env python3
"""Render visual overlays for human verification of the hybrid pipeline.

For each (set_id, side) pair, produces two side-by-side images:

  1. Original photo with analyze_image's face quads overlaid (drawn
     edges + per-face label showing the re-keyed slot and what
     analyze_image's center_face was). Shows whether quads visually
     align with actual face boundaries.

  2. Rectified 300×300 face with the (1/6, 3/6, 5/6) sample positions
     marked. Shows whether sample points land on actual stickers or
     drift toward bezels/edges.

Intended use is human visual verification of the hypothesis from
PR #152 / the hull-guard negative-result PR: "rectification through
analyze_image's extrapolated quads is genuinely lossy because some
grids span multiple physical faces of the cube." If the quads in (1)
visually align but stickers in (2) are off-center → fix sampling.
If quads in (1) are visibly off (extending across multiple faces) →
fix face quad detection (Path B(2): learned face-quad regressor).

Usage:
  .venv/bin/python tools/render_hybrid_overlays.py --sets 61 49 47 21 17
  .venv/bin/python tools/render_hybrid_overlays.py \\
    --worst-from runs/hybrid_pipeline_report.json --top 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_hybrid_pipeline import (  # noqa: E402
    EXPECTED_FACES_BY_SIDE,
    _load_processing_image,
    _proposer_face_quads,
)
from tools.extract_color_samples import (  # noqa: E402
    discover_additional_tasks,
    load_corpus_tasks,
)
from tools.rectify_faces import DEFAULT_FACE_SIZE, rectify_face  # noqa: E402

CORPUS_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_OUT = REPO_ROOT / "runs" / "hybrid_overlays"

# Distinct colors per face slot for readable overlays
SLOT_COLORS = {
    "U": (50, 200, 50),    # green
    "R": (255, 80, 80),    # red
    "F": (80, 80, 255),    # blue
    "D": (255, 220, 60),   # yellow
    "L": (255, 140, 50),   # orange
    "B": (180, 80, 255),   # purple
}


def _draw_quad_on_image(image: Image.Image, quad: List[Tuple[float, float]],
                        color: Tuple[int, int, int], label: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    pts = [(int(x), int(y)) for (x, y) in quad]
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        draw.line([a, b], fill=color + (255,), width=4)
    for x, y in pts:
        r = 7
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color + (255,))
    cx = sum(p[0] for p in pts) // 4
    cy = sum(p[1] for p in pts) // 4
    text_w, text_h = 90, 20
    draw.rectangle((cx - text_w // 2, cy - text_h // 2,
                    cx + text_w // 2, cy + text_h // 2),
                   fill=(0, 0, 0, 200))
    draw.text((cx - text_w // 2 + 4, cy - text_h // 2 + 3),
              label, fill=color + (255,))
    return out


def _annotate_rectified(face_img: Image.Image, label: str) -> Image.Image:
    """Render rectified face with the 9 sticker-sample positions marked
    (crosshair at center; box at the 40% patch boundary used by
    extract_stickers_from_rectified)."""
    out = face_img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = face_img.size
    cell = w / 3.0
    for r in range(3):
        for c in range(3):
            cx = int((c + 0.5) * cell)
            cy = int((r + 0.5) * cell)
            draw.line([(cx - 12, cy), (cx + 12, cy)], fill=(255, 0, 0, 220), width=2)
            draw.line([(cx, cy - 12), (cx, cy + 12)], fill=(255, 0, 0, 220), width=2)
            half = int(cell * 0.40 / 2)
            draw.rectangle((cx - half, cy - half, cx + half, cy + half),
                           outline=(255, 0, 0, 150), width=1)
    strip_h = 22
    canvas = Image.new("RGB", (w, h + strip_h), (250, 250, 250))
    canvas.paste(out, (0, strip_h))
    cdraw = ImageDraw.Draw(canvas)
    cdraw.text((6, 4), label, fill=(20, 20, 20))
    return canvas


def render_pair(set_id: str, image_a: Path, image_b: Path,
                out_dir: Path, *,
                hull_guard: bool = False,
                fit_error_fallback: bool = False) -> Dict[str, str]:
    written: Dict[str, str] = {}

    for side, image_path in (("A", image_a), ("B", image_b)):
        img, _ = _load_processing_image(image_path)
        quads, debug = _proposer_face_quads(
            image_path, side,
            hull_guard=hull_guard,
            fit_error_fallback=fit_error_fallback,
            processing_image=img,
        )

        overlay = img.copy()
        for slot_label in EXPECTED_FACES_BY_SIDE[side]:
            quad = quads.get(slot_label)
            if quad is None:
                continue
            src_face = debug["selectedPerFace"].get(slot_label, {}).get("sourceCenterFace")
            label_str = f"slot={slot_label} src={src_face or '?'}"
            color = SLOT_COLORS[slot_label]
            overlay = _draw_quad_on_image(overlay, quad, color, label_str)

        max_dim = 900
        if max(overlay.size) > max_dim:
            ratio = max_dim / max(overlay.size)
            overlay = overlay.resize(
                (int(overlay.width * ratio), int(overlay.height * ratio)),
                Image.Resampling.LANCZOS,
            )
        out_path = out_dir / f"set{set_id}_{side}_quads.png"
        overlay.save(out_path, quality=88)
        written[f"{side}_quads"] = str(out_path)

        rectified_strips: List[Image.Image] = []
        for slot_label in EXPECTED_FACES_BY_SIDE[side]:
            quad = quads.get(slot_label)
            if quad is None:
                continue
            try:
                rect = rectify_face(img, quad, output_size=DEFAULT_FACE_SIZE)
            except Exception:
                continue
            src_face = debug["selectedPerFace"].get(slot_label, {}).get("sourceCenterFace")
            label = f"slot={slot_label} src={src_face or '?'}  (sample patches in red)"
            rectified_strips.append(_annotate_rectified(rect, label))

        if rectified_strips:
            total_w = sum(s.width for s in rectified_strips) + 10 * (len(rectified_strips) - 1)
            max_h = max(s.height for s in rectified_strips)
            strip_canvas = Image.new("RGB", (total_w, max_h), (240, 240, 240))
            x = 0
            for s in rectified_strips:
                strip_canvas.paste(s, (x, 0))
                x += s.width + 10
            out_path = out_dir / f"set{set_id}_{side}_rectified.png"
            strip_canvas.save(out_path, quality=88)
            written[f"{side}_rectified"] = str(out_path)

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="*", default=None,
                    help="explicit set IDs to render (e.g. 61 49 47)")
    ap.add_argument("--worst-from", default=None,
                    help="path to a hybrid_pipeline_report.json; renders top-N worst")
    ap.add_argument("--top", type=int, default=5,
                    help="when --worst-from is used: render the N worst pairs")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--hull-guard", action="store_true",
                    help="enable rembg hull guard (production-equivalent path)")
    ap.add_argument("--fit-error-fallback", action="store_true",
                    help="enable topology-aware fit-error fallback (PR #160 + #163)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks_by_id: Dict[str, Tuple[Path, Path]] = {}
    for task in load_corpus_tasks(CORPUS_MANIFEST):
        tasks_by_id[task.set_id] = (task.image_a, task.image_b)
    for task in discover_additional_tasks(set(tasks_by_id.keys())):
        tasks_by_id[task.set_id] = (task.image_a, task.image_b)

    wanted_ids: List[str] = []
    if args.sets:
        wanted_ids = list(args.sets)
    elif args.worst_from:
        report = json.loads(Path(args.worst_from).read_text())
        valid = [r for r in report
                 if "error" not in r and r.get("perStickerAccuracy") is not None]
        valid.sort(key=lambda r: r["perStickerAccuracy"])
        wanted_ids = [r["setId"] for r in valid[:args.top]]
    else:
        print("error: pass --sets or --worst-from", file=sys.stderr)
        return 2

    print(f"rendering overlays for {len(wanted_ids)} pairs → {out_dir}",
          file=sys.stderr)
    for i, set_id in enumerate(wanted_ids, 1):
        if set_id not in tasks_by_id:
            print(f"  [{i}/{len(wanted_ids)}] set {set_id}: NO TASK FOUND",
                  file=sys.stderr)
            continue
        image_a, image_b = tasks_by_id[set_id]
        try:
            written = render_pair(set_id, image_a, image_b, out_dir,
                                  hull_guard=args.hull_guard,
                                  fit_error_fallback=args.fit_error_fallback)
        except Exception as e:
            print(f"  [{i}/{len(wanted_ids)}] set {set_id}: ERROR "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            continue
        files = ", ".join(Path(p).name for p in written.values())
        print(f"  [{i}/{len(wanted_ids)}] set {set_id}: {files}",
              file=sys.stderr)
    print(f"\nwrote {len(wanted_ids)} pairs' overlays to {out_dir}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
