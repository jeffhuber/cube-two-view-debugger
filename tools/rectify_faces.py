#!/usr/bin/env python3
"""Rectify (perspective-correct) the 3 visible cube faces from one
isometric image, given their 4-corner face quads.

Takes any source of face quads (hand-labeled hulls from runs/labels/,
auto-geometry proposer outputs, or future recognizer_mask output) and
produces flat 300×300 face images plus a per-sticker color extraction
that's trivial pixel-slicing instead of per-sticker homography.

Why this exists:
  * Sticker color sampling becomes a flat 3×3 grid lookup at known
    pixel positions — no homography per sample, no inset tuning, no
    canonical-corner ambiguity. The geometry math happens once per
    face instead of once per sticker.
  * Visual debugging is dramatically clearer: a flat 100×100 patch
    shows "is this red or orange?" obviously; a perspective rhombus
    does not.
  * Direct bridge to the synthetic corpus (PR #132): both produce
    300×300 flat face images, so domain-transfer tests become
    apples-to-apples without perspective noise in the way.
  * Mirrors the Fixer's diamond view — same concept, just made into
    the recognizer's internal representation.
  * Natural input format for a learned (CNN) classifier in a future PR.

This is investigation/eval tooling. NO production-recognizer changes.

Usage examples:

  # Render rectified faces from a hand-labeled hull:
  .venv/bin/python tools/rectify_faces.py \\
      --image /Users/jhuber/Downloads/Set\\ 15\\ -\\ A\\ -\\ white\\ up\\ IMG_6707.JPG \\
      --hull-label runs/labels/<id>-set-15-a-geometry-label.json \\
      --output /tmp/rect-set15-a

  # Auto-discover everything for a corpus pair:
  .venv/bin/python tools/rectify_faces.py --set 15

  # Build a flat-net grid PNG (U/R/F + D/L/B side-by-side):
  .venv/bin/python tools/rectify_faces.py --set 15 --net
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import COLOR_TO_FACE, FACE_TO_COLOR, classify_rgb  # noqa: E402
from rubik_recognizer.validation import FACE_ORDER  # noqa: E402
from tools.extract_color_samples import (  # noqa: E402
    EXPECTED_FACES_BY_SIDE,
    discover_additional_tasks,
    face_colors_from_state,
    load_corpus_tasks,
    parse_ground_truth,
)
from tools.sample_stickers_from_hull import (  # noqa: E402
    apply_orientation,
    canonical_corner_order,
    discover_orientation,
    latest_hull_label,
    load_hull_label,
    scaled_face_quads,
)

PROCESSING_MAX = 1150
DEFAULT_OUTPUT_DIR = Path("/tmp")
DEFAULT_FACE_SIZE = 300


# ---------------- perspective transform ----------------


def _perspective_coeffs(
    src_quad: Sequence[Tuple[float, float]],
    dst_size: int,
) -> Tuple[float, ...]:
    """Compute PIL's 8-coefficient perspective transform that maps the
    OUTPUT image (dst_size × dst_size square) back to the SOURCE quad
    in the input image. PIL.Image.transform with PERSPECTIVE wants the
    inverse map: for each output pixel, where does it come from in the
    source.

    src_quad is the face quad in source image coords, in canonical
    CW-from-N order — the same order sample_stickers_from_hull uses,
    so corner correspondence stays consistent across the toolchain."""
    # PIL convention: dst→src mapping. Output is dst_size square.
    dst = [(0, 0), (dst_size, 0), (dst_size, dst_size), (0, dst_size)]
    src = list(src_quad)
    # Solve the 8 equations for the perspective coefficients
    A = []
    b = []
    for (dx, dy), (sx, sy) in zip(dst, src):
        A.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        A.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
        b.append(sx)
        b.append(sy)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    return tuple(coeffs.tolist())


def rectify_face(
    image: Image.Image,
    face_quad: Sequence[Tuple[float, float]],
    output_size: int = DEFAULT_FACE_SIZE,
) -> Image.Image:
    """Warp a face quad from the source image to a flat (output_size ×
    output_size) square. Uses bicubic interpolation."""
    canonical = canonical_corner_order([tuple(p) for p in face_quad])
    coeffs = _perspective_coeffs(canonical, output_size)
    return image.transform(
        (output_size, output_size),
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC,
    )


# ---------------- per-sticker color extraction ----------------


@dataclass
class StickerSample:
    row: int
    col: int
    rgb: Tuple[int, int, int]
    classified_color: str
    center_xy: Tuple[int, int]  # in rectified-face coords


def extract_stickers_from_rectified(
    face_img: Image.Image,
    patch_fraction: float = 0.40,
) -> List[List[StickerSample]]:
    """Sample 9 stickers from a rectified face. Sticker centers are at
    known pixel coords ((1/6, 3/6, 5/6) * face_size). We sample a square
    patch of `patch_fraction` of the cell width (default 40% — well
    inside the sticker, away from bezels) and take the median RGB."""
    w, h = face_img.size
    assert w == h, "rectified face must be square"
    cell = w / 3.0
    patch_half = int(cell * patch_fraction / 2)
    arr = np.asarray(face_img)
    out: List[List[StickerSample]] = []
    for r in range(3):
        row_out: List[StickerSample] = []
        cy = int((r + 0.5) * cell)
        for c in range(3):
            cx = int((c + 0.5) * cell)
            patch = arr[max(0, cy - patch_half):cy + patch_half + 1,
                        max(0, cx - patch_half):cx + patch_half + 1]
            if patch.size == 0:
                rgb = (0, 0, 0)
            else:
                rgb = tuple(int(np.median(patch.reshape(-1, 3)[:, ch])) for ch in range(3))
            row_out.append(StickerSample(
                row=r, col=c, rgb=rgb,
                classified_color=classify_rgb(rgb).color,
                center_xy=(cx, cy),
            ))
        out.append(row_out)
    return out


# ---------------- loaders ----------------


@dataclass
class RectifyInput:
    image_path: Path
    side: str  # "A" or "B"
    face_quads: Dict[str, List[Tuple[float, float]]]  # in processing-resolution image coords
    gt_state: Optional[str] = None  # 54-char URFDLB for accuracy comparison


def load_image_processed(image_path: Path) -> Tuple[Image.Image, int, int]:
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    natural_max = max(image.size)
    if natural_max > PROCESSING_MAX:
        scale = PROCESSING_MAX / float(natural_max)
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return image, image.width, image.height


def load_inputs_for_set(set_id: str) -> List[RectifyInput]:
    """Auto-discover (set, side) pairs from corpus_manifest + Downloads."""
    tasks = load_corpus_tasks(REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json")
    tasks.extend(discover_additional_tasks({t.set_id for t in tasks}))
    inputs: List[RectifyInput] = []
    for task in tasks:
        if task.set_id != set_id:
            continue
        try:
            gt_state = parse_ground_truth(task.ground_truth)
        except Exception:
            gt_state = None
        for side, image_path in (("A", task.image_a), ("B", task.image_b)):
            label_path = latest_hull_label(set_id, side)
            if label_path is None or not image_path.exists():
                continue
            image, proc_w, proc_h = load_image_processed(image_path)
            doc = load_hull_label(label_path)
            face_quads_scaled = scaled_face_quads(doc, proc_w, proc_h)
            face_quads = {
                f: [(float(x), float(y)) for (x, y) in q]
                for f, q in face_quads_scaled.items()
            }
            inputs.append(RectifyInput(
                image_path=image_path, side=side,
                face_quads=face_quads, gt_state=gt_state,
            ))
    return inputs


# ---------------- output helpers ----------------


def render_face_panel(
    rectified: Image.Image,
    label: str,
    stickers: List[List[StickerSample]],
    show_grid: bool = True,
    show_classified: bool = True,
) -> Image.Image:
    """Add a label + optional 3×3 grid lines + classified-color tags to a
    rectified face image."""
    img = rectified.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)
        big_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
        big_font = ImageFont.load_default()

    if show_grid:
        cell = w / 3
        for i in range(1, 3):
            draw.line([(i * cell, 0), (i * cell, h)], fill=(255, 255, 255), width=1)
            draw.line([(0, i * cell), (w, i * cell)], fill=(255, 255, 255), width=1)

    if show_classified:
        for row in stickers:
            for s in row:
                cx, cy = s.center_xy
                draw.text((cx - 5, cy - 8), s.classified_color[0].upper(),
                          font=font, fill=(0, 0, 0))

    # Face label in top-left
    draw.rectangle((0, 0, 40, 36), fill=(0, 0, 0))
    draw.text((6, 4), label, font=big_font, fill=(255, 255, 255))
    return img


def build_net_grid(rectified_panels: Dict[str, Image.Image], face_size: int) -> Image.Image:
    """Lay out faces in the standard "T" net format:
            U
          F R B  (using F-R-B order from view A, plus D below)
            D
    For yaw=0 3-face views (URF visible in A, DLB in B), we use:
            U R       D L
            F .       B .
    rendered side-by-side to compare A and B. Non-zero capture yaw changes
    which side faces occupy these slots."""
    pad = 10
    # 2 sides × 2 rows × 2 cols = 4×2 grid of slots, but only fill 3 per side
    cols = 6  # U R | _ D L | _
    rows = 2
    canvas_w = cols * face_size + (cols + 1) * pad
    canvas_h = rows * face_size + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (40, 40, 40))

    def paste_face(face: str, col: int, row: int):
        if face not in rectified_panels:
            return
        x = pad + col * (face_size + pad)
        y = pad + row * (face_size + pad)
        canvas.paste(rectified_panels[face], (x, y))

    # Side A yaw=0: U R F arranged top-row U/R, bottom-row F (mimics partial net)
    paste_face("U", 0, 0)
    paste_face("R", 1, 0)
    paste_face("F", 0, 1)
    # Side B
    paste_face("D", 3, 0)
    paste_face("L", 4, 0)
    paste_face("B", 3, 1)
    return canvas


# ---------------- main pipeline ----------------


def process_input(
    inp: RectifyInput,
    output_prefix: Path,
    face_size: int,
    write_panels: bool,
    sticker_data: List[Dict],
) -> Dict[str, Image.Image]:
    image, _, _ = load_image_processed(inp.image_path)
    panels: Dict[str, Image.Image] = {}
    for face, quad in inp.face_quads.items():
        rectified = rectify_face(image, quad, face_size)
        stickers = extract_stickers_from_rectified(rectified)
        # Optional GT accuracy. Account for two known sources of
        # label-vs-actual mismatch from PR #126:
        #   1. The U-center has a Rubik's logo → its center classifies as
        #      red/brown, not white. For the side's known anchor (U for A,
        #      D for B), trust the labeler's name.
        #   2. The labeler's L/B labels on image B may be swapped because
        #      of how the physical flip mechanically rotates the cube. For
        #      non-anchor faces, classify the center sticker and use that
        #      to map to the *true* face identity for GT lookup.
        anchor = "U" if inp.side == "A" else "D"
        true_face = face
        if face != anchor:
            center_sticker = stickers[1][1]
            cls = classify_rgb(center_sticker.rgb).color
            true_face = COLOR_TO_FACE.get(cls, face)
        gt_colors: Optional[List[str]] = None
        if inp.gt_state is not None and true_face in FACE_ORDER:
            try:
                gt_colors = face_colors_from_state(inp.gt_state, true_face)
            except Exception:
                gt_colors = None
        correct = None
        orientation_info: Optional[Dict] = None
        if gt_colors is not None and len(gt_colors) == 9:
            flat = [s for row in stickers for s in row]
            rgbs = [s.rgb for s in flat]
            # Discover the rotation/mirror that aligns the rectified
            # face's canonical-corner-order with the GT row-major order.
            # Reuses the same brute-force search from PR #126 — robust to
            # the U-logo + L/B-swap + yaw confusions documented there.
            mirror, rot, _score = discover_orientation(rgbs, gt_colors)
            aligned_flat = apply_orientation(flat, mirror, rot)
            correct = sum(
                1 for s, gt in zip(aligned_flat, gt_colors)
                if s.classified_color == gt
            )
            orientation_info = {"mirror": mirror, "rotation_quarters_ccw": rot}
        sticker_data.append({
            "side": inp.side,
            "labelFace": face,
            "trueFace": true_face,
            "stickers": [
                {"row": s.row, "col": s.col, "rgb": list(s.rgb),
                 "classified": s.classified_color}
                for row in stickers for s in row
            ],
            "groundTruth": gt_colors,
            "correctVsGt": correct,
            "orientationVsGt": orientation_info,
        })
        panels[face] = render_face_panel(rectified, face, stickers)
        if write_panels:
            out_path = output_prefix.with_name(f"{output_prefix.name}_{inp.side}_{face}.png")
            panels[face].save(out_path, "PNG", optimize=True)
    return panels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src_grp = ap.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--set", help="setId (auto-discovers image + hull label)")
    src_grp.add_argument("--image", help="explicit image path")
    ap.add_argument("--hull-label", help="explicit hull-label JSON (required if --image)")
    ap.add_argument("--side", default="A", choices=("A", "B"),
                    help="side identity for --image mode (default A)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "rect"),
                    help="output filename prefix")
    ap.add_argument("--face-size", type=int, default=DEFAULT_FACE_SIZE)
    ap.add_argument("--net", action="store_true",
                    help="Also render a combined U/R/F + D/L/B net PNG")
    ap.add_argument("--no-panels", action="store_true",
                    help="Skip per-face PNGs (only emit JSON + optional net)")
    args = ap.parse_args()

    inputs: List[RectifyInput]
    if args.set:
        inputs = load_inputs_for_set(args.set)
        if not inputs:
            print(f"set {args.set}: no labeled image discoverable", file=sys.stderr)
            return 2
        prefix = Path(args.output)
        if str(prefix) == str(DEFAULT_OUTPUT_DIR / "rect"):
            prefix = DEFAULT_OUTPUT_DIR / f"rect-set{args.set}"
    else:
        if not args.hull_label:
            print("--image requires --hull-label", file=sys.stderr)
            return 2
        image_path = Path(args.image)
        image, proc_w, proc_h = load_image_processed(image_path)
        doc = load_hull_label(Path(args.hull_label))
        face_quads_scaled = scaled_face_quads(doc, proc_w, proc_h)
        face_quads = {f: [(float(x), float(y)) for (x, y) in q]
                       for f, q in face_quads_scaled.items()}
        inputs = [RectifyInput(image_path=image_path, side=args.side,
                                face_quads=face_quads)]
        prefix = Path(args.output)

    all_panels: Dict[str, Image.Image] = {}
    sticker_data: List[Dict] = []
    for inp in inputs:
        panels = process_input(
            inp, prefix, args.face_size,
            write_panels=not args.no_panels,
            sticker_data=sticker_data,
        )
        # When building a net, panels from both sides merge — use side-prefix keys
        for face, panel in panels.items():
            all_panels[face] = panel
        print(f"  side {inp.side}: rectified {len(panels)} faces",
              file=sys.stderr)
        for s in sticker_data[-len(panels):]:
            if s.get("correctVsGt") is not None:
                tag = "" if s['labelFace'] == s['trueFace'] else f"  (labeled {s['labelFace']} → actual {s['trueFace']})"
                print(f"    face {s['labelFace']}: {s['correctVsGt']}/9 classified correctly{tag}",
                      file=sys.stderr)

    # Metadata JSON
    meta = {
        "stickerSamples": sticker_data,
        "faceSize": args.face_size,
    }
    meta_path = prefix.with_name(prefix.name + ".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {meta_path}", file=sys.stderr)

    if args.net:
        net_img = build_net_grid(all_panels, args.face_size)
        net_path = prefix.with_name(prefix.name + "_net.png")
        net_img.save(net_path, "PNG", optimize=True)
        print(f"wrote {net_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
