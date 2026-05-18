#!/usr/bin/env python3
"""Per-face image equalization on rectified cube faces.

Operates on the flat 300×300 face images produced by `tools/rectify_faces.py`
(PR #136). Inputs are perspective-corrected so the geometry is normalized
out — color/illumination work can happen on a regular grid with known
pixel positions.

Three transforms, applied in order:

  1. **Center-anchor white balance**: each face's center sticker IS the
     face's own color by Rubik's invariant (orange for L, blue for B,
     etc.). Measure the center sticker's RGB, compute a per-channel
     scale factor to bring it to the canonical target color, apply to
     the whole face image. For U face: use the outer ring of the center
     sticker (the Rubik's logo sits in the inner area).

  2. **Glare suppression** (optional): clamp specular-highlight pixels
     (very high brightness + away from face's median chromaticity) to
     the patch median. Reduces sticker-glint contamination.

  3. **Bezel margin** (optional): when downstream extracts per-sticker
     patches, shrink the patch fraction slightly to stay further from
     bezel/edge contamination. Trivial pixel-coord adjustment.

This is investigation/eval tooling. NO production-recognizer changes.

The end-to-end measurement of whether equalization closes the OOD-set
classification gap lives in `tools/evaluate_equalize_lift.py` (this
PR), which runs the rectify → [equalize] → classify pipeline twice
(off/on) and reports per-pair lift.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import CANONICAL_RGB, FACE_TO_COLOR, classify_rgb  # noqa: E402


# ---------------- core equalization ----------------


def _sample_center_rgb(
    face_img: Image.Image,
    is_u_face: bool = False,
    sample_fraction: float = 0.30,
) -> Tuple[int, int, int]:
    """Median RGB of the center sticker.

    For non-U faces: sample a 30%-of-cell-width square at the face center.
    For U face: sample an outer ring (the Rubik's logo sits in the inner
    ~50% of the center sticker — sampling the ring avoids it).
    """
    arr = np.asarray(face_img)
    h, w = arr.shape[:2]
    assert w == h, "rectified face must be square"
    cell = w / 3.0
    cx = int((1 + 0.5) * cell)  # center of (row=1, col=1)
    cy = int((1 + 0.5) * cell)
    half_cell = int(cell / 2)

    if is_u_face:
        # Outer ring: sample 4 corner patches of the center cell, far from logo
        patch = max(4, int(cell * 0.18))
        offset = int(cell * 0.30)  # how far from center to sample
        rings = []
        for dx, dy in ((-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)):
            sx = max(0, cx + dx - patch)
            sy = max(0, cy + dy - patch)
            ex = min(w, cx + dx + patch + 1)
            ey = min(h, cy + dy + patch + 1)
            if ex > sx and ey > sy:
                rings.append(arr[sy:ey, sx:ex].reshape(-1, 3))
        if rings:
            stacked = np.concatenate(rings, axis=0)
            return tuple(int(np.median(stacked[:, c])) for c in range(3))  # type: ignore
        # Fallback: sample whole center cell
    half_patch = int(cell * sample_fraction / 2)
    patch = arr[cy - half_patch:cy + half_patch + 1,
                cx - half_patch:cx + half_patch + 1].reshape(-1, 3)
    if patch.size == 0:
        return (0, 0, 0)
    return tuple(int(np.median(patch[:, c])) for c in range(3))  # type: ignore


def white_balance_face(
    face_img: Image.Image,
    true_face: str,
    target_palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
    mode: str = "brightness",
) -> Image.Image:
    """Adjust the face image so the center sticker approaches the
    canonical target color for that face.

    Two modes:
      * "brightness" (default, safer): scalar luminance scale only.
        Preserves chromaticity (channel ratios), only adjusts overall
        face brightness. Won't distort colors.
      * "per_channel" (aggressive): scale each RGB channel independently
        so the center exactly matches target. CAN distort non-center
        sticker colors when the anchor isn't neutral (face center is
        the face's own color, NOT neutral grey/white). Use only with
        a real grey-world calibration or when target is white/grey.

    For U face: center has the Rubik's logo, so we sample the outer ring
    of the center sticker instead of the geometric center patch.
    """
    palette = target_palette or CANONICAL_RGB
    target_color_name = FACE_TO_COLOR[true_face]
    target = palette[target_color_name]
    observed = _sample_center_rgb(face_img, is_u_face=(true_face == "U"))

    if mode == "per_channel":
        scale = np.array([
            min(3.0, max(0.3, t / max(8, o)))
            for t, o in zip(target, observed)
        ])
    else:
        # Brightness-only: single scalar based on luminance.
        observed_lum = sum(observed) / 3.0
        target_lum = sum(target) / 3.0
        s = min(3.0, max(0.3, target_lum / max(8.0, observed_lum)))
        scale = np.array([s, s, s])

    arr = np.asarray(face_img, dtype=np.float64)
    arr = arr * scale.reshape(1, 1, 3)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def suppress_glare(face_img: Image.Image, brightness_threshold: int = 235) -> Image.Image:
    """Clamp specular-highlight pixels (very bright) to the median brightness
    of their surrounding 7×7 patch. Cheap glare reduction; helps where
    sticker glints dominate the patch median during classification."""
    arr = np.asarray(face_img, dtype=np.uint8).copy()
    # Identify "very bright" pixels (likely glare). Using max channel.
    max_chan = arr.max(axis=2)
    glare_mask = max_chan > brightness_threshold
    if not glare_mask.any():
        return Image.fromarray(arr)

    # For each glare pixel, replace with median of 7×7 neighborhood
    h, w = arr.shape[:2]
    radius = 3
    ys, xs = np.where(glare_mask)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        neighborhood = arr[y0:y1, x0:x1].reshape(-1, 3)
        # Median ignoring the glare pixel itself (rough, but cheap)
        arr[y, x] = np.median(neighborhood, axis=0).astype(np.uint8)
    return Image.fromarray(arr)


def equalize_face(
    face_img: Image.Image,
    true_face: str,
    do_white_balance: bool = True,
    do_glare: bool = True,
) -> Image.Image:
    """Apply equalization pipeline: white balance → glare suppression."""
    out = face_img
    if do_white_balance:
        out = white_balance_face(out, true_face)
    if do_glare:
        out = suppress_glare(out)
    return out


# ---------------- CLI: equalize one face / one set ----------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to a rectified face PNG (300×300)")
    ap.add_argument("--true-face", required=True, choices=list("URFDLB"),
                    help="The face this image actually shows (U/R/F/D/L/B)")
    ap.add_argument("--output", required=True, help="Output path for equalized PNG")
    ap.add_argument("--no-white-balance", action="store_true")
    ap.add_argument("--no-glare", action="store_true")
    args = ap.parse_args()

    img = Image.open(args.input).convert("RGB")
    out = equalize_face(
        img, args.true_face,
        do_white_balance=not args.no_white_balance,
        do_glare=not args.no_glare,
    )
    out.save(args.output, "PNG", optimize=True)
    print(f"wrote {args.output}", file=sys.stderr)

    # Diagnostics: center RGB before/after + target
    target = CANONICAL_RGB[FACE_TO_COLOR[args.true_face]]
    before = _sample_center_rgb(img, is_u_face=(args.true_face == "U"))
    after = _sample_center_rgb(out, is_u_face=(args.true_face == "U"))
    print(f"true face: {args.true_face} (target color = {target})", file=sys.stderr)
    print(f"  center before: {before}", file=sys.stderr)
    print(f"  center after:  {after}", file=sys.stderr)
    print(f"  (closer to target = better calibration)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
