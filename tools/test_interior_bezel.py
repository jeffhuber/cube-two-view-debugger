"""Test driver + visualizer for tools/interior_bezel_detection.py.

Runs `detect_interior_bezel_lines` against one or more corpus pairs,
renders a side-by-side visualization, and writes per-pair JSON sidecars
with the detection result.

Usage:
    .venv/bin/python tools/test_interior_bezel.py \\
        --sets 47 21 17 30 31 44 57 58 61 \\
        --out /tmp/interior_bezel_results

Default --sets is `47` (the canonical extreme-degeneracy case from
PR #176's taxonomy). With no --out, results print as JSON to stdout
and no PNGs are written.

Per-pair output (when --out is supplied):
    set_<N>_<side>_overlay.png    side-by-side visualization
    set_<N>_<side>_data.json      structured detection result

DIAGNOSTICS-ONLY tool. Validates whether interior bezel-line detection
is a viable signal for finding h1/h3/h5 (the 3 hexagon vertices INSIDE
the silhouette where hull-based fitters provably cannot find them).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.interior_bezel_detection import (  # noqa: E402
    InteriorBezelDetection,
    detect_interior_bezel_lines,
)


def _exif_correct(path: Path) -> Image.Image:
    """Load + EXIF-correct an iPhone JPEG. Required because iPhone
    landscape JPEGs are stored with `Orientation=6` (rotate 90 CW) and
    raw pixels are 4032x3024 sideways."""
    img = Image.open(path)
    return ImageOps.exif_transpose(img).convert("RGB")


def _compute_rembg_mask(rgb: np.ndarray) -> np.ndarray:
    """Lazy rembg call to compute a silhouette mask. Caches the session
    so multiple pairs don't reload the model."""
    from rembg import new_session, remove  # type: ignore

    global _REMBG_SESSION
    try:
        sess = _REMBG_SESSION
    except NameError:
        sess = new_session("u2net")
        _REMBG_SESSION = sess  # type: ignore

    pil = Image.fromarray(rgb)
    rgba = remove(pil, session=sess)
    alpha = np.array(rgba.split()[-1], dtype=np.uint8)
    return alpha > 128


def _load_manifests() -> list:
    """Return both hard-case + corpus manifests so we can find any set."""
    out = []
    for fname in ("hard_case_manifest.json", "corpus_manifest.json"):
        with (REPO_ROOT / "tests" / "fixtures" / fname).open() as f:
            out.append(json.load(f))
    return out


def _resolve_pair_paths(manifests: list, set_id: str) -> Tuple[Path, Path]:
    for manifest in manifests:
        for entry in manifest["pairs"]:
            if entry["setId"] == set_id:
                return Path(entry["imageAPath"]), Path(entry["imageBPath"])
    raise SystemExit(f"set {set_id!r} not in any manifest")


def _draw_visualization(
    rgb: np.ndarray,
    mask: np.ndarray,
    detection: InteriorBezelDetection,
    title: str,
) -> Image.Image:
    """Side-by-side panel: original photo, mask, photo + overlay.

    Overlay draws:
      * cube_center as a magenta dot
      * each boundary line as a magenta segment from center outward
      * signal_quality printed in the top-left corner
    """
    h, w = rgb.shape[:2]
    # Crop to silhouette bounding box (with 50px margin) so the cube
    # fills the panel instead of being lost in empty background.
    ys_m, xs_m = np.where(mask)
    if len(xs_m) == 0:
        crop_box = (0, 0, w, h)
    else:
        margin = 50
        x0 = max(0, int(xs_m.min()) - margin)
        y0 = max(0, int(ys_m.min()) - margin)
        x1 = min(w, int(xs_m.max()) + margin)
        y1 = min(h, int(ys_m.max()) + margin)
        crop_box = (x0, y0, x1, y1)
    cw = crop_box[2] - crop_box[0]
    ch = crop_box[3] - crop_box[1]

    # Downscale for visualization
    max_dim = 900
    scale = min(1.0, max_dim / max(cw, ch))
    nw, nh = int(cw * scale), int(ch * scale)

    photo_full = Image.fromarray(rgb)
    photo = photo_full.crop(crop_box).resize((nw, nh))
    mask_full = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8))
    mask_img = mask_full.crop(crop_box).resize((nw, nh))
    overlay = photo.copy()
    # Coordinate transform: image_xy -> overlay_xy
    # overlay_x = (image_x - crop_box[0]) * scale
    # overlay_y = (image_y - crop_box[1]) * scale
    draw = ImageDraw.Draw(overlay)

    def _tx(x: float, y: float) -> Tuple[float, float]:
        """image_xy -> cropped-overlay_xy"""
        return ((x - crop_box[0]) * scale, (y - crop_box[1]) * scale)

    if detection.cube_center is not None:
        cx, cy = detection.cube_center
        cx_s, cy_s = _tx(cx, cy)

        # boundary lines first (so the cube-center dot lands on top).
        # Draw full chords through the silhouette (extend in both
        # directions, not just the longer end).
        line_colors = [(255, 0, 255), (255, 255, 0), (0, 255, 255)]
        import math as _math
        for i, theta in enumerate(detection.boundary_angles or []):
            dx_u = _math.cos(theta)
            dy_u = _math.sin(theta)
            L = max(crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
            p_minus = _tx(cx - dx_u * L, cy - dy_u * L)
            p_plus = _tx(cx + dx_u * L, cy + dy_u * L)
            color = line_colors[i % len(line_colors)]
            draw.line((p_minus[0], p_minus[1], p_plus[0], p_plus[1]),
                      fill=color, width=4)
        # cube-center dot
        r = 14
        draw.ellipse(
            (cx_s - r, cy_s - r, cx_s + r, cy_s + r),
            fill=(255, 255, 255),
            outline=(0, 0, 0),
            width=3,
        )

    # Header text
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", size=18
        )
    except Exception:
        font = ImageFont.load_default()
    n_lines = len(detection.boundary_lines)
    lq_str = " ".join(f"{q:.2f}" for q in detection.line_qualities) or "—"
    n_iter = detection.debug.get("iter_count", "?")
    header = (
        f"{title}  sq={detection.signal_quality:.2f}  "
        f"per-line=[{lq_str}]  lines={n_lines}  iter={n_iter}"
    )
    draw.rectangle((0, 0, nw, 28), fill=(0, 0, 0, 180))
    draw.text((6, 4), header, fill=(255, 255, 255), font=font)

    # Compose 3 panels horizontally
    panel = Image.new("RGB", (nw * 3, nh), (32, 32, 32))
    panel.paste(photo, (0, 0))
    panel.paste(mask_img.convert("RGB"), (nw, 0))
    panel.paste(overlay, (nw * 2, 0))
    # Panel labels
    pdraw = ImageDraw.Draw(panel)
    for i, label in enumerate(("photo", "rembg mask", "interior bezel overlay")):
        pdraw.rectangle((nw * i, nh - 22, nw * i + 200, nh), fill=(0, 0, 0, 200))
        pdraw.text((nw * i + 6, nh - 20), label, fill=(255, 255, 255), font=font)

    return panel


def _serialize_detection(d: InteriorBezelDetection) -> dict:
    import math as _math
    return {
        "cube_center": list(d.cube_center) if d.cube_center else None,
        "boundary_lines": [
            [[round(p[0], 1), round(p[1], 1)] for p in seg] for seg in d.boundary_lines
        ],
        "boundary_angles_deg": [
            round(_math.degrees(a), 2) for a in d.boundary_angles
        ],
        # Line equations in ax + by + c = 0 form, one per boundary
        # angle, for downstream slot/cell-level geometric joins.
        # Pair with `tools/interior_bezel_detection.cell_line_diagnostics`
        # to compute per-cell distance + crossing flags.
        "line_equations": [
            [round(eq[0], 6), round(eq[1], 6), round(eq[2], 4)]
            for eq in d.line_equations
        ],
        "line_qualities": [round(q, 3) for q in d.line_qualities],
        "signal_quality": round(d.signal_quality, 3),
        "detector_version": "iterative-v1",
        "debug": d.debug,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["47"],
        help="Set IDs from hard_case_manifest.json (default: 47)",
    )
    parser.add_argument(
        "--sides",
        nargs="+",
        choices=["A", "B"],
        default=["A", "B"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir for PNG overlays and JSON sidecars",
    )
    args = parser.parse_args()

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)

    manifests = _load_manifests()
    summary: List[dict] = []

    for set_id in args.sets:
        path_a, path_b = _resolve_pair_paths(manifests, set_id)
        for side, path in (("A", path_a), ("B", path_b)):
            if side not in args.sides:
                continue
            if not path.exists():
                print(f"[set {set_id} {side}] SKIP: {path} not found", file=sys.stderr)
                continue
            print(f"[set {set_id} {side}] loading + masking {path.name} ...", file=sys.stderr)
            img = _exif_correct(path)
            rgb = np.asarray(img, dtype=np.uint8)
            mask = _compute_rembg_mask(rgb)
            print(f"[set {set_id} {side}]   mask pixels={int(mask.sum()):,}", file=sys.stderr)
            detection = detect_interior_bezel_lines(rgb, mask)
            row = {
                "setId": set_id,
                "side": side,
                "imagePath": str(path),
                "imageSize": [rgb.shape[1], rgb.shape[0]],
                "maskPixels": int(mask.sum()),
                **_serialize_detection(detection),
            }
            summary.append(row)
            lq = ",".join(f"{q:.2f}" for q in detection.line_qualities)
            print(
                f"[set {set_id} {side}]   sq={detection.signal_quality:.2f}  "
                f"per-line=[{lq}]  iter={detection.debug.get('iter_count', '?')}  "
                f"converged={detection.debug.get('converged', '?')}  "
                f"final_shift={detection.debug.get('centroid_to_final_shift_px', '?')}  "
                f"lines={len(detection.boundary_lines)}",
                file=sys.stderr,
            )

            if args.out is not None:
                panel = _draw_visualization(
                    rgb, mask, detection, title=f"Set {set_id} {side}"
                )
                png_path = args.out / f"set_{set_id}_{side}_overlay.png"
                json_path = args.out / f"set_{set_id}_{side}_data.json"
                panel.save(png_path, optimize=True)
                with json_path.open("w") as f:
                    json.dump(row, f, indent=2)
                print(f"[set {set_id} {side}]   wrote {png_path.name} + {json_path.name}", file=sys.stderr)

    if args.out is not None:
        index_path = args.out / "summary.json"
        with index_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] wrote summary to {index_path}", file=sys.stderr)
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
