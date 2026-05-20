"""Test driver + visualizer for tools/global_cube_model.py.

Runs the global cube model fitter on one or more corpus pairs,
renders the fitted 3-face / 27-cell overlay, and writes per-pair JSON
sidecars with the fit parameters + quality scores.

Usage:
    .venv/bin/python tools/test_global_cube_model.py \\
        --sets 17 21 30 31 44 47 57 58 61 \\
        --out /tmp/global_cube_model_results

DIAGNOSTICS-ONLY tool. Validates the global cube model approach
(per the 2026-05-20 pivot Decision Log entry) on the 18-pair
worst-case corpus before any recognition wiring.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.global_cube_model import (  # noqa: E402
    GlobalCubeModel, fit_global_cube_model,
)
from tools.interior_bezel_detection import (  # noqa: E402
    detect_interior_bezel_lines,
)


def _exif_correct(path: Path) -> Image.Image:
    img = Image.open(path)
    return ImageOps.exif_transpose(img).convert("RGB")


def _compute_rembg_mask(rgb: np.ndarray) -> np.ndarray:
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
    model: GlobalCubeModel,
    title: str,
) -> Image.Image:
    """3-panel side-by-side: photo / mask / photo with model overlay."""
    h, w = rgb.shape[:2]
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
    max_dim = 700
    scale = min(1.0, max_dim / max(cw, ch))
    nw, nh = int(cw * scale), int(ch * scale)

    photo_full = Image.fromarray(rgb)
    photo = photo_full.crop(crop_box).resize((nw, nh))
    mask_full = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8))
    mask_img = mask_full.crop(crop_box).resize((nw, nh))
    overlay = photo.copy()
    draw = ImageDraw.Draw(overlay)

    def tx(x: float, y: float) -> Tuple[float, float]:
        return ((x - crop_box[0]) * scale, (y - crop_box[1]) * scale)

    if model is not None and model.face_quads:
        face_colors = {
            "face_01": (255, 100, 100),
            "face_12": (100, 255, 100),
            "face_02": (100, 100, 255),
        }
        # Sticker cells (thin lines)
        for name, cells in model.sticker_cells.items():
            color = face_colors.get(name, (200, 200, 200))
            for cell in cells:
                pts = [tx(p[0], p[1]) for p in cell]
                pts.append(pts[0])
                for i in range(len(pts) - 1):
                    draw.line(pts[i] + pts[i + 1], fill=color, width=1)
        # Face quads (thick outlines)
        for name, quad in model.face_quads.items():
            color = face_colors.get(name, (200, 200, 200))
            pts = [tx(p[0], p[1]) for p in quad]
            pts.append(pts[0])
            for i in range(len(pts) - 1):
                draw.line(pts[i] + pts[i + 1], fill=color, width=3)
        # Hexagon vertices
        for k in ("001", "011", "010", "110", "100", "101"):
            cx_s, cy_s = tx(*model.visible_corners[k])
            draw.ellipse((cx_s - 5, cy_s - 5, cx_s + 5, cy_s + 5),
                         fill=(255, 255, 0), outline=(0, 0, 0))
        # Cube center
        cc = tx(*model.cube_center)
        r = 10
        draw.ellipse((cc[0] - r, cc[1] - r, cc[0] + r, cc[1] + r),
                     fill=(255, 255, 255), outline=(0, 0, 0), width=2)

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", size=16
        )
    except Exception:
        font = ImageFont.load_default()

    if model is not None:
        header = (
            f"{title}  fit={model.fit_quality:.2f}  "
            f"IoU={model.debug.get('final_silhouette_iou', '?')}  "
            f"bezel={model.debug.get('final_bezel_match', '?')}  "
            f"L={model.edge_length_px:.0f}px"
        )
    else:
        header = f"{title}  (no fit)"
    draw.rectangle((0, 0, nw, 24), fill=(0, 0, 0))
    draw.text((6, 3), header, fill=(255, 255, 255), font=font)

    panel = Image.new("RGB", (nw * 3, nh), (32, 32, 32))
    panel.paste(photo, (0, 0))
    panel.paste(mask_img.convert("RGB"), (nw, 0))
    panel.paste(overlay, (nw * 2, 0))
    pdraw = ImageDraw.Draw(panel)
    for i, label in enumerate(("photo", "rembg mask", "global model overlay")):
        pdraw.rectangle((nw * i, nh - 22, nw * i + 200, nh), fill=(0, 0, 0))
        pdraw.text((nw * i + 6, nh - 20), label, fill=(255, 255, 255), font=font)
    return panel


def _serialize_model(m: GlobalCubeModel) -> dict:
    return {
        "cube_center": list(m.cube_center),
        "axis_angles_deg": [round(math.degrees(a), 2) for a in m.axis_angles_rad],
        "edge_length_px": round(m.edge_length_px, 1),
        "fit_loss": round(m.fit_loss, 4),
        "fit_quality": round(m.fit_quality, 3),
        "visible_corners": {
            k: [round(v[0], 1), round(v[1], 1)]
            for k, v in m.visible_corners.items()
        },
        "face_quads": {
            k: [[round(p[0], 1), round(p[1], 1)] for p in q]
            for k, q in m.face_quads.items()
        },
        "debug": m.debug,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", nargs="+", default=["31"])
    parser.add_argument("--sides", nargs="+", choices=["A", "B"], default=["A", "B"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-optimize", action="store_true",
                        help="Skip optimization; show initialization only")
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
                print(f"[set {set_id} {side}] SKIP: {path}", file=sys.stderr)
                continue
            print(f"[set {set_id} {side}] processing {path.name} ...", file=sys.stderr)
            img = _exif_correct(path)
            rgb = np.asarray(img, dtype=np.uint8)
            mask = _compute_rembg_mask(rgb)
            det = detect_interior_bezel_lines(rgb, mask)
            model = fit_global_cube_model(
                det, mask, optimize=(not args.no_optimize)
            )
            row = {
                "setId": set_id,
                "side": side,
                "imagePath": str(path),
                "imageSize": [rgb.shape[1], rgb.shape[0]],
                "bezel_sq": round(det.signal_quality, 3),
                **(_serialize_model(model) if model else {"error": "no_model"}),
            }
            summary.append(row)
            if model is not None:
                print(
                    f"[set {set_id} {side}]   "
                    f"fit_quality={model.fit_quality:.3f}  "
                    f"IoU={model.debug.get('final_silhouette_iou', model.debug.get('initial_silhouette_iou'))}  "
                    f"bezel={model.debug.get('final_bezel_match', model.debug.get('initial_bezel_match'))}  "
                    f"L={model.edge_length_px:.0f}px  "
                    f"shift={model.debug.get('center_shift_from_init_px', 0)}px  "
                    f"iter={model.debug.get('optimizer_iterations', 'init-only')}",
                    file=sys.stderr,
                )
            else:
                print(f"[set {set_id} {side}]   no model (init failed)", file=sys.stderr)

            if args.out is not None:
                panel = _draw_visualization(
                    rgb, mask, model, title=f"Set {set_id} {side}"
                )
                png_path = args.out / f"set_{set_id}_{side}_overlay.png"
                json_path = args.out / f"set_{set_id}_{side}_data.json"
                panel.save(png_path, optimize=True)
                with json_path.open("w") as f:
                    json.dump(row, f, indent=2)

    if args.out is not None:
        with (args.out / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] wrote summary.json", file=sys.stderr)
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
