#!/usr/bin/env python3
"""Generate step-by-step visual walkthroughs of `_fit_hexagon_to_hull`
for a set of corpus pairs.

Each walkthrough is a 7-panel image:
  1. Original photo (EXIF-corrected, resized to processing dimension)
  2. Rembg silhouette overlay
  3. Convex hull with all vertices marked
  4. Centroid + topmost-anchor + 6 angular-sector boundary rays
  5. Per-sector winner (farthest-from-centroid in sector), color-coded
  6. Resulting hexagon with collapsed-edge callouts in red
  7. Visvalingam-Whyatt comparison hexagon on the same hull

Each pair also yields a per-sector data dump captured in the companion
`runs/hex_fitter_walkthroughs/{set}_{side}_data.json` for analysis.

Usage:
  .venv/bin/python tools/generate_hex_fitter_walkthroughs.py \\
      --sets 17 21 30 31 44 47 57 58 61 \\
      --out runs/hex_fitter_walkthroughs

Outputs (per set, per side):
  runs/hex_fitter_walkthroughs/set{NN}_{A,B}.png
  runs/hex_fitter_walkthroughs/set{NN}_{A,B}_data.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Repo imports are cheap (no optional deps).
from tools.evaluate_hybrid_pipeline import _load_processing_image  # noqa: E402
from tools.extract_color_samples import (  # noqa: E402
    discover_additional_tasks,
    load_corpus_tasks,
)
from tools.propose_geometry_labels import (  # noqa: E402
    _fit_hexagon_to_hull,
    _get_rembg_session,
    _hull_from_mask,
)

# `rembg` (and its torch backend) is heavy. Import inside the function
# that needs it so `--help` works in a clean venv without the optional
# research-tooling dependency installed.
def _import_rembg_remove():
    try:
        from rembg import remove  # type: ignore
        return remove
    except ImportError as e:
        raise ImportError(
            "This script requires the optional 'rembg' dependency. "
            "Install with: .venv/bin/pip install rembg"
        ) from e


CORPUS_MANIFEST = REPO_ROOT / "tests/fixtures/corpus_manifest.json"
PANEL_HEADER_HEIGHT = 36
DISPLAY_MAX = 480
DEFAULT_OUT = REPO_ROOT / "runs/hex_fitter_walkthroughs"


def _visvalingam_simplify(points: List[Tuple[float, float]],
                          target: int = 6) -> List[Tuple[float, float]]:
    """Iteratively remove the vertex whose removal least changes the
    polygon shape, until `target` remain."""
    pts = list(points)
    while len(pts) > target:
        n = len(pts)
        min_area, min_idx = float("inf"), 0
        for i in range(n):
            a = pts[(i - 1) % n]
            b = pts[i]
            c = pts[(i + 1) % n]
            area = abs((b[0] - a[0]) * (c[1] - a[1])
                       - (c[0] - a[0]) * (b[1] - a[1]))
            if area < min_area:
                min_area, min_idx = area, i
        pts.pop(min_idx)
    return pts


def _hexagon_min_edge(hexagon: List[Tuple[float, float]]) -> float:
    return min(
        math.hypot(
            hexagon[i][0] - hexagon[(i + 1) % 6][0],
            hexagon[i][1] - hexagon[(i + 1) % 6][1],
        )
        for i in range(6)
    )


def _compute_walkthrough_data(image_path: Path) -> Dict[str, Any]:
    """Run the full algorithm and capture all intermediate data needed
    for both visualization and structured analysis."""
    remove = _import_rembg_remove()
    image, _ = _load_processing_image(image_path)
    w, h = image.size

    rgba = remove(image, session=_get_rembg_session("u2net"))
    alpha = np.array(rgba.split()[-1], dtype=np.uint8)
    mask = alpha > 128

    hull = _hull_from_mask(mask)
    if not hull or len(hull) < 6:
        return {"error": f"hull has only {len(hull) if hull else 0} vertices"}

    cx = sum(p[0] for p in hull) / len(hull)
    cy = sum(p[1] for p in hull) / len(hull)
    anchor_idx = min(range(len(hull)), key=lambda i: hull[i][1])
    anchor = hull[anchor_idx]

    def angle_image(p):
        a = math.atan2(p[0] - cx, -(p[1] - cy))
        return a + 2 * math.pi if a < 0 else a

    anchor_angle = angle_image(anchor)
    sectors: List[List[Tuple[float, Tuple[float, float]]]] = [[] for _ in range(6)]
    for p in hull:
        rel = (angle_image(p) - anchor_angle) % (2 * math.pi)
        s = int(rel / (math.pi / 3)) % 6
        dist = math.hypot(p[0] - cx, p[1] - cy)
        sectors[s].append((dist, tuple(p)))

    winners: List[Optional[Tuple[float, float]]] = []
    sector_data = []
    for s_idx in range(6):
        if not sectors[s_idx]:
            winners.append(None)
            sector_data.append({"count": 0, "winner": None, "dist": None})
            continue
        dist, best = max(sectors[s_idx], key=lambda dp: dp[0])
        winners.append(best)
        sector_data.append({
            "count": len(sectors[s_idx]),
            "winner": [float(best[0]), float(best[1])],
            "dist": float(dist),
        })

    hexagon = _fit_hexagon_to_hull(hull)
    hexagon_min_edge = _hexagon_min_edge(hexagon) if hexagon else None
    vw_hexagon = _visvalingam_simplify(hull, 6)
    vw_min_edge = _hexagon_min_edge(vw_hexagon)

    # Mine collapsed-pair info
    collapsed_pairs = []
    if hexagon:
        for i in range(6):
            edge_len = math.hypot(
                hexagon[i][0] - hexagon[(i + 1) % 6][0],
                hexagon[i][1] - hexagon[(i + 1) % 6][1],
            )
            if edge_len < 20:
                collapsed_pairs.append({
                    "i": i,
                    "j": (i + 1) % 6,
                    "edge_len_px": round(edge_len, 1),
                })

    return {
        "image_path": str(image_path),
        "image_size": [w, h],
        "hull_vertex_count": len(hull),
        "hull": [[float(p[0]), float(p[1])] for p in hull],
        "centroid": [float(cx), float(cy)],
        "anchor": [float(anchor[0]), float(anchor[1])],
        "anchor_angle_rad": float(anchor_angle),
        "sectors": sector_data,
        "winners": [None if w is None else [float(w[0]), float(w[1])]
                    for w in winners],
        "hexagon": [[float(v[0]), float(v[1])] for v in hexagon] if hexagon else None,
        "hexagon_min_edge_px": (
            round(hexagon_min_edge, 1) if hexagon_min_edge is not None else None
        ),
        "is_degenerate": (
            hexagon_min_edge is not None and hexagon_min_edge < 20.0
        ),
        "collapsed_pairs": collapsed_pairs,
        "visvalingam": [[float(v[0]), float(v[1])] for v in vw_hexagon],
        "visvalingam_min_edge_px": round(vw_min_edge, 1),
    }


def _make_panel(title: str, base_image: Image.Image,
                disp_w: int, disp_h: int) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = base_image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (disp_w, disp_h + PANEL_HEADER_HEIGHT), (245, 245, 245))
    head = Image.new("RGB", (disp_w, PANEL_HEADER_HEIGHT), (40, 40, 40))
    ImageDraw.Draw(head).text((10, 10), title, fill=(255, 255, 255))
    out.paste(head, (0, 0))
    out.paste(img, (0, PANEL_HEADER_HEIGHT))
    return out, ImageDraw.Draw(out, "RGBA")


def render_walkthrough(image_path: Path, data: Dict[str, Any],
                       out_png: Path, *, label: str) -> None:
    """Render the 7-panel walkthrough image."""
    image, _ = _load_processing_image(image_path)
    w, h = image.size
    scale = DISPLAY_MAX / max(w, h)
    disp_w, disp_h = int(w * scale), int(h * scale)

    def to_disp(p):
        return (int(p[0] * scale), int(p[1] * scale))

    def offset_y(p):
        return (p[0], p[1] + PANEL_HEADER_HEIGHT)

    remove = _import_rembg_remove()
    rgba = remove(image, session=_get_rembg_session("u2net"))
    alpha = np.array(rgba.split()[-1], dtype=np.uint8)
    mask = alpha > 128

    hull = [tuple(p) for p in data["hull"]]
    cx, cy = data["centroid"]
    anchor = tuple(data["anchor"])
    anchor_angle = data["anchor_angle_rad"]
    hexagon = data["hexagon"]
    winners = data["winners"]
    vw_hexagon = data["visvalingam"]

    # Panel 1: Original
    p1, _ = _make_panel("1. Original photo", image, disp_w, disp_h)

    # Panel 2: Rembg mask
    mo = image.copy().convert("RGBA")
    tint = np.zeros((h, w, 4), dtype=np.uint8)
    tint[mask] = (50, 200, 50, 100)
    mo = Image.alpha_composite(mo, Image.fromarray(tint, mode="RGBA"))
    p2, _ = _make_panel("2. Rembg silhouette (green)", mo.convert("RGB"),
                        disp_w, disp_h)

    # Panel 3: Convex hull
    p3, d3 = _make_panel(f"3. Convex hull ({len(hull)} vertices)",
                         image, disp_w, disp_h)
    for i in range(len(hull)):
        a_d = offset_y(to_disp(hull[i]))
        b_d = offset_y(to_disp(hull[(i + 1) % len(hull)]))
        d3.line([a_d, b_d], fill=(50, 100, 255, 220), width=2)
    for p in hull:
        pd = offset_y(to_disp(p))
        d3.ellipse((pd[0] - 3, pd[1] - 3, pd[0] + 3, pd[1] + 3),
                   fill=(50, 100, 255, 255))

    # Panel 4: Centroid + anchor + sectors
    p4, d4 = _make_panel("4. Centroid + anchor + 6 sectors",
                         image, disp_w, disp_h)
    cd = offset_y(to_disp((cx, cy)))
    d4.ellipse((cd[0] - 6, cd[1] - 6, cd[0] + 6, cd[1] + 6),
               fill=(255, 220, 0, 255), outline=(0, 0, 0, 255))
    d4.text((cd[0] + 8, cd[1] - 6), "centroid", fill=(255, 220, 0, 255))
    ad = offset_y(to_disp(anchor))
    d4.ellipse((ad[0] - 7, ad[1] - 7, ad[0] + 7, ad[1] + 7),
               fill=(255, 50, 50, 255), outline=(0, 0, 0, 255))
    d4.text((ad[0] + 9, ad[1] - 7), "anchor", fill=(255, 50, 50, 255))
    radius = max(disp_w, disp_h) * 0.7
    for s_idx in range(7):
        ray_angle = anchor_angle + s_idx * (math.pi / 3)
        dx = math.sin(ray_angle)
        dy = -math.cos(ray_angle)
        end_img = (cx + radius / scale * dx, cy + radius / scale * dy)
        end_d = offset_y(to_disp(end_img))
        d4.line([cd, end_d], fill=(100, 100, 100, 200), width=1)
        if s_idx < 6:
            mid_angle = anchor_angle + (s_idx + 0.5) * (math.pi / 3)
            mid_dx = math.sin(mid_angle)
            mid_dy = -math.cos(mid_angle)
            label_img = (cx + (radius * 0.45) / scale * mid_dx,
                         cy + (radius * 0.45) / scale * mid_dy)
            label_d = offset_y(to_disp(label_img))
            d4.text((label_d[0] - 6, label_d[1] - 8), f"s{s_idx}",
                    fill=(100, 100, 100, 255))

    # Panel 5: Per-sector winners
    p5, d5 = _make_panel("5. Sector winners (farthest in sector)",
                         image, disp_w, disp_h)
    for i in range(len(hull)):
        a_d = offset_y(to_disp(hull[i]))
        b_d = offset_y(to_disp(hull[(i + 1) % len(hull)]))
        d5.line([a_d, b_d], fill=(180, 180, 200, 150), width=1)
    for s_idx in range(6):
        ray_angle = anchor_angle + s_idx * (math.pi / 3)
        dx = math.sin(ray_angle); dy = -math.cos(ray_angle)
        end_img = (cx + (radius * 0.9) / scale * dx,
                   cy + (radius * 0.9) / scale * dy)
        end_d = offset_y(to_disp(end_img))
        d5.line([cd, end_d], fill=(220, 220, 220, 180), width=1)
    sector_colors = [
        (255, 60, 60), (255, 165, 0), (255, 220, 0),
        (50, 200, 50), (50, 100, 255), (180, 80, 255),
    ]
    for s_idx, winner in enumerate(winners):
        if winner is None:
            continue
        color = sector_colors[s_idx]
        wd = offset_y(to_disp(tuple(winner)))
        d5.ellipse((wd[0] - 8, wd[1] - 8, wd[0] + 8, wd[1] + 8),
                   fill=color + (255,), outline=(0, 0, 0, 255), width=1)
        d5.text((wd[0] + 10, wd[1] - 7), f"s{s_idx}", fill=color + (255,))

    # Panel 6: Resulting hexagon
    me_label = (f"min_edge={data['hexagon_min_edge_px']:.0f} px"
                if data["hexagon_min_edge_px"] is not None else "no hexagon")
    degen_tag = " (DEGENERATE)" if data["is_degenerate"] else ""
    p6, d6 = _make_panel(f"6. Hexagon: {me_label}{degen_tag}",
                         image, disp_w, disp_h)
    if hexagon:
        for i in range(6):
            a_d = offset_y(to_disp(tuple(hexagon[i])))
            b_d = offset_y(to_disp(tuple(hexagon[(i + 1) % 6])))
            edge_len = math.hypot(
                hexagon[i][0] - hexagon[(i + 1) % 6][0],
                hexagon[i][1] - hexagon[(i + 1) % 6][1],
            )
            color = (255, 0, 0, 255) if edge_len < 20 else (50, 200, 50, 255)
            d6.line([a_d, b_d], fill=color, width=3)
        for i, v in enumerate(hexagon):
            vd = offset_y(to_disp(tuple(v)))
            d6.ellipse((vd[0] - 6, vd[1] - 6, vd[0] + 6, vd[1] + 6),
                       fill=(255, 220, 0, 255), outline=(0, 0, 0, 255))
            d6.text((vd[0] + 8, vd[1] - 7), f"h{i}", fill=(0, 0, 0, 255))
    if data["collapsed_pairs"]:
        cps = ", ".join(
            f"h{c['i']}-h{c['j']}={c['edge_len_px']}px"
            for c in data["collapsed_pairs"]
        )
        d6.rectangle((4, p6.height - 26, p6.width - 4, p6.height - 4),
                     fill=(0, 0, 0, 200))
        d6.text((10, p6.height - 22), f"Collapsed: {cps}",
                fill=(255, 100, 100, 255))

    # Panel 7: Visvalingam-Whyatt
    p7, d7 = _make_panel(
        f"7. Visvalingam alt: min_edge={data['visvalingam_min_edge_px']:.0f} px",
        image, disp_w, disp_h,
    )
    for i in range(len(hull)):
        a_d = offset_y(to_disp(hull[i]))
        b_d = offset_y(to_disp(hull[(i + 1) % len(hull)]))
        d7.line([a_d, b_d], fill=(180, 180, 200, 100), width=1)
    for i in range(6):
        a_d = offset_y(to_disp(tuple(vw_hexagon[i])))
        b_d = offset_y(to_disp(tuple(vw_hexagon[(i + 1) % 6])))
        d7.line([a_d, b_d], fill=(50, 200, 50, 255), width=3)
    for v in vw_hexagon:
        vd = offset_y(to_disp(tuple(v)))
        d7.ellipse((vd[0] - 6, vd[1] - 6, vd[0] + 6, vd[1] + 6),
                   fill=(255, 220, 0, 255), outline=(0, 0, 0, 255))

    # Compose final image: 4 cols × 2 rows + title
    panel_w, panel_h = p1.size
    cols, rows = 4, 2
    canvas = Image.new("RGB", (panel_w * cols, panel_h * rows + 60),
                       (250, 250, 250))
    title_strip = Image.new("RGB", (panel_w * cols, 60), (20, 20, 20))
    td = ImageDraw.Draw(title_strip)
    td.text((20, 12), f"_fit_hexagon_to_hull walkthrough — {label}",
            fill=(255, 255, 255))
    me = data["hexagon_min_edge_px"]
    td.text((20, 36),
            f"hull={data['hull_vertex_count']} vertices, "
            f"hexagon min_edge={me} px"
            f"{', DEGENERATE' if data['is_degenerate'] else ''}, "
            f"Visvalingam min_edge={data['visvalingam_min_edge_px']} px",
            fill=(200, 200, 200))
    canvas.paste(title_strip, (0, 0))
    for i, p in enumerate((p1, p2, p3, p4, p5, p6, p7)):
        col = i % cols
        row = i // cols
        canvas.paste(p, (col * panel_w, 60 + row * panel_h))
    canvas.save(out_png, quality=88)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", nargs="+", required=True,
                    help="set IDs to render (e.g. 17 21 30 31 44 47 57 58 61)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"output directory (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    tasks = list(load_corpus_tasks(CORPUS_MANIFEST))
    tasks += list(discover_additional_tasks({t.set_id for t in tasks}))
    by_id = {t.set_id: t for t in tasks}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"pairs": []}
    for set_id in args.sets:
        if set_id not in by_id:
            print(f"  skip set {set_id}: not in corpus", file=sys.stderr)
            continue
        t = by_id[set_id]
        for side, path in [("A", t.image_a), ("B", t.image_b)]:
            print(f"  set {set_id} {side}...", file=sys.stderr, flush=True)
            data = _compute_walkthrough_data(path)
            png_path = out_dir / f"set{set_id}_{side}.png"
            data_path = out_dir / f"set{set_id}_{side}_data.json"
            render_walkthrough(path, data, png_path,
                               label=f"Set {set_id} {side}")
            data_path.write_text(json.dumps(data, indent=2) + "\n")
            summary["pairs"].append({
                "set_id": set_id, "side": side,
                "hull_vertex_count": data.get("hull_vertex_count"),
                "hexagon_min_edge_px": data.get("hexagon_min_edge_px"),
                "is_degenerate": data.get("is_degenerate"),
                "collapsed_pairs": [
                    {k: c[k] for k in ("i", "j", "edge_len_px")}
                    for c in data.get("collapsed_pairs", [])
                ],
                "visvalingam_min_edge_px": data.get("visvalingam_min_edge_px"),
                "png": str(png_path),
                "data_json": str(data_path),
            })
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"\nwrote {len(summary['pairs'])} walkthroughs to {out_dir}/",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
