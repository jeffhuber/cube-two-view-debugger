#!/usr/bin/env python3
"""Render a per-row visual contact sheet comparing production's
rectified face_quads against the oracle ground-truth rectified faces.

Diagnostic-only. Given the PR #274 finding that 9/12 oracle rows are
already broken at the `affine_selected` stage with ~zero stage-to-stage
delta (i.e. PnP / mean3 / phase-check / vertex refinement mostly
PRESERVE the wrong affine pick rather than introduce new error), the
useful visualization isn't "row across stages" — every stage shows the
same broken faces. The useful visualization is "production output vs
ground truth" per row, so a human eye can pattern-match across the 9
broken rows and look for structural cause.

For each of the 12 approved full-corner ground-truth rows this tool:

  1. Runs the production pipeline (rembg + bezel + global_cube_model)
     to get the production fit's 3 face_quads.
  2. Computes the oracle face_quads from the truth fixture via
     `FACE_DEFS_BY_SIDE`.
  3. Rectifies both quad sets (production + oracle) into flat 300×300
     face images via the existing `rectify_faces.py` helper.
  4. Composites a per-row contact-sheet panel showing source image
     thumbnail + 3 production rectified faces + 3 oracle rectified
     faces, annotated with axis misfit + axis_state bucket.

Output:

  /tmp/production_vs_oracle_contact_sheet/
    by_row/{key}.png        per-row panels (one image, side-by-side)
    gallery.html            sortable HTML gallery (all 12 rows)
    index.json              per-row metadata (axis misfit, bucket)

CLI:

  python tools/render_production_vs_oracle_contact_sheet.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.corner_conventions import FACE_DEFS_BY_SIDE  # noqa: E402
from tools.diagnose_pipeline_phase_parity import _processing_image  # noqa: E402
from tools.global_cube_model import fit_global_cube_model  # noqa: E402
from tools.interior_bezel_detection import detect_interior_bezel_lines  # noqa: E402
from tools.measure_axis_correctness import (  # noqa: E402
    _ground_truth_axes,
    _match_axes_to_ground_truth,
)
from tools.rectify_faces import rectify_face  # noqa: E402


DEFAULT_TRUTH = REPO_ROOT / "tests" / "fixtures" / "full_corner_ground_truth.json"
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_OUT_DIR = Path("/tmp/production_vs_oracle_contact_sheet")
DEFAULT_MAX_IMAGE_DIM = 1600
FACE_SIZE_PX = 200
SOURCE_THUMB_PX = 320


# Axis-state buckets per Codex's PR #274 convention (≤30° usable,
# ≥150° broken, otherwise marginal). Reused here so the panel labels
# line up with the stage-transition report.
def _axis_state(total_misfit_deg: float) -> str:
    if total_misfit_deg <= 30.0:
        return "usable"
    if total_misfit_deg >= 150.0:
        return "broken"
    return "marginal"


def _oracle_face_quads(
    truth_row: Dict[str, Any], side: str, scale: float,
) -> Dict[str, List[Tuple[float, float]]]:
    """For a given truth row + side, return {slot: [4 (x, y) corners]}
    for upper/right/front. Coordinates are in processing-resolution
    pixels (full-res corners scaled by `scale`)."""
    defs = FACE_DEFS_BY_SIDE[side]
    out: Dict[str, List[Tuple[float, float]]] = {}
    for slot, corner_names in defs.items():
        quad = []
        for cname in corner_names:
            cx, cy = truth_row[cname]
            quad.append((float(cx) * scale, float(cy) * scale))
        out[slot] = quad
    return out


def _draw_overlay(
    base: Image.Image,
    production_quads: Sequence[Sequence[Tuple[float, float]]],
    oracle_quads: Sequence[Sequence[Tuple[float, float]]],
) -> Image.Image:
    """Source-image thumbnail with production face_quads (red) and
    oracle face_quads (green) overlaid. Helps the eye see how the two
    differ before looking at the rectified outputs."""
    img = base.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    for quad in production_quads:
        coords = [(int(p[0]), int(p[1])) for p in quad] + [
            (int(quad[0][0]), int(quad[0][1])),
        ]
        draw.line(coords, fill=(220, 50, 50, 230), width=3)
    for quad in oracle_quads:
        coords = [(int(p[0]), int(p[1])) for p in quad] + [
            (int(quad[0][0]), int(quad[0][1])),
        ]
        draw.line(coords, fill=(80, 220, 80, 230), width=3)
    return img


def _composite_panel(
    key: str,
    side: str,
    source: Image.Image,
    overlay: Image.Image,
    production_faces: Dict[str, Image.Image],
    oracle_faces: Dict[str, Image.Image],
    axis_misfit: Optional[float],
    axis_state: str,
    error: Optional[str] = None,
) -> Image.Image:
    """Compose source thumbnail + overlay thumbnail + 3 production
    rectified + 3 oracle rectified into one wide panel for the row."""
    slot_order = ("upper", "right", "front")
    # Source thumb + overlay sit on the left at SOURCE_THUMB_PX width.
    src_thumb = source.copy()
    src_thumb.thumbnail((SOURCE_THUMB_PX, SOURCE_THUMB_PX))
    overlay_thumb = overlay.copy()
    overlay_thumb.thumbnail((SOURCE_THUMB_PX, SOURCE_THUMB_PX))
    margin = 16
    header_h = 56
    label_h = 22
    face_block_h = FACE_SIZE_PX + label_h
    body_h = max(
        src_thumb.height + overlay_thumb.height + margin,
        2 * face_block_h + margin,
    )
    total_w = (
        margin + SOURCE_THUMB_PX + margin
        + 3 * (FACE_SIZE_PX + margin)
    )
    total_h = header_h + margin + body_h + margin

    panel = Image.new("RGB", (total_w, total_h), (32, 32, 38))
    draw = ImageDraw.Draw(panel)
    # Header line.
    state_color = {
        "usable": (80, 220, 80),
        "marginal": (220, 200, 80),
        "broken": (220, 80, 80),
    }.get(axis_state, (200, 200, 200))
    header = f"{key} (side {side}) — axis misfit {axis_misfit:.1f}°  [{axis_state.upper()}]" \
        if axis_misfit is not None else f"{key} (side {side}) — {error or 'no fit'}"
    draw.text((margin, margin), header, fill=state_color)
    # Source on the left, stacked.
    panel.paste(src_thumb, (margin, header_h + margin))
    panel.paste(overlay_thumb, (margin, header_h + margin + src_thumb.height + margin // 2))
    draw.text(
        (margin, header_h + margin - 14),
        "source / overlay (red=production, green=oracle)",
        fill=(200, 200, 200),
    )
    # Faces: 2 rows of 3 to the right.
    face_x0 = margin + SOURCE_THUMB_PX + margin
    prod_y = header_h + margin + label_h
    oracle_y = prod_y + FACE_SIZE_PX + margin + label_h
    draw.text(
        (face_x0, prod_y - label_h + 4),
        "Production rectified faces:",
        fill=(220, 120, 120),
    )
    draw.text(
        (face_x0, oracle_y - label_h + 4),
        "Oracle rectified faces:",
        fill=(120, 220, 120),
    )
    for i, slot in enumerate(slot_order):
        x = face_x0 + i * (FACE_SIZE_PX + margin)
        pf = production_faces.get(slot)
        if pf is not None:
            pf_r = pf.resize((FACE_SIZE_PX, FACE_SIZE_PX), Image.Resampling.BICUBIC)
            panel.paste(pf_r, (x, prod_y))
        of = oracle_faces.get(slot)
        if of is not None:
            of_r = of.resize((FACE_SIZE_PX, FACE_SIZE_PX), Image.Resampling.BICUBIC)
            panel.paste(of_r, (x, oracle_y))
        draw.text((x, prod_y + FACE_SIZE_PX + 2), slot, fill=(200, 200, 200))
    return panel


def process_row(
    sess: Any, key: str, image_path: Path, truth_row: Dict[str, Any],
    max_image_dim: int, out_dir: Path,
) -> Dict[str, Any]:
    from rembg import remove  # noqa: E402
    side = key.rsplit("_", 1)[-1]
    record: Dict[str, Any] = {"key": key, "side": side, "status": "pending"}
    try:
        image, scale = _processing_image(image_path, max_image_dim)
        rgb = np.array(image)
        # Use the PRODUCTION rembg path: full `remove(...)` → RGBA → split
        # alpha channel → mask. This applies rembg's alpha-matting refinement
        # step, which `only_mask=True` skips. Codex caught (2026-05-24) that
        # several diagnostics (`measure_axis_correctness.py`,
        # `diagnose_center_color_phase_gate.py`, original `#274` stage trace)
        # used `only_mask=True` and so produced subtly cruder hexagons than
        # production sees via `propose_geometry_labels.py` and
        # `rubik_recognizer/image_pipeline.py`. For "production vs oracle"
        # comparison to be honest, the production hexagon here must come
        # through the same rembg path production uses.
        rgba = remove(image, session=sess)
        alpha = np.array(rgba.split()[-1], dtype=np.uint8)
        mask = alpha > 128
        det = detect_interior_bezel_lines(rgb, mask)
        if det.cube_center is None:
            record.update({"status": "fit_failed", "error": "bezel detection produced no cube_center"})
            return record
        # Production model — fit_global_cube_model returns the chosen-perm model,
        # which we know per PR #274's stage-transition trace is essentially the
        # affine_selected model on these rows (zero stage-to-stage delta on 9/12).
        model = fit_global_cube_model(det, rgb, mask, apply_phase_correction=True)
        if model is None:
            record.update({"status": "fit_failed", "error": "fit returned None"})
            return record
        production_quads = {
            slot: list(model.face_quads.get(_slot_to_face_key(slot), []))
            for slot in ("upper", "right", "front")
        }
        # Drop any that came back empty.
        production_quads = {s: q for s, q in production_quads.items() if len(q) == 4}
        oracle_quads = _oracle_face_quads(truth_row, side, scale)
        # Rectify both.
        production_faces = {
            slot: rectify_face(image, quad, output_size=FACE_SIZE_PX)
            for slot, quad in production_quads.items()
        }
        oracle_faces = {
            slot: rectify_face(image, quad, output_size=FACE_SIZE_PX)
            for slot, quad in oracle_quads.items()
        }
        # Axis misfit for the header (use the diagnostic's standard).
        _gt_vertex, gt_axes = _ground_truth_axes(truth_row, side, scale)
        predicted_axes = [model.axis_x_2d, model.axis_y_2d, model.axis_z_2d]
        axis_match = _match_axes_to_ground_truth(predicted_axes, gt_axes)
        misfit = float(axis_match["total_misfit_deg"])
        state = _axis_state(misfit)
        overlay = _draw_overlay(
            image, list(production_quads.values()), list(oracle_quads.values()),
        )
        panel = _composite_panel(
            key=key, side=side, source=image, overlay=overlay,
            production_faces=production_faces, oracle_faces=oracle_faces,
            axis_misfit=misfit, axis_state=state,
        )
        out_path = out_dir / "by_row" / f"{key}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(out_path)
        record.update({
            "status": "rendered",
            "axis_misfit_deg": round(misfit, 1),
            "axis_state": state,
            "panel_path": str(out_path.relative_to(out_dir)),
        })
    except Exception as exc:  # noqa: BLE001
        record.update({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
    return record


def _slot_to_face_key(slot: str) -> str:
    """`tools/global_cube_model.GlobalCubeModel.face_quads` keys are
    `face_yz`, `face_xz`, `face_xy` (cube-local axis-pair names) rather
    than the rectification-tool's `upper`/`right`/`front` slot names.
    The mapping per the GlobalCubeModel docstring is:
      face_yz <-> upper, face_xz <-> right, face_xy <-> front.
    """
    return {
        "upper": "face_yz",
        "right": "face_xz",
        "front": "face_xy",
    }[slot]


def render_gallery_html(records: Sequence[Dict[str, Any]], out_dir: Path) -> None:
    rendered = [r for r in records if r.get("status") == "rendered"]
    rendered.sort(key=lambda r: (r["axis_state"] != "broken", r["key"]))
    lines: List[str] = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Production vs Oracle contact sheet</title>",
        "<style>",
        "body{background:#1c1c20;color:#ddd;font-family:system-ui,sans-serif;margin:24px}",
        "h1{color:#fff;margin:0 0 8px 0}",
        ".intro{color:#aaa;margin-bottom:24px;max-width:920px;line-height:1.4}",
        ".row{margin-bottom:32px;padding:8px;border:1px solid #333;border-radius:6px}",
        ".row.broken{border-left:6px solid #d85050}",
        ".row.marginal{border-left:6px solid #d8c850}",
        ".row.usable{border-left:6px solid #50d850}",
        ".meta{color:#bbb;margin-bottom:8px;font-size:13px}",
        "img{max-width:100%;height:auto;border-radius:4px}",
        ".errors{color:#d88;margin-top:32px}",
        "</style></head><body>",
        "<h1>Production vs Oracle rectified faces — 12-row contact sheet</h1>",
        "<p class='intro'>One panel per approved full-corner ground-truth row. ",
        "Left column shows the source image (top) and the same image with face_quads overlaid (bottom) — ",
        "<span style='color:#dc5050'>red = production</span>, ",
        "<span style='color:#50dc50'>green = oracle</span>. ",
        "Top-right row: 3 faces rectified from production's chosen face_quads. ",
        "Bottom-right row: 3 faces rectified from oracle ground-truth corners — the canonical correct output. ",
        "Per Codex PR #274 stage-transition finding: 9/12 rows are broken at the very first ",
        "(<code>affine_selected</code>) stage and stay broken with zero stage-to-stage delta, ",
        "so this single snapshot represents production behavior across the whole pipeline. ",
        "Rows sorted broken-first.</p>",
    ]
    for r in rendered:
        cls = r["axis_state"]
        lines.append(
            f"<div class='row {cls}'>"
            f"<div class='meta'><b>{r['key']}</b> "
            f"(side {r['side']}) — axis misfit {r['axis_misfit_deg']}° "
            f"[{cls.upper()}]</div>"
            f"<img src='{r['panel_path']}' alt='{r['key']}'/>"
            f"</div>"
        )
    errors = [r for r in records if r.get("status") not in ("rendered",)]
    if errors:
        lines.append("<div class='errors'><h2>Skipped / errored rows</h2><ul>")
        for r in errors:
            lines.append(
                f"<li><b>{r.get('key')}</b>: {r.get('status')} — "
                f"{r.get('error', '?')}</li>"
            )
        lines.append("</ul></div>")
    lines.append("</body></html>")
    (out_dir / "gallery.html").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-image-dim", type=int, default=DEFAULT_MAX_IMAGE_DIM)
    args = ap.parse_args(list(argv) if argv is not None else None)

    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    set_index = {str(p["setId"]): p for p in manifest.get("pairs", [])}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "by_row").mkdir(exist_ok=True)

    from rembg import new_session  # noqa: E402
    sess = new_session()

    records: List[Dict[str, Any]] = []
    for key in sorted(truth):
        row = truth[key]
        if not row.get("approved"):
            continue
        set_id, _, side = key.rpartition("_")
        pair = set_index.get(set_id)
        if pair is None:
            records.append({"key": key, "status": "skipped",
                            "error": f"set {set_id} not in manifest"})
            continue
        img_path_str = pair.get(f"image{side}Path")
        if not img_path_str:
            records.append({"key": key, "status": "skipped",
                            "error": "no image path"})
            continue
        img_path = Path(img_path_str)
        if not img_path.exists():
            records.append({"key": key, "status": "skipped",
                            "error": f"image not found: {img_path}"})
            continue
        print(f"[{len([r for r in records if r.get('status')=='rendered'])+1}] {key}...",
              flush=True)
        rec = process_row(sess, key, img_path, row, args.max_image_dim, args.out_dir)
        records.append(rec)

    render_gallery_html(records, args.out_dir)
    (args.out_dir / "index.json").write_text(
        json.dumps({"schema": "production_vs_oracle_contact_sheet_v1",
                    "per_row": records}, indent=2),
        encoding="utf-8",
    )
    n_rendered = sum(1 for r in records if r.get("status") == "rendered")
    print(
        f"\nWrote {n_rendered} per-row panels + gallery.html to {args.out_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
