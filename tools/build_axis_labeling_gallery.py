#!/usr/bin/env python
"""Build a self-contained interactive HTML gallery for labeling cube vertex
+ 3 outgoing axis endpoints across the labeled corpus.

The 4-points-per-photo schema (vertex + 3 "near" hexagon corners) is enough
to fully determine the projected cube model (8 DOF in 2D). Vertex alone
only gave us 2 DOF and missed axis orientation — which the 2026-05 vertex
correlation experiments showed matters more than vertex precision for
downstream sticker sampling.

Workflow:
1. Run this tool to produce a static HTML + per-photo cropped PNGs in --out.
2. Open the gallery in a browser. Each photo shows 4 draggable markers
   prefilled at the global cube model's current best guess.
3. Drag each marker to the correct on-photo position. Click "Approve"
   per photo when done.
4. Click "Download JSON" to save the labeled set.
5. The downstream scorer (tools/evaluate_axis_ground_truth.py) consumes
   this JSON.

The prefill from the on-main global cube model means most photos need
only small corrections (vs labeling from scratch), keeping labeling
effort to ~5-10 min per session instead of ~30 min.

Usage:
    .venv/bin/python tools/build_axis_labeling_gallery.py \\
        --out /tmp/axis_labeling_v1

    # Limit to specific sets (useful for testing the tool):
    .venv/bin/python tools/build_axis_labeling_gallery.py \\
        --out /tmp/axis_labeling_v1 --sets 12 14 17 28 44
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_manifests() -> list:
    out = []
    for fname in ("hard_case_manifest.json", "corpus_manifest.json"):
        with (REPO_ROOT / "tests" / "fixtures" / fname).open() as f:
            out.append(json.load(f))
    return out


def _resolve_pair_paths(manifests: list, set_id: str) -> Optional[Tuple[Path, Path]]:
    for manifest in manifests:
        for entry in manifest["pairs"]:
            if entry["setId"] == set_id:
                return Path(entry["imageAPath"]), Path(entry["imageBPath"])
    return None


def _exif_correct(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _compute_rembg_mask(rgb_array: np.ndarray) -> np.ndarray:
    """Cached rembg session shared across all photos."""
    from rembg import new_session, remove  # type: ignore

    global _REMBG_SESSION
    try:
        sess = _REMBG_SESSION
    except NameError:
        sess = new_session("u2net")
        _REMBG_SESSION = sess  # type: ignore
    pil = Image.fromarray(rgb_array)
    rgba = remove(pil, session=sess)
    alpha = np.array(rgba.split()[-1], dtype=np.uint8)
    return alpha > 128


def _global_model_prefill(
    rgb: np.ndarray,
    mask: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Run the on-main global cube model and extract (vertex, 3 near corners)
    in original-image coordinates. Returns None on failure."""
    try:
        from tools.global_cube_model import fit_global_cube_model
        from tools.interior_bezel_detection import detect_interior_bezel_lines
    except ImportError as e:
        print(f"global model import failed: {e}", file=sys.stderr)
        return None
    try:
        detection = detect_interior_bezel_lines(rgb, mask)
        model = fit_global_cube_model(detection, rgb, mask, optimize=True)
    except Exception as e:
        print(f"global model fit failed: {e}", file=sys.stderr)
        return None
    if model is None:
        return None
    corners = model.visible_corners
    if "front" not in corners or any(k not in corners for k in ("h_x", "h_y", "h_z")):
        return None
    return {
        "vertex": [round(corners["front"][0], 1), round(corners["front"][1], 1)],
        "near_x": [round(corners["h_x"][0], 1), round(corners["h_x"][1], 1)],
        "near_y": [round(corners["h_y"][0], 1), round(corners["h_y"][1], 1)],
        "near_z": [round(corners["h_z"][0], 1), round(corners["h_z"][1], 1)],
    }


def _crop_around_points(
    img_size: Tuple[int, int],
    points: List[Tuple[float, float]],
    margin: int = 100,
    max_dim: int = 1000,
) -> Tuple[Tuple[int, int, int, int], float, Tuple[int, int]]:
    """Compute crop box around the given points, scale to fit display max_dim."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0 = max(0, int(min(xs)) - margin)
    y0 = max(0, int(min(ys)) - margin)
    x1 = min(img_size[0], int(max(xs)) + margin)
    y1 = min(img_size[1], int(max(ys)) + margin)
    cw, ch = x1 - x0, y1 - y0
    scale = min(1.0, max_dim / max(cw, ch))
    nw, nh = int(cw * scale), int(ch * scale)
    return (x0, y0, x1, y1), scale, (nw, nh)


def _build_case_data(
    set_id: str,
    side: str,
    img_path: Path,
    out_dir: Path,
    skip_rembg: bool,
) -> Optional[Dict[str, Any]]:
    img = _exif_correct(img_path)
    rgb = np.asarray(img, dtype=np.uint8)

    prefill: Optional[Dict[str, Any]] = None
    if not skip_rembg:
        try:
            mask = _compute_rembg_mask(rgb)
            prefill = _global_model_prefill(rgb, mask)
        except Exception as e:
            print(f"[set {set_id} {side}] prefill failed: {e}", file=sys.stderr)

    if prefill is None:
        # Center of image fallback so the labeling tool still works without rembg
        cx, cy = img.size[0] / 2, img.size[1] / 2
        prefill = {
            "vertex": [cx, cy],
            "near_x": [cx + 200, cy],
            "near_y": [cx - 100, cy - 200],
            "near_z": [cx - 100, cy + 200],
        }

    pts: List[Tuple[float, float]] = [
        tuple(prefill["vertex"]),
        tuple(prefill["near_x"]),
        tuple(prefill["near_y"]),
        tuple(prefill["near_z"]),
    ]
    crop_box, scale, (nw, nh) = _crop_around_points(img.size, pts)

    cropped = img.crop(crop_box).resize((nw, nh))
    out_png = out_dir / f"set_{set_id}_{side}.png"
    cropped.save(out_png, optimize=True)

    return {
        "key": f"{set_id}_{side}",
        "set_id": set_id,
        "side": side,
        "image_file": out_png.name,
        "image_path_full": str(img_path),
        "crop_x0": crop_box[0],
        "crop_y0": crop_box[1],
        "scale": scale,
        "display_w": nw,
        "display_h": nh,
        "prefill": prefill,
        "prefill_source": "global_model" if not skip_rembg else "image_center",
    }


HTML_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Cube axis labeling gallery</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 0; background: #1a1a1a; color: #eee; }
  .toolbar { position: sticky; top: 0; background: #2a2a2a; padding: 10px 20px;
             border-bottom: 1px solid #444; z-index: 100; display: flex; gap: 16px; align-items: center; }
  .toolbar button { background: #4a4a4a; color: white; border: none; padding: 8px 16px;
                    border-radius: 4px; cursor: pointer; font-size: 14px; }
  .toolbar button:hover { background: #5a5a5a; }
  .toolbar .stats { margin-left: auto; font-size: 13px; color: #aaa; }
  .case { margin: 20px; padding: 16px; background: #2a2a2a; border-radius: 8px; border: 2px solid #444; }
  .case.approved { border-color: #4a8; }
  .case h3 { margin: 0 0 8px 0; color: #fff; }
  .case .meta { color: #aaa; font-size: 12px; margin-bottom: 8px; }
  .case .canvas-wrap { position: relative; display: inline-block; }
  .case canvas { display: block; cursor: crosshair; border: 1px solid #555; touch-action: none; }
  .case .controls { margin-top: 10px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .case .controls button { padding: 6px 14px; border-radius: 4px; border: 1px solid #555; cursor: pointer; }
  .btn-approve { background: #2a5a3a; color: white; }
  .btn-approve.active { background: #4a8; color: white; }
  .btn-reset { background: #5a5a2a; color: white; }
  .legend { display: inline-flex; gap: 8px; font-size: 12px; align-items: center; }
  .swatch { display: inline-block; width: 12px; height: 12px; border: 1px solid #000; vertical-align: middle; }
  .point-readout { font-family: monospace; font-size: 11px; color: #aaa; margin-left: 12px; }
</style>
</head><body>

<div class="toolbar">
  <button onclick="downloadJSON()">Download JSON</button>
  <button onclick="copyToClipboard()">Copy JSON</button>
  <button onclick="approveAll()">Approve all (use prefills as-is)</button>
  <span class="legend">
    <span><span class="swatch" style="background:#fff"></span> vertex</span>
    <span><span class="swatch" style="background:#f55"></span> near_x</span>
    <span><span class="swatch" style="background:#5a5"></span> near_y</span>
    <span><span class="swatch" style="background:#55f"></span> near_z</span>
  </span>
  <span class="stats" id="stats"></span>
</div>

<h2 style="margin: 20px;">Cube axis labeling — 4 points per photo</h2>
<p style="margin: 0 20px 8px;">For each photo, position the 4 colored markers:
  <b>WHITE</b> at the trihedral vertex (where 3 cube faces meet),
  <b>RED/GREEN/BLUE</b> at the 3 hexagon corners that are 1 cube-edge away from the vertex
  (where each visible face edge terminates at a far corner of that face).</p>
<p style="margin: 0 20px 8px;">Drag markers to reposition; or click on the photo to move the most-recently-touched marker there.</p>
<p style="margin: 0 20px 16px;">Prefills come from the on-main global cube model. Often correct on well-lit cases; needs nudging on shadowed/hard cases.</p>

<div id="cases"></div>

<script>
const CASES = __CASES_JSON__;
const judgments = {};  // key → { vertex, near_x, near_y, near_z, approved }

function deepCopy(o) { return JSON.parse(JSON.stringify(o)); }

function init() {
  CASES.forEach(c => {
    judgments[c.key] = {
      vertex: c.prefill.vertex.slice(),
      near_x: c.prefill.near_x.slice(),
      near_y: c.prefill.near_y.slice(),
      near_z: c.prefill.near_z.slice(),
      approved: false,
      lastTouched: null,
    };
  });
  render();
}

function origToDisplay(c, pt) {
  return [(pt[0] - c.crop_x0) * c.scale, (pt[1] - c.crop_y0) * c.scale];
}
function displayToOrig(c, dx, dy) {
  return [dx / c.scale + c.crop_x0, dy / c.scale + c.crop_y0];
}

const MARKER_RADIUS = 9;
const POINT_NAMES = ['vertex', 'near_x', 'near_y', 'near_z'];
const POINT_COLORS = {
  vertex: '#ffffff',
  near_x: '#ff5050',
  near_y: '#50aa50',
  near_z: '#5050ff',
};

function pointAt(c, dx, dy) {
  const j = judgments[c.key];
  for (const name of POINT_NAMES) {
    const [px, py] = origToDisplay(c, j[name]);
    const dist = Math.hypot(px - dx, py - dy);
    if (dist <= MARKER_RADIUS + 6) return name;
  }
  return null;
}

function render() {
  const root = document.getElementById('cases');
  root.innerHTML = '';
  CASES.forEach(c => {
    const j = judgments[c.key];
    const div = document.createElement('div');
    div.className = 'case' + (j.approved ? ' approved' : '');
    div.innerHTML = `
      <h3>${c.key} <span style="color:#888;font-weight:normal;font-size:13px">prefill: ${c.prefill_source}</span></h3>
      <div class="meta">${c.image_path_full}</div>
      <canvas id="cv-${c.key}" width="${c.display_w}" height="${c.display_h}"></canvas>
      <div class="controls">
        <button class="btn-approve ${j.approved ? 'active' : ''}" onclick="toggleApprove('${c.key}')">${j.approved ? 'Approved ✓' : 'Approve'}</button>
        <button class="btn-reset" onclick="resetToPrefill('${c.key}')">Reset to prefill</button>
        <span class="point-readout" id="readout-${c.key}"></span>
      </div>
    `;
    root.appendChild(div);
    const cv = document.getElementById('cv-' + c.key);
    const ctx = cv.getContext('2d');
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0);
      drawMarkers(c, ctx);
      updateReadout(c);
    };
    img.src = c.image_file;
    attachHandlers(c, cv);
  });
  updateStats();
}

function drawMarkers(c, ctx) {
  const j = judgments[c.key];
  for (const name of POINT_NAMES) {
    const [px, py] = origToDisplay(c, j[name]);
    ctx.beginPath();
    ctx.arc(px, py, MARKER_RADIUS, 0, 2 * Math.PI);
    ctx.fillStyle = POINT_COLORS[name];
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  // Draw axis lines from vertex
  const [vx, vy] = origToDisplay(c, j.vertex);
  for (const name of ['near_x', 'near_y', 'near_z']) {
    const [nx, ny] = origToDisplay(c, j[name]);
    ctx.beginPath();
    ctx.moveTo(vx, vy);
    ctx.lineTo(nx, ny);
    ctx.strokeStyle = POINT_COLORS[name];
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

function attachHandlers(c, cv) {
  let dragging = null;
  cv.addEventListener('pointerdown', (e) => {
    const rect = cv.getBoundingClientRect();
    const dx = (e.clientX - rect.left) * (cv.width / rect.width);
    const dy = (e.clientY - rect.top) * (cv.height / rect.height);
    const hit = pointAt(c, dx, dy);
    if (hit) {
      dragging = hit;
      cv.setPointerCapture(e.pointerId);
    } else if (judgments[c.key].lastTouched) {
      const orig = displayToOrig(c, dx, dy);
      judgments[c.key][judgments[c.key].lastTouched] = orig;
      redraw(c);
    }
  });
  cv.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const rect = cv.getBoundingClientRect();
    const dx = (e.clientX - rect.left) * (cv.width / rect.width);
    const dy = (e.clientY - rect.top) * (cv.height / rect.height);
    const orig = displayToOrig(c, dx, dy);
    judgments[c.key][dragging] = orig;
    judgments[c.key].lastTouched = dragging;
    redraw(c);
  });
  cv.addEventListener('pointerup', () => { dragging = null; });
  cv.addEventListener('pointercancel', () => { dragging = null; });
}

function redraw(c) {
  const cv = document.getElementById('cv-' + c.key);
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const img = new Image();
  img.onload = () => {
    ctx.drawImage(img, 0, 0);
    drawMarkers(c, ctx);
    updateReadout(c);
  };
  img.src = c.image_file;
}

function updateReadout(c) {
  const el = document.getElementById('readout-' + c.key);
  if (!el) return;
  const j = judgments[c.key];
  const fmt = (p) => `(${p[0].toFixed(0)},${p[1].toFixed(0)})`;
  el.textContent = `vertex=${fmt(j.vertex)} near_x=${fmt(j.near_x)} near_y=${fmt(j.near_y)} near_z=${fmt(j.near_z)}`;
}

function toggleApprove(key) {
  judgments[key].approved = !judgments[key].approved;
  const caseDiv = document.querySelector(`#cv-${key}`).closest('.case');
  caseDiv.classList.toggle('approved', judgments[key].approved);
  const btn = caseDiv.querySelector('.btn-approve');
  btn.classList.toggle('active', judgments[key].approved);
  btn.textContent = judgments[key].approved ? 'Approved ✓' : 'Approve';
  updateStats();
}
function resetToPrefill(key) {
  const c = CASES.find(c => c.key === key);
  judgments[key].vertex = c.prefill.vertex.slice();
  judgments[key].near_x = c.prefill.near_x.slice();
  judgments[key].near_y = c.prefill.near_y.slice();
  judgments[key].near_z = c.prefill.near_z.slice();
  judgments[key].approved = false;
  render();
}
function approveAll() {
  Object.keys(judgments).forEach(k => { judgments[k].approved = true; });
  render();
}
function updateStats() {
  const n = Object.keys(judgments).length;
  const approved = Object.values(judgments).filter(j => j.approved).length;
  document.getElementById('stats').textContent = `approved: ${approved}/${n}`;
}

function buildExport() {
  const out = {};
  CASES.forEach(c => {
    const j = judgments[c.key];
    if (j.approved) {
      out[c.key] = {
        vertex: [Math.round(j.vertex[0]*10)/10, Math.round(j.vertex[1]*10)/10],
        near_x: [Math.round(j.near_x[0]*10)/10, Math.round(j.near_x[1]*10)/10],
        near_y: [Math.round(j.near_y[0]*10)/10, Math.round(j.near_y[1]*10)/10],
        near_z: [Math.round(j.near_z[0]*10)/10, Math.round(j.near_z[1]*10)/10],
        approved: true,
      };
    }
  });
  return out;
}
function downloadJSON() {
  const blob = new Blob([JSON.stringify(buildExport(), null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'gcm_axis_ground_truth.json'; a.click();
}
function copyToClipboard() {
  navigator.clipboard.writeText(JSON.stringify(buildExport(), null, 2));
  alert('Copied to clipboard');
}

init();
</script>
</body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sets", nargs="*", default=None,
                        help="Set IDs to include (default: every labeled set with both photos)")
    parser.add_argument("--skip-rembg", action="store_true",
                        help="Skip global model prefill; use image center fallback (debug only)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifests = _load_manifests()
    all_pairs = [e for m in manifests for e in m["pairs"]]
    seen: set = set()
    unique = []
    for p in all_pairs:
        if p["setId"] in seen:
            continue
        seen.add(p["setId"])
        unique.append(p)

    sets_to_process = args.sets if args.sets else [p["setId"] for p in unique]
    cases: List[Dict[str, Any]] = []
    for set_id in sets_to_process:
        paths = _resolve_pair_paths(manifests, set_id)
        if paths is None:
            print(f"set {set_id}: not in any manifest", file=sys.stderr)
            continue
        path_a, path_b = paths
        for side, p in (("A", path_a), ("B", path_b)):
            if not p.exists():
                print(f"set {set_id} {side}: {p} not found", file=sys.stderr)
                continue
            print(f"[set {set_id} {side}] processing {p.name}", file=sys.stderr)
            data = _build_case_data(set_id, side, p, args.out, args.skip_rembg)
            if data is not None:
                cases.append(data)

    html = HTML_TEMPLATE.replace("__CASES_JSON__", json.dumps(cases))
    out_path = args.out / "gallery.html"
    out_path.write_text(html)
    print(f"Wrote {out_path}", file=sys.stderr)
    print(f"  {len(cases)} cases", file=sys.stderr)
    print(f"  Open: file://{out_path.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
