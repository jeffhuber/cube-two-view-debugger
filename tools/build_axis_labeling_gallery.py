#!/usr/bin/env python
"""Build a legacy interactive HTML gallery for labeling cube vertex
+ 3 outgoing axis endpoints across the labeled corpus.

The 4-points-per-photo schema (vertex + 3 "near" hexagon corners) is enough
to fully determine the projected cube model (8 DOF in 2D). Vertex alone
only gave us 2 DOF and missed axis orientation — which the 2026-05 vertex
correlation experiments showed matters more than vertex precision for
downstream sticker sampling.

This tool is retained for historical reproducibility. New canonical geometry
labels should use `tools/build_full_corner_labeling_gallery.py`, which records
`Va/Vb + 0..5` and derives one-edge vs far triplets side-specifically. Do not
create new canonical `axis_*` labels with this tool (legacy fixtures may
still use the older `near_*` key set — readers accept both; see
``tools/FULL_CORNER_LABELING.md`` "Axis-truth schema convention").

Workflow:
1. Run this tool to produce a static HTML + per-photo full-image PNGs in --out.
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
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

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

    for root in _candidate_corpus_roots(manifests):
        path_a = _find_corpus_side(root, set_id, "A")
        path_b = _find_corpus_side(root, set_id, "B")
        if path_a is not None and path_b is not None:
            return path_a, path_b
    return None


def _candidate_corpus_roots(manifests: list) -> List[Path]:
    roots: List[Path] = []
    for manifest in manifests:
        for entry in manifest.get("pairs", []):
            for field in ("imageAPath", "imageBPath"):
                path = Path(entry[field]).expanduser()
                parent = path.parent
                if parent not in roots:
                    roots.append(parent)
    home_corpus = Path.home() / "cube-corpus"
    if home_corpus not in roots:
        roots.append(home_corpus)
    return roots


def _find_corpus_side(root: Path, set_id: str, side: str) -> Optional[Path]:
    for pattern in (f"Set {set_id} - {side} -*", f"Set {set_id} - {side} *"):
        candidates = sorted(p for p in root.glob(pattern) if p.is_file())
        if candidates:
            return candidates[0]
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
    """Run the legacy global model prefill in original-image coordinates.

    The returned fields are the canonical vertex + 3 `axis_*` schema. Those
    labels are not canonical human one-edge labels; see FULL_CORNER_LABELING.
    """
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
        # axis_x/y/z labels the 3 FAR silhouette corners (the corner
        # along each world axis direction from the vertex; in iso
        # projection this is the two-cube-edge corner of each visible
        # face). See FULL_CORNER_LABELING.md "Axis-truth schema
        # convention". The model's h_x/h_y/h_z output happens to sit
        # at the FAR positions despite the "h_" name (legacy from
        # before the convention was nailed down).
        "axis_x": [round(corners["h_x"][0], 1), round(corners["h_x"][1], 1)],
        "axis_y": [round(corners["h_y"][0], 1), round(corners["h_y"][1], 1)],
        "axis_z": [round(corners["h_z"][0], 1), round(corners["h_z"][1], 1)],
    }


def _full_image_display(
    img_size: Tuple[int, int],
) -> Tuple[Tuple[int, int, int, int], float, Tuple[int, int]]:
    """Use the whole EXIF-corrected image and let CSS fit it to the viewport."""
    width, height = img_size
    return (0, 0, width, height), 1.0, (width, height)


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
            "axis_x": [cx + 200, cy],
            "axis_y": [cx - 100, cy - 200],
            "axis_z": [cx - 100, cy + 200],
        }

    crop_box, scale, (nw, nh) = _full_image_display(img.size)

    cropped = img.crop(crop_box)
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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cube axis labeling gallery</title>
<style>
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; overflow: hidden; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #171717; color: #eee; }
  .app { height: 100vh; min-height: 0; display: flex; flex-direction: column; }
  .toolbar { flex: 0 0 auto; background: #292929; padding: 8px 12px;
             border-bottom: 1px solid #444; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .toolbar button { background: #4a4a4a; color: white; border: none; padding: 8px 16px;
                    border-radius: 4px; cursor: pointer; font-size: 14px; }
  .toolbar button:hover { background: #5a5a5a; }
  .toolbar button:disabled { opacity: .45; cursor: default; }
  .toolbar select { background: #1f1f1f; color: #fff; border: 1px solid #555; border-radius: 4px; padding: 7px 8px; min-width: 150px; }
  .toolbar .stats { margin-left: auto; font-size: 13px; color: #aaa; white-space: nowrap; }
  .intro { flex: 0 0 auto; padding: 8px 12px; border-bottom: 1px solid #333; color: #cfcfcf; font-size: 13px; line-height: 1.3; }
  .intro h2 { display: inline; margin: 0 10px 0 0; color: #fff; font-size: 16px; }
  .intro p { display: inline; margin: 0; }
  .viewer { flex: 1 1 auto; min-height: 0; display: grid; place-items: center; padding: 8px 12px 12px; }
  .case { width: min(100%, 980px); max-height: 100%; min-height: 0; padding: 10px; background: #252525; border-radius: 8px; border: 2px solid #444;
          display: grid; grid-template-rows: auto minmax(0, 1fr) auto; gap: 8px; }
  .case.approved { border-color: #4a8; }
  .case-head { min-width: 0; display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
  .case h3 { margin: 0; color: #fff; font-size: 16px; }
  .case .meta { min-width: 0; color: #aaa; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .case .canvas-wrap { min-height: 0; display: flex; align-items: center; justify-content: center; overflow: visible; }
  .case canvas { display: block; cursor: crosshair; border: 1px solid #555; touch-action: none;
                 width: auto; height: auto; max-width: 100%; max-height: 100%; }
  .case .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .case .controls button { padding: 6px 14px; border-radius: 4px; border: 1px solid #555; cursor: pointer; }
  .btn-approve { background: #2a5a3a; color: white; }
  .btn-approve.active { background: #4a8; color: white; }
  .btn-reset { background: #5a5a2a; color: white; }
  .btn-point { background: #333; color: white; }
  .btn-point.active { outline: 2px solid #fff; outline-offset: 1px; }
  .legend { display: inline-flex; gap: 8px; font-size: 12px; align-items: center; }
  .swatch { display: inline-block; width: 12px; height: 12px; border: 1px solid #000; vertical-align: middle; }
  .point-readout { flex: 1 1 260px; min-width: 0; font-family: monospace; font-size: 11px; color: #aaa; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .empty { color: #bbb; padding: 24px; border: 1px solid #444; border-radius: 8px; }
</style>
</head><body>

<div class="app">
  <div class="toolbar">
    <button id="prevBtn" onclick="goCase(-1)">Previous</button>
    <select id="caseSelect" onchange="goToIndex(this.selectedIndex)"></select>
    <button id="nextBtn" onclick="goCase(1)">Next</button>
    <button onclick="downloadJSON()">Download JSON</button>
    <button onclick="copyToClipboard()">Copy JSON</button>
    <button onclick="approveAll()">Approve all</button>
    <span class="legend">
      <span><span class="swatch" style="background:#fff"></span> vertex</span>
      <span><span class="swatch" style="background:#f55"></span> axis_x</span>
      <span><span class="swatch" style="background:#5a5"></span> axis_y</span>
      <span><span class="swatch" style="background:#55f"></span> axis_z</span>
    </span>
    <span class="stats" id="stats"></span>
  </div>

  <div class="intro">
    <h2>Cube axis labeling</h2>
    <p>Place WHITE at the trihedral vertex and RED/GREEN/BLUE at the three <strong>FAR / silhouette-axis-endpoint</strong> corners (the silhouette corner that visually marks each world-axis direction from the vertex — in iso projection these are the two-cube-edge corners of each visible face, NOT the one-edge-away neighbors of the vertex). See <code>tools/FULL_CORNER_LABELING.md</code> "Axis-truth schema convention". Drag a marker, or choose a marker button and click the photo.</p>
  </div>

  <div class="viewer" id="case-host"></div>
</div>

<script>
const CASES = __CASES_JSON__;
const judgments = {};  // key → { vertex, axis_x, axis_y, axis_z, approved }
let currentIndex = 0;

function deepCopy(o) { return JSON.parse(JSON.stringify(o)); }

function init() {
  CASES.forEach(c => {
    judgments[c.key] = {
      vertex: c.prefill.vertex.slice(),
      axis_x: c.prefill.axis_x.slice(),
      axis_y: c.prefill.axis_y.slice(),
      axis_z: c.prefill.axis_z.slice(),
      approved: false,
      lastTouched: 'vertex',
    };
  });
  const select = document.getElementById('caseSelect');
  select.innerHTML = CASES.map((c, i) => `<option value="${i}">${i + 1}. ${c.key}</option>`).join('');
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') goCase(-1);
    if (e.key === 'ArrowRight') goCase(1);
  });
  renderCase();
}

function origToDisplay(c, pt) {
  return [(pt[0] - c.crop_x0) * c.scale, (pt[1] - c.crop_y0) * c.scale];
}
function displayToOrig(c, dx, dy) {
  return [dx / c.scale + c.crop_x0, dy / c.scale + c.crop_y0];
}

const MARKER_RADIUS = 9;
const POINT_NAMES = ['vertex', 'axis_x', 'axis_y', 'axis_z'];
const POINT_COLORS = {
  vertex: '#ffffff',
  axis_x: '#ff5050',
  axis_y: '#50aa50',
  axis_z: '#5050ff',
};

function canvasToCssScale(cv) {
  const rect = cv.getBoundingClientRect();
  if (!rect.width || !rect.height) return 1;
  return Math.max(cv.width / rect.width, cv.height / rect.height);
}

function markerRadiusForCanvas(cv) {
  return MARKER_RADIUS * canvasToCssScale(cv);
}

function currentCase() {
  return CASES[currentIndex];
}

function pointAt(c, dx, dy, cv) {
  const j = judgments[c.key];
  const hitRadius = markerRadiusForCanvas(cv) + 6 * canvasToCssScale(cv);
  for (const name of POINT_NAMES) {
    const [px, py] = origToDisplay(c, j[name]);
    const dist = Math.hypot(px - dx, py - dy);
    if (dist <= hitRadius) return name;
  }
  return null;
}

function pointButtons(c) {
  const active = judgments[c.key].lastTouched || 'vertex';
  return POINT_NAMES.map(name => {
    const label = name.replace('axis_', '');
    return `<button class="btn-point ${active === name ? 'active' : ''}" onclick="selectPoint('${name}')">${label}</button>`;
  }).join('');
}

function renderCase() {
  const host = document.getElementById('case-host');
  if (!CASES.length) {
    host.innerHTML = '<div class="empty">No cases generated.</div>';
    updateStats();
    return;
  }
  const c = currentCase();
  const j = judgments[c.key];
  host.innerHTML = `
    <div class="case ${j.approved ? 'approved' : ''}">
      <div class="case-head">
        <h3>${c.key}</h3>
        <div class="meta">prefill: ${c.prefill_source} · ${c.image_path_full}</div>
      </div>
      <div class="canvas-wrap"><canvas id="caseCanvas" width="${c.display_w}" height="${c.display_h}"></canvas></div>
      <div class="controls">
        <button class="btn-approve ${j.approved ? 'active' : ''}" onclick="toggleApprove('${c.key}')">${j.approved ? 'Approved' : 'Approve'}</button>
        <button class="btn-approve" onclick="approveAndNext()">Approve & next</button>
        <button class="btn-reset" onclick="resetToPrefill('${c.key}')">Reset</button>
        ${pointButtons(c)}
        <span class="point-readout" id="readout-${c.key}"></span>
      </div>
    </div>
  `;
  const select = document.getElementById('caseSelect');
  if (select) select.selectedIndex = currentIndex;
  const cv = document.getElementById('caseCanvas');
  const ctx = cv.getContext('2d');
  const img = new Image();
  img.onload = () => {
    ctx.drawImage(img, 0, 0);
    drawMarkers(c, ctx);
    updateReadout(c);
  };
  img.src = c.image_file;
  attachHandlers(c, cv);
  updateStats();
}

function drawMarkers(c, ctx) {
  const j = judgments[c.key];
  const markerRadius = markerRadiusForCanvas(ctx.canvas);
  const strokeWidth = Math.max(1, 2 * canvasToCssScale(ctx.canvas));
  for (const name of POINT_NAMES) {
    const [px, py] = origToDisplay(c, j[name]);
    ctx.beginPath();
    ctx.arc(px, py, markerRadius, 0, 2 * Math.PI);
    ctx.fillStyle = POINT_COLORS[name];
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = strokeWidth;
    ctx.stroke();
  }
  // Draw axis lines from vertex
  const [vx, vy] = origToDisplay(c, j.vertex);
  for (const name of ['axis_x', 'axis_y', 'axis_z']) {
    const [nx, ny] = origToDisplay(c, j[name]);
    ctx.beginPath();
    ctx.moveTo(vx, vy);
    ctx.lineTo(nx, ny);
    ctx.strokeStyle = POINT_COLORS[name];
    ctx.lineWidth = Math.max(1, 1.5 * canvasToCssScale(ctx.canvas));
    ctx.stroke();
  }
}

function attachHandlers(c, cv) {
  let dragging = null;
  cv.addEventListener('pointerdown', (e) => {
    const rect = cv.getBoundingClientRect();
    const dx = (e.clientX - rect.left) * (cv.width / rect.width);
    const dy = (e.clientY - rect.top) * (cv.height / rect.height);
    const hit = pointAt(c, dx, dy, cv);
    if (hit) {
      dragging = hit;
      judgments[c.key].lastTouched = hit;
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
  const cv = document.getElementById('caseCanvas');
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

function selectPoint(name) {
  const c = currentCase();
  judgments[c.key].lastTouched = name;
  renderCase();
}

function updateReadout(c) {
  const el = document.getElementById('readout-' + c.key);
  if (!el) return;
  const j = judgments[c.key];
  const fmt = (p) => `(${p[0].toFixed(0)},${p[1].toFixed(0)})`;
  el.textContent = `vertex=${fmt(j.vertex)} axis_x=${fmt(j.axis_x)} axis_y=${fmt(j.axis_y)} axis_z=${fmt(j.axis_z)}`;
}

function toggleApprove(key) {
  judgments[key].approved = !judgments[key].approved;
  renderCase();
}
function approveAndNext() {
  const c = currentCase();
  judgments[c.key].approved = true;
  if (currentIndex < CASES.length - 1) currentIndex += 1;
  renderCase();
}
function resetToPrefill(key) {
  const c = CASES.find(c => c.key === key);
  judgments[key].vertex = c.prefill.vertex.slice();
  judgments[key].axis_x = c.prefill.axis_x.slice();
  judgments[key].axis_y = c.prefill.axis_y.slice();
  judgments[key].axis_z = c.prefill.axis_z.slice();
  judgments[key].approved = false;
  judgments[key].lastTouched = 'vertex';
  renderCase();
}
function approveAll() {
  Object.keys(judgments).forEach(k => { judgments[k].approved = true; });
  renderCase();
}
function goToIndex(index) {
  if (!CASES.length) return;
  currentIndex = Math.max(0, Math.min(CASES.length - 1, index));
  renderCase();
}
function goCase(delta) {
  goToIndex(currentIndex + delta);
}
function updateStats() {
  const n = Object.keys(judgments).length;
  const approved = Object.values(judgments).filter(j => j.approved).length;
  document.getElementById('stats').textContent = CASES.length
    ? `case: ${currentIndex + 1}/${CASES.length} · approved: ${approved}/${n}`
    : `approved: ${approved}/${n}`;
  const prev = document.getElementById('prevBtn');
  const next = document.getElementById('nextBtn');
  if (prev) prev.disabled = currentIndex <= 0;
  if (next) next.disabled = currentIndex >= CASES.length - 1;
}

function buildExport() {
  const out = {};
  CASES.forEach(c => {
    const j = judgments[c.key];
    if (j.approved) {
      out[c.key] = {
        vertex: [Math.round(j.vertex[0]*10)/10, Math.round(j.vertex[1]*10)/10],
        axis_x: [Math.round(j.axis_x[0]*10)/10, Math.round(j.axis_x[1]*10)/10],
        axis_y: [Math.round(j.axis_y[0]*10)/10, Math.round(j.axis_y[1]*10)/10],
        axis_z: [Math.round(j.axis_z[0]*10)/10, Math.round(j.axis_z[1]*10)/10],
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
  const text = JSON.stringify(buildExport(), null, 2);
  if (navigator.clipboard) {
    navigator.clipboard.writeText(text).then(
      () => alert('Copied to clipboard'),
      () => window.prompt('Copy JSON:', text)
    );
  } else {
    window.prompt('Copy JSON:', text);
  }
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
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)
    print(f"  {len(cases)} cases", file=sys.stderr)
    print(f"  Open: file://{out_path.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
