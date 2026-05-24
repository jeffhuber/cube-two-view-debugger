#!/usr/bin/env python
"""Build a static full-corner labeling gallery for visible cube geometry.

This is the convention-reset labeler. It asks a human to label all seven
visible geometry corners directly:

    vertex, corner_0, corner_1, corner_2, corner_3, corner_4, corner_5

The six numbered corners use the project-owner visual convention:

    image A: Va with upper/right/front view slots
      upper = Va + 1,0,5
      right = Va + 3,2,1
      front = Va + 5,4,3

    image B: Vb with upper/right/front view slots after the 180-degree
    camera-X flip
      upper = Vb + 2,3,4
      right = Vb + 0,1,2
      front = Vb + 4,5,0

This tool intentionally avoids model-axis names such as h_x/h_y/h_z and
near_x/near_y/near_z. The output JSON is meant to be the unambiguous bridge
from human-visible geometry to any downstream model convention.

Usage:
    .venv/bin/python tools/build_full_corner_labeling_gallery.py \\
        --out /tmp/full_corner_labeling_v1 \\
        --sets 20 38 40 41 43 45
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.corner_conventions import (  # noqa: E402
    FACE_DEFS_BY_SIDE,
    POINT_NAMES,
    VERTEX_NAME_BY_SIDE,
)


Point = Tuple[float, float]


def _load_manifests() -> List[Dict[str, Any]]:
    out = []
    for fname in ("hard_case_manifest.json", "corpus_manifest.json"):
        with (REPO_ROOT / "tests" / "fixtures" / fname).open(encoding="utf-8") as handle:
            out.append(json.load(handle))
    return out


def _resolve_pair_paths(manifests: List[Dict[str, Any]], set_id: str) -> Optional[Tuple[Path, Path]]:
    for manifest in manifests:
        for entry in manifest["pairs"]:
            if str(entry["setId"]) == str(set_id):
                return Path(entry["imageAPath"]).expanduser(), Path(entry["imageBPath"]).expanduser()

    for root in _candidate_corpus_roots(manifests):
        path_a = _find_corpus_side(root, set_id, "A")
        path_b = _find_corpus_side(root, set_id, "B")
        if path_a is not None and path_b is not None:
            return path_a, path_b
    return None


def _candidate_corpus_roots(manifests: List[Dict[str, Any]]) -> List[Path]:
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


def _full_image_display(img_size: Tuple[int, int]) -> Tuple[Tuple[int, int, int, int], float, Tuple[int, int]]:
    width, height = img_size
    return (0, 0, width, height), 1.0, (width, height)


def _initial_full_corner_prefill(img_size: Tuple[int, int]) -> Dict[str, List[float]]:
    width, height = img_size
    cx, cy = width / 2.0, height / 2.0
    radius = max(120.0, min(width, height) * 0.22)
    return {
        "vertex": [round(cx, 1), round(cy, 1)],
        "corner_0": [round(cx, 1), round(cy - radius, 1)],
        "corner_1": [round(cx + 0.82 * radius, 1), round(cy - 0.48 * radius, 1)],
        "corner_2": [round(cx + 0.82 * radius, 1), round(cy + 0.48 * radius, 1)],
        "corner_3": [round(cx, 1), round(cy + radius, 1)],
        "corner_4": [round(cx - 0.82 * radius, 1), round(cy + 0.48 * radius, 1)],
        "corner_5": [round(cx - 0.82 * radius, 1), round(cy - 0.48 * radius, 1)],
    }


def _load_truth_prefill(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Load existing full-corner truth as prefill, preserving point
    coordinates AND yaw_quarter_turns when present.

    The yaw field is OPTIONAL in the schema (Codex P2 on the yaw-fixture
    PR) but if a row carries it on disk we must round-trip it through
    the gallery so an edit-and-redownload cycle doesn't drop the value.
    """
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for key, item in data.items():
        if all(name in item for name in POINT_NAMES):
            row: Dict[str, Any] = {
                name: [float(item[name][0]), float(item[name][1])]
                for name in POINT_NAMES
            }
            if "yaw_quarter_turns" in item:
                row["yaw_quarter_turns"] = int(item["yaw_quarter_turns"])
            out[str(key)] = row
    return out


def _build_case_data(
    set_id: str,
    side: str,
    img_path: Path,
    out_dir: Path,
    truth_prefill: Dict[str, Dict[str, List[float]]],
) -> Dict[str, Any]:
    img = _exif_correct(img_path)
    crop_box, scale, (display_w, display_h) = _full_image_display(img.size)
    out_png = out_dir / f"set_{set_id}_{side}.png"
    img.crop(crop_box).save(out_png, compress_level=1)
    key = f"{set_id}_{side}"
    if key in truth_prefill:
        prefill = truth_prefill[key]
        prefill_source = "truth"
    else:
        prefill = _initial_full_corner_prefill(img.size)
        prefill_source = "layout"
    return {
        "key": key,
        "set_id": set_id,
        "side": side,
        "image_file": out_png.name,
        "image_path_full": str(img_path),
        "crop_x0": crop_box[0],
        "crop_y0": crop_box[1],
        "scale": scale,
        "display_w": display_w,
        "display_h": display_h,
        "prefill": prefill,
        "prefill_source": prefill_source,
    }


HTML_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Full cube corner labeling gallery</title>
<style>
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; overflow: hidden; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #171717; color: #eee; }
  .app { height: 100vh; min-height: 0; display: flex; flex-direction: column; }
  .toolbar { flex: 0 0 auto; background: #292929; padding: 8px 12px;
             border-bottom: 1px solid #444; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .toolbar button { background: #4a4a4a; color: white; border: none; padding: 8px 12px;
                    border-radius: 4px; cursor: pointer; font-size: 14px; }
  .toolbar button:hover { background: #5a5a5a; }
  .toolbar button:disabled { opacity: .45; cursor: default; }
  .toolbar select { background: #1f1f1f; color: #fff; border: 1px solid #555; border-radius: 4px; padding: 7px 8px; min-width: 150px; }
  .toolbar .stats { margin-left: auto; font-size: 13px; color: #aaa; white-space: nowrap; }
  .intro { flex: 0 0 auto; padding: 8px 12px; border-bottom: 1px solid #333; color: #cfcfcf; font-size: 13px; line-height: 1.35; }
  .intro h2 { display: inline; margin: 0 10px 0 0; color: #fff; font-size: 16px; }
  .intro p { display: inline; margin: 0; }
  .viewer { flex: 1 1 auto; min-height: 0; display: grid; place-items: center; padding: 8px 12px 12px; }
  .case { width: min(100%, 1100px); max-height: 100%; min-height: 0; padding: 10px; background: #252525; border-radius: 8px; border: 2px solid #444;
          display: grid; grid-template-rows: auto minmax(0, 1fr) auto; gap: 8px; }
  .case.approved { border-color: #4a8; }
  .case-head { min-width: 0; display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
  .case h3 { margin: 0; color: #fff; font-size: 16px; }
  .case .meta { min-width: 0; color: #aaa; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .canvas-wrap { min-height: 0; display: flex; align-items: center; justify-content: center; overflow: visible; }
  canvas { display: block; cursor: crosshair; border: 1px solid #555; touch-action: none; width: auto; height: auto; max-width: 100%; max-height: 100%; }
  .controls { display: flex; gap: 7px; align-items: center; flex-wrap: wrap; }
  .controls button { padding: 6px 10px; border-radius: 4px; border: 1px solid #555; cursor: pointer; }
  .btn-approve { background: #2a5a3a; color: white; }
  .btn-approve.active { background: #4a8; color: white; }
  .btn-reset { background: #5a5a2a; color: white; }
  .btn-point { background: #333; color: white; min-width: 38px; }
  .btn-point.active { outline: 2px solid #fff; outline-offset: 1px; }
  .legend { display: inline-flex; gap: 8px; font-size: 12px; align-items: center; }
  .swatch { display: inline-block; width: 12px; height: 12px; border: 1px solid #000; vertical-align: middle; }
  .point-readout { flex: 1 1 360px; min-width: 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; color: #aaa; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
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
      <span><span class="swatch" style="background:#fff"></span> Va/Vb</span>
      <span><span class="swatch" style="background:#f5d142"></span> 0</span>
      <span><span class="swatch" style="background:#4fa3ff"></span> 1</span>
      <span><span class="swatch" style="background:#ff5a5a"></span> 2</span>
      <span><span class="swatch" style="background:#46d96c"></span> 3</span>
      <span><span class="swatch" style="background:#d86cff"></span> 4</span>
      <span><span class="swatch" style="background:#ff9f43"></span> 5</span>
    </span>
    <span class="stats" id="stats"></span>
  </div>
  <div class="intro">
    <h2>Full corner labeling</h2>
    <p>Label Va/Vb plus all six outer corners. A slots: upper=Va+1,0,5; right=Va+3,2,1; front=Va+5,4,3. B slots: upper=Vb+2,3,4; right=Vb+0,1,2; front=Vb+4,5,0. Canonical face letters depend on capture yaw. Use point buttons or number keys, then click/drag markers.</p>
  </div>
  <div class="viewer" id="case-host"></div>
</div>
<script>
const CASES = __CASES_JSON__;
const POINT_NAMES = __POINT_NAMES_JSON__;
const VERTEX_NAME_BY_SIDE = __VERTEX_NAME_BY_SIDE_JSON__;
const FACE_DEFS_BY_SIDE = __FACE_DEFS_BY_SIDE_JSON__;
const POINT_COLORS = {
  vertex: '#ffffff',
  corner_0: '#f5d142',
  corner_1: '#4fa3ff',
  corner_2: '#ff5a5a',
  corner_3: '#46d96c',
  corner_4: '#d86cff',
  corner_5: '#ff9f43',
};
const FACE_STYLES = {
  upper: { fill: 'rgba(255,255,255,.30)', stroke: 'rgba(255,255,255,.9)' },
  right: { fill: 'rgba(255,90,90,.24)', stroke: 'rgba(255,90,90,.9)' },
  front: { fill: 'rgba(70,217,108,.22)', stroke: 'rgba(70,217,108,.9)' },
};
const SLOT_LABELS = {
  upper: 'Up',
  right: 'Rt',
  front: 'Fr',
};
const judgments = {};
let currentIndex = 0;

function init() {
  CASES.forEach(c => {
    const j = { approved: false, lastTouched: 'vertex' };
    POINT_NAMES.forEach(name => { j[name] = c.prefill[name].slice(); });
    // Optional yaw_quarter_turns: load from prefill if present, else
    // null (= unspecified; will be omitted from export). Codex P2 on
    // the yaw-fixture PR: gallery must round-trip yaw so an edit-and-
    // redownload cycle doesn't drop the value on rows that have it.
    j.yaw_quarter_turns = (c.prefill.yaw_quarter_turns !== undefined)
      ? c.prefill.yaw_quarter_turns
      : null;
    judgments[c.key] = j;
  });
  const select = document.getElementById('caseSelect');
  select.innerHTML = CASES.map((c, i) => `<option value="${i}">${i + 1}. ${c.key}</option>`).join('');
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') goCase(-1);
    if (e.key === 'ArrowRight') goCase(1);
    if (e.key === 'v' || e.key === 'V') selectPoint('vertex');
    if (/^[0-5]$/.test(e.key)) selectPoint('corner_' + e.key);
  });
  renderCase();
}

function currentCase() { return CASES[currentIndex]; }
function origToDisplay(c, pt) { return [(pt[0] - c.crop_x0) * c.scale, (pt[1] - c.crop_y0) * c.scale]; }
function displayToOrig(c, dx, dy) { return [dx / c.scale + c.crop_x0, dy / c.scale + c.crop_y0]; }
function canvasToCssScale(cv) {
  const rect = cv.getBoundingClientRect();
  if (!rect.width || !rect.height) return 1;
  return Math.max(cv.width / rect.width, cv.height / rect.height);
}
function markerRadiusForCanvas(cv) { return 9 * canvasToCssScale(cv); }

function pointButtons(c) {
  const active = judgments[c.key].lastTouched || 'vertex';
  return POINT_NAMES.map(name => {
    const label = name === 'vertex' ? vertexLabel(c) : name.replace('corner_', '');
    return `<button class="btn-point ${active === name ? 'active' : ''}" onclick="selectPoint('${name}')">${label}</button>`;
  }).join('');
}
function vertexLabel(c) { return VERTEX_NAME_BY_SIDE[c.side] || 'V'; }
function faceSummary(c) {
  return c.side === 'B'
    ? 'B slots: upper=Vb+2,3,4; right=Vb+0,1,2; front=Vb+4,5,0'
    : 'A slots: upper=Va+1,0,5; right=Va+3,2,1; front=Va+5,4,3';
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
        <div class="meta">${faceSummary(c)} - prefill: ${c.prefill_source} - ${c.image_path_full}</div>
      </div>
      <div class="canvas-wrap"><canvas id="caseCanvas" width="${c.display_w}" height="${c.display_h}"></canvas></div>
      <div class="controls">
        <button class="btn-approve ${j.approved ? 'active' : ''}" onclick="toggleApprove('${c.key}')">${j.approved ? 'Approved' : 'Approve'}</button>
        <button class="btn-approve" onclick="approveAndNext()">Approve & next</button>
        <button class="btn-reset" onclick="resetToPrefill('${c.key}')">Reset</button>
        ${pointButtons(c)}
        <label style="margin-left:8px;">yaw_quarter_turns:
          <select onchange="setYaw('${c.key}', this.value)">
            <option value="">(unset)</option>
            <option value="0" ${j.yaw_quarter_turns === 0 ? 'selected' : ''}>0</option>
            <option value="1" ${j.yaw_quarter_turns === 1 ? 'selected' : ''}>1</option>
            <option value="2" ${j.yaw_quarter_turns === 2 ? 'selected' : ''}>2</option>
            <option value="3" ${j.yaw_quarter_turns === 3 ? 'selected' : ''}>3</option>
          </select>
        </label>
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
    drawOverlays(c, ctx);
    updateReadout(c);
  };
  img.src = c.image_file;
  attachHandlers(c, cv);
  updateStats();
}

function drawOverlays(c, ctx) {
  const j = judgments[c.key];
  const markerRadius = markerRadiusForCanvas(ctx.canvas);
  const scale = canvasToCssScale(ctx.canvas);
  const faceDefs = FACE_DEFS_BY_SIDE[c.side] || FACE_DEFS_BY_SIDE.A;
  for (const [faceName, names] of Object.entries(faceDefs)) {
    const points = names.map(name => origToDisplay(c, j[name]));
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    points.slice(1).forEach(p => ctx.lineTo(p[0], p[1]));
    ctx.closePath();
    ctx.fillStyle = FACE_STYLES[faceName].fill;
    ctx.fill();
    ctx.strokeStyle = FACE_STYLES[faceName].stroke;
    ctx.lineWidth = Math.max(1, 2.2 * scale);
    ctx.stroke();
    const cx = points.reduce((acc, p) => acc + p[0], 0) / points.length;
    const cy = points.reduce((acc, p) => acc + p[1], 0) / points.length;
    ctx.font = `${Math.max(18, 18 * scale)}px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif`;
    ctx.lineWidth = Math.max(2, 4 * scale);
    ctx.strokeStyle = '#000';
    const slotLabel = SLOT_LABELS[faceName] || faceName;
    ctx.strokeText(slotLabel, cx, cy);
    ctx.fillStyle = FACE_STYLES[faceName].stroke;
    ctx.fillText(slotLabel, cx, cy);
  }
  for (const name of POINT_NAMES) {
    const [px, py] = origToDisplay(c, j[name]);
    ctx.beginPath();
    ctx.arc(px, py, markerRadius, 0, 2 * Math.PI);
    ctx.fillStyle = POINT_COLORS[name];
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = Math.max(1, 2 * scale);
    ctx.stroke();
    const label = name === 'vertex' ? vertexLabel(c) : name.replace('corner_', '');
    ctx.font = `${Math.max(14, 14 * scale)}px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif`;
    ctx.lineWidth = Math.max(2, 3 * scale);
    ctx.strokeStyle = '#000';
    ctx.strokeText(label, px + markerRadius + 4 * scale, py - markerRadius - 2 * scale);
    ctx.fillStyle = POINT_COLORS[name];
    ctx.fillText(label, px + markerRadius + 4 * scale, py - markerRadius - 2 * scale);
  }
}

function pointAt(c, dx, dy, cv) {
  const j = judgments[c.key];
  const hitRadius = markerRadiusForCanvas(cv) + 6 * canvasToCssScale(cv);
  for (const name of POINT_NAMES) {
    const [px, py] = origToDisplay(c, j[name]);
    if (Math.hypot(px - dx, py - dy) <= hitRadius) return name;
  }
  return null;
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
      judgments[c.key][judgments[c.key].lastTouched] = displayToOrig(c, dx, dy);
      redraw(c);
    }
  });
  cv.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const rect = cv.getBoundingClientRect();
    const dx = (e.clientX - rect.left) * (cv.width / rect.width);
    const dy = (e.clientY - rect.top) * (cv.height / rect.height);
    judgments[c.key][dragging] = displayToOrig(c, dx, dy);
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
    drawOverlays(c, ctx);
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
  el.textContent = POINT_NAMES.map(name => {
    const label = name === 'vertex' ? vertexLabel(c) : name.replace('corner_', '');
    return `${label}=${fmt(j[name])}`;
  }).join(' ');
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
  const c = CASES.find(item => item.key === key);
  POINT_NAMES.forEach(name => { judgments[key][name] = c.prefill[name].slice(); });
  judgments[key].approved = false;
  judgments[key].lastTouched = 'vertex';
  // Restore yaw from prefill too — otherwise a stale user-edited yaw
  // survives the reset and gets exported as if it were the truth value
  // (Codex P3 on the yaw-fixture PR, round-3). Match the init() logic:
  // load from prefill if present, else null (= unset).
  judgments[key].yaw_quarter_turns = (c.prefill.yaw_quarter_turns !== undefined)
    ? c.prefill.yaw_quarter_turns
    : null;
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
function goCase(delta) { goToIndex(currentIndex + delta); }
function updateStats() {
  const n = Object.keys(judgments).length;
  const approved = Object.values(judgments).filter(j => j.approved).length;
  document.getElementById('stats').textContent = CASES.length
    ? `case: ${currentIndex + 1}/${CASES.length} - approved: ${approved}/${n}`
    : `approved: ${approved}/${n}`;
  const prev = document.getElementById('prevBtn');
  const next = document.getElementById('nextBtn');
  if (prev) prev.disabled = currentIndex <= 0;
  if (next) next.disabled = currentIndex >= CASES.length - 1;
}
function setYaw(key, value) {
  // Persist yaw_quarter_turns to the judgment for `key`. Empty string
  // means "unset" (not exported). Integer string in {0..3} stored as
  // int so the JSON output matches the canonical schema.
  if (value === '' || value == null) {
    judgments[key].yaw_quarter_turns = null;
  } else {
    judgments[key].yaw_quarter_turns = parseInt(value, 10);
  }
}
function buildExport() {
  const out = {};
  CASES.forEach(c => {
    const j = judgments[c.key];
    if (j.approved) {
      out[c.key] = { approved: true };
      POINT_NAMES.forEach(name => {
        out[c.key][name] = [Math.round(j[name][0] * 10) / 10, Math.round(j[name][1] * 10) / 10];
      });
      // Emit yaw_quarter_turns only when set, so the schema stays
      // optional. Round-trips values loaded from existing truth (Codex
      // P2 on the yaw-fixture PR).
      if (j.yaw_quarter_turns !== null && j.yaw_quarter_turns !== undefined) {
        out[c.key].yaw_quarter_turns = j.yaw_quarter_turns;
      }
    }
  });
  return out;
}
function downloadJSON() {
  const blob = new Blob([JSON.stringify(buildExport(), null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'full_corner_ground_truth.json';
  a.click();
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


def _extract_cases_json(html: str) -> List[Dict[str, Any]]:
    match = re.search(r"const CASES = (.*?);\nconst POINT_NAMES", html, re.S)
    if not match:
        raise ValueError("CASES block not found")
    return json.loads(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--sets",
        nargs="*",
        default=None,
        help="Set IDs to include (default: every labeled set with both photos)",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=None,
        help="Optional existing full-corner JSON to prefill matching rows.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    truth_prefill = _load_truth_prefill(args.truth)
    manifests = _load_manifests()
    all_pairs = [entry for manifest in manifests for entry in manifest["pairs"]]
    seen = set()
    unique = []
    for pair in all_pairs:
        set_id = str(pair["setId"])
        if set_id in seen:
            continue
        seen.add(set_id)
        unique.append(pair)

    sets_to_process = args.sets if args.sets else [str(pair["setId"]) for pair in unique]
    cases: List[Dict[str, Any]] = []
    for set_id in sets_to_process:
        paths = _resolve_pair_paths(manifests, str(set_id))
        if paths is None:
            print(f"set {set_id}: not in any manifest or corpus root", file=sys.stderr)
            continue
        path_a, path_b = paths
        for side, path in (("A", path_a), ("B", path_b)):
            if not path.exists():
                print(f"set {set_id} {side}: {path} not found", file=sys.stderr)
                continue
            print(f"[set {set_id} {side}] processing {path.name}", file=sys.stderr)
            cases.append(_build_case_data(str(set_id), side, path, args.out, truth_prefill))

    html = HTML_TEMPLATE.replace("__CASES_JSON__", json.dumps(cases))
    html = html.replace("__POINT_NAMES_JSON__", json.dumps(POINT_NAMES))
    html = html.replace("__VERTEX_NAME_BY_SIDE_JSON__", json.dumps(VERTEX_NAME_BY_SIDE))
    html = html.replace("__FACE_DEFS_BY_SIDE_JSON__", json.dumps(FACE_DEFS_BY_SIDE))
    out_path = args.out / "gallery.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}", file=sys.stderr)
    print(f"  {len(cases)} cases", file=sys.stderr)
    print(f"  Open: file://{out_path.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
