#!/usr/bin/env python3
"""Local click-to-label UI for vertex-point human feedback.

Diagnostics/data-only helper. This starts a tiny localhost server that lets a
human click the true visible trihedral vertex point on each overlay and writes
the labels into ``tests/fixtures/vertex_point_human_feedback.json``.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.vertex_point_feedback import (  # noqa: E402
    DEFAULT_FEEDBACK,
    DEFAULT_REPORT,
    evaluate_feedback,
    render_report,
)


Point = Tuple[float, float]


@dataclass(frozen=True)
class ServerConfig:
    feedback_path: Path
    report_path: Path
    host: str
    port: int


def load_feedback(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_feedback(path: Path, report_path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evaluation = evaluate_feedback(payload)
    report_path.write_text(render_report(payload, evaluation), encoding="utf-8")
    return evaluation


def update_feedback_row(
    feedback: Dict[str, Any],
    *,
    set_id: str,
    side: str,
    status: str,
    human_vertex_point: Optional[Point],
    label_quality: Optional[str],
    notes: str,
) -> None:
    if status not in set(feedback.get("allowedStatuses", [])):
        raise ValueError(f"unsupported status: {status}")
    for row in feedback.get("rows", []):
        if str(row.get("setId")) == str(set_id) and str(row.get("side")) == side:
            row["status"] = status
            row["humanVertexPoint"] = (
                [round(float(human_vertex_point[0]), 2), round(float(human_vertex_point[1]), 2)]
                if human_vertex_point is not None
                else None
            )
            row["labelQuality"] = label_quality
            row["notes"] = notes
            return
    raise KeyError(f"feedback row not found: {set_id}:{side}")


def create_handler(config: ServerConfig) -> type[BaseHTTPRequestHandler]:
    class VertexLabelHandler(BaseHTTPRequestHandler):
        server_version = "VertexLabelServer/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_text(INDEX_HTML, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/feedback":
                feedback = load_feedback(config.feedback_path)
                evaluation = evaluate_feedback(feedback)
                self._send_json({
                    "feedback": feedback,
                    "evaluation": evaluation,
                    "feedbackPath": str(config.feedback_path),
                    "reportPath": str(config.report_path),
                })
                return
            if parsed.path == "/image":
                params = parse_qs(parsed.query)
                raw_path = (params.get("path") or [""])[0]
                self._send_file(raw_path)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/label":
                self._handle_label()
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}", file=sys.stderr)

        def _handle_label(self) -> None:
            try:
                payload = self._read_json_body()
                point = payload.get("humanVertexPoint")
                human_vertex_point = _parse_point(point)
                status = str(payload.get("status") or "unlabeled")
                if status == "labeled" and human_vertex_point is None:
                    raise ValueError("labeled rows require humanVertexPoint=[x,y]")
                feedback = load_feedback(config.feedback_path)
                update_feedback_row(
                    feedback,
                    set_id=str(payload.get("setId")),
                    side=str(payload.get("side")),
                    status=status,
                    human_vertex_point=human_vertex_point,
                    label_quality=payload.get("labelQuality"),
                    notes=str(payload.get("notes") or ""),
                )
                evaluation = save_feedback(config.feedback_path, config.report_path, feedback)
                self._send_json({"ok": True, "feedback": feedback, "evaluation": evaluation})
            except Exception as exc:
                self._send_json({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}, status=400)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            data = self.rfile.read(length)
            return json.loads(data.decode("utf-8"))

        def _send_file(self, raw_path: str) -> None:
            path = Path(raw_path)
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, f"file not found: {raw_path}")
                return
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 512)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

        def _send_json(self, payload: Dict[str, Any], *, status: int = 200) -> None:
            body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, content_type: str) -> None:
            body = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return VertexLabelHandler


def _parse_point(value: Any) -> Optional[Point]:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return (float(value[0]), float(value[1]))
    return None


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vertex Labeler</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #647080;
      --line: #d8dee8;
      --accent: #0f7b6c;
      --danger: #b3261e;
      --warn: #9b6700;
      --ok: #137333;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    button, select, input, textarea {
      font: inherit;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr) 320px;
    }
    aside, .details {
      background: var(--panel);
      border-color: var(--line);
      overflow: auto;
      max-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      padding: 12px;
    }
    .details {
      border-left: 1px solid var(--line);
      padding: 14px;
    }
    main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 14px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 650;
    }
    .summary {
      display: flex;
      gap: 12px;
      color: var(--muted);
      flex-wrap: wrap;
    }
    .row-list {
      display: grid;
      gap: 6px;
    }
    .row-button {
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      padding: 8px;
      border-radius: 6px;
      text-align: left;
      cursor: pointer;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 4px 8px;
      align-items: center;
    }
    .row-button.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(15, 123, 108, 0.15);
    }
    .row-button .sub {
      grid-column: 1 / -1;
      color: var(--muted);
      font-size: 12px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 1px 7px;
      border-radius: 999px;
      background: #eef2f7;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .pill.ok { color: var(--ok); background: #e6f4ea; }
    .pill.warn { color: var(--warn); background: #fff4d8; }
    .pill.bad { color: var(--danger); background: #fce8e6; }
    .stage-wrap {
      min-height: 0;
      padding: 14px;
      overflow: auto;
    }
    .stage {
      position: relative;
      max-width: min(100%, 980px);
      margin: 0 auto;
      border: 1px solid var(--line);
      background: #111;
      border-radius: 6px;
      overflow: hidden;
    }
    .stage img {
      width: 100%;
      height: auto;
      display: block;
      user-select: none;
    }
    .marker {
      position: absolute;
      width: 22px;
      height: 22px;
      margin: -11px 0 0 -11px;
      border: 3px solid #fff;
      border-radius: 50%;
      background: #ffdf22;
      box-shadow: 0 0 0 3px #111, 0 2px 10px rgba(0,0,0,0.4);
      pointer-events: none;
    }
    .candidate-hit {
      position: absolute;
      width: 16px;
      height: 16px;
      margin: -8px 0 0 -8px;
      border: 2px solid #111;
      border-radius: 50%;
      background: #fff;
      color: #111;
      font-size: 10px;
      font-weight: 700;
      display: grid;
      place-items: center;
      pointer-events: none;
    }
    .field {
      display: grid;
      gap: 6px;
      margin-bottom: 12px;
    }
    label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    select, input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 8px;
      color: var(--ink);
    }
    textarea {
      min-height: 84px;
      resize: vertical;
    }
    .buttons {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 12px 0;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 8px 10px;
      cursor: pointer;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
      font-weight: 650;
    }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .metrics, .candidates {
      border-top: 1px solid var(--line);
      padding-top: 12px;
      margin-top: 12px;
      display: grid;
      gap: 8px;
    }
    .metric-line, .candidate-line {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: baseline;
    }
    code {
      background: #eef2f7;
      border-radius: 4px;
      padding: 1px 4px;
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 220px minmax(0, 1fr); }
      .details { grid-column: 1 / -1; max-height: none; border-left: 0; border-top: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>Vertex Labels</h1>
      <p class="hint">Click the shared corner where the three visible cube faces meet. It is usually near the existing candidate cluster, not the lower silhouette corner.</p>
      <div id="rowList" class="row-list"></div>
    </aside>
    <main>
      <div class="topbar">
        <h1 id="title">Loading...</h1>
        <div class="summary" id="summary"></div>
      </div>
      <div class="stage-wrap">
        <div class="stage" id="stage">
          <img id="image" alt="vertex overlay">
          <div id="marker" class="marker" hidden></div>
          <div id="candidateLayer"></div>
        </div>
      </div>
    </main>
    <section class="details">
      <div class="field">
        <label>Target definition</label>
        <div class="hint">Mark the three-face meeting point: the single physical corner shared by the top face and the two side faces. Do not mark the bottom/front outer hull corner.</div>
      </div>
      <div class="field">
        <label for="imageMode">Image</label>
        <select id="imageMode">
          <option value="overlay">Overlay</option>
          <option value="original">Original</option>
        </select>
      </div>
      <div class="field">
        <label for="status">Label</label>
        <select id="status">
          <option value="labeled">Labeled</option>
          <option value="ambiguous">Ambiguous</option>
          <option value="not_visible">Not visible</option>
          <option value="unlabeled">Unlabeled</option>
        </select>
      </div>
      <div class="field">
        <label for="quality">Quality</label>
        <select id="quality">
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
          <option value="">None</option>
        </select>
      </div>
      <div class="field">
        <label for="point">Human vertex point</label>
        <input id="point" readonly>
      </div>
      <div class="field">
        <label for="notes">Notes</label>
        <textarea id="notes" placeholder="Optional note"></textarea>
      </div>
      <div class="buttons">
        <button id="prev">Previous</button>
        <button id="next">Next</button>
        <button id="clear">Clear</button>
        <button id="save" class="primary">Save</button>
      </div>
      <div id="saveStatus" class="hint"></div>
      <div class="metrics" id="metrics"></div>
      <div class="candidates" id="candidates"></div>
    </section>
  </div>
  <script>
    const state = {
      feedback: null,
      evaluation: null,
      selected: 0,
      draftPoint: null,
      imageSizes: new Map(),
      imageNatural: { width: 0, height: 0 },
    };
    const els = {
      rowList: document.getElementById('rowList'),
      title: document.getElementById('title'),
      summary: document.getElementById('summary'),
      image: document.getElementById('image'),
      stage: document.getElementById('stage'),
      marker: document.getElementById('marker'),
      candidateLayer: document.getElementById('candidateLayer'),
      imageMode: document.getElementById('imageMode'),
      status: document.getElementById('status'),
      quality: document.getElementById('quality'),
      point: document.getElementById('point'),
      notes: document.getElementById('notes'),
      prev: document.getElementById('prev'),
      next: document.getElementById('next'),
      clear: document.getElementById('clear'),
      save: document.getElementById('save'),
      saveStatus: document.getElementById('saveStatus'),
      metrics: document.getElementById('metrics'),
      candidates: document.getElementById('candidates'),
    };

    async function loadData() {
      const res = await fetch('/api/feedback');
      const data = await res.json();
      state.feedback = data.feedback;
      state.evaluation = data.evaluation;
      render();
    }

    function selectedRow() {
      return state.feedback.rows[state.selected];
    }

    function selectedEval() {
      const row = selectedRow();
      return state.evaluation.rows.find(item => item.setId === row.setId && item.side === row.side) || {};
    }

    function imagePathFor(row) {
      return els.imageMode.value === 'original' ? row.imagePath : row.overlayPath;
    }

    function render() {
      const row = selectedRow();
      const evaluation = state.evaluation.summary;
      els.summary.innerHTML = [
        `labeled ${evaluation.labeledRowCount}/${evaluation.rowCount}`,
        `top-1 ${formatRate(evaluation.top1HitRate)}`,
        `top-3 ${formatRate(evaluation.top3HitRate)}`
      ].map(text => `<span>${text}</span>`).join('');

      els.title.textContent = `Set ${row.setId} ${row.side}`;
      awaitImageSizes(row).then(() => {
        renderMarker();
        renderCandidateDots();
      });
      els.image.src = `/image?path=${encodeURIComponent(imagePathFor(row))}`;
      els.status.value = row.status === 'unlabeled' && row.humanVertexPoint ? 'labeled' : row.status;
      els.quality.value = row.labelQuality || 'high';
      els.notes.value = row.notes || '';
      state.draftPoint = row.humanVertexPoint ? [...row.humanVertexPoint] : null;
      updatePointControls();
      renderRowList();
      renderCandidates();
      renderMetrics();
      renderMarker();
    }

    function renderRowList() {
      els.rowList.innerHTML = '';
      state.feedback.rows.forEach((row, idx) => {
        const evalRow = state.evaluation.rows.find(item => item.setId === row.setId && item.side === row.side) || {};
        const button = document.createElement('button');
        button.className = `row-button ${idx === state.selected ? 'active' : ''}`;
        button.type = 'button';
        button.innerHTML = `
          <strong>${row.setId} ${row.side}</strong>
          <span class="pill ${pillClass(evalRow)}">${evalRow.evaluationStatus || row.status}</span>
          <span class="sub">${row.candidateStatus || ''}${evalRow.top1DistancePx !== undefined ? ` · top1 ${evalRow.top1DistancePx}px` : ''}</span>
        `;
        button.addEventListener('click', () => {
          state.selected = idx;
          render();
        });
        els.rowList.appendChild(button);
      });
    }

    function renderCandidates() {
      const row = selectedRow();
      els.candidates.innerHTML = '<strong>Top candidates</strong>';
      row.topCandidates.forEach(candidate => {
        const line = document.createElement('div');
        line.className = 'candidate-line';
        const distance = state.draftPoint ? `${dist(state.draftPoint, candidate.vertexPoint).toFixed(2)}px` : '';
        line.innerHTML = `<span>r${candidate.rank} <code>${candidate.source}</code></span><span>${distance}</span>`;
        els.candidates.appendChild(line);
      });
      renderCandidateDots();
    }

    function renderMetrics() {
      const evalRow = selectedEval();
      const rows = [
        ['Candidate status', selectedRow().candidateStatus || ''],
        ['Top-1 distance', evalRow.top1DistancePx !== undefined ? `${evalRow.top1DistancePx}px` : ''],
        ['Top-1 hit', boolText(evalRow.top1WithinThreshold)],
        ['Top-3 contains truth', boolText(evalRow.top3ContainsTruth)],
        ['Best rank', evalRow.bestRank || ''],
        ['Best distance', evalRow.bestDistancePx !== undefined ? `${evalRow.bestDistancePx}px` : ''],
      ];
      els.metrics.innerHTML = '<strong>Metrics</strong>' + rows.map(([label, value]) =>
        `<div class="metric-line"><span>${label}</span><span>${value}</span></div>`
      ).join('');
    }

    function renderCandidateDots() {
      els.candidateLayer.innerHTML = '';
      const row = selectedRow();
      row.topCandidates.forEach(candidate => {
        const dot = document.createElement('div');
        dot.className = 'candidate-hit';
        dot.textContent = candidate.rank;
        positionElement(dot, overlayPointToDisplayPoint(candidate.vertexPoint));
        els.candidateLayer.appendChild(dot);
      });
    }

    function renderMarker() {
      if (!state.draftPoint) {
        els.marker.hidden = true;
        return;
      }
      els.marker.hidden = false;
      positionElement(els.marker, overlayPointToDisplayPoint(state.draftPoint));
    }

    function positionElement(el, point) {
      const scaleX = els.image.clientWidth / Math.max(1, state.imageNatural.width || els.image.naturalWidth);
      const scaleY = els.image.clientHeight / Math.max(1, state.imageNatural.height || els.image.naturalHeight);
      el.style.left = `${point[0] * scaleX}px`;
      el.style.top = `${point[1] * scaleY}px`;
    }

    function updatePointControls() {
      els.point.value = state.draftPoint ? `${state.draftPoint[0].toFixed(2)}, ${state.draftPoint[1].toFixed(2)}` : '';
      if (state.draftPoint && els.status.value === 'unlabeled') {
        els.status.value = 'labeled';
      }
    }

    async function saveCurrent({ advance = false } = {}) {
      const row = selectedRow();
      const status = els.status.value;
      const body = {
        setId: row.setId,
        side: row.side,
        status,
        humanVertexPoint: status === 'labeled' ? state.draftPoint : null,
        labelQuality: els.quality.value || null,
        notes: els.notes.value,
      };
      els.save.disabled = true;
      els.saveStatus.textContent = 'Saving...';
      try {
        const res = await fetch('/api/label', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || 'save failed');
        state.feedback = data.feedback;
        state.evaluation = data.evaluation;
        els.saveStatus.textContent = 'Saved';
        if (advance) state.selected = Math.min(state.feedback.rows.length - 1, state.selected + 1);
        render();
      } catch (err) {
        els.saveStatus.textContent = err.message;
      } finally {
        els.save.disabled = false;
      }
    }

    els.image.addEventListener('load', () => {
      state.imageNatural = { width: els.image.naturalWidth, height: els.image.naturalHeight };
      renderMarker();
      renderCandidateDots();
    });
    els.stage.addEventListener('click', event => {
      const rect = els.image.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) return;
      const x = (event.clientX - rect.left) * state.imageNatural.width / rect.width;
      const y = (event.clientY - rect.top) * state.imageNatural.height / rect.height;
      state.draftPoint = displayPointToOverlayPoint([x, y]);
      els.status.value = 'labeled';
      updatePointControls();
      renderMarker();
      renderCandidates();
    });
    els.imageMode.addEventListener('change', render);
    els.status.addEventListener('change', () => {
      if (els.status.value !== 'labeled') state.draftPoint = null;
      updatePointControls();
      renderMarker();
    });
    els.save.addEventListener('click', () => saveCurrent());
    els.next.addEventListener('click', () => {
      state.selected = Math.min(state.feedback.rows.length - 1, state.selected + 1);
      render();
    });
    els.prev.addEventListener('click', () => {
      state.selected = Math.max(0, state.selected - 1);
      render();
    });
    els.clear.addEventListener('click', () => {
      state.draftPoint = null;
      els.status.value = 'unlabeled';
      updatePointControls();
      renderMarker();
      renderCandidates();
    });
    window.addEventListener('keydown', event => {
      if (event.key === 'ArrowRight') els.next.click();
      if (event.key === 'ArrowLeft') els.prev.click();
      if ((event.metaKey || event.ctrlKey) && event.key === 's') {
        event.preventDefault();
        saveCurrent({ advance: event.shiftKey });
      }
    });
    window.addEventListener('resize', () => {
      renderMarker();
      renderCandidateDots();
    });

    function dist(left, right) {
      return Math.hypot(left[0] - right[0], left[1] - right[1]);
    }
    async function awaitImageSizes(row) {
      await Promise.all([ensureImageSize(row.overlayPath), ensureImageSize(row.imagePath)]);
    }
    function ensureImageSize(path) {
      if (state.imageSizes.has(path)) return Promise.resolve(state.imageSizes.get(path));
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          const size = { width: img.naturalWidth, height: img.naturalHeight };
          state.imageSizes.set(path, size);
          resolve(size);
        };
        img.onerror = reject;
        img.src = `/image?path=${encodeURIComponent(path)}`;
      });
    }
    function overlayPointToDisplayPoint(point) {
      const row = selectedRow();
      if (els.imageMode.value !== 'original') return point;
      const overlay = state.imageSizes.get(row.overlayPath) || state.imageNatural;
      const original = state.imageSizes.get(row.imagePath) || state.imageNatural;
      return [
        point[0] * original.width / Math.max(1, overlay.width),
        point[1] * original.height / Math.max(1, overlay.height),
      ];
    }
    function displayPointToOverlayPoint(point) {
      const row = selectedRow();
      if (els.imageMode.value !== 'original') return point;
      const overlay = state.imageSizes.get(row.overlayPath) || state.imageNatural;
      const original = state.imageSizes.get(row.imagePath) || state.imageNatural;
      return [
        point[0] * overlay.width / Math.max(1, original.width),
        point[1] * overlay.height / Math.max(1, original.height),
      ];
    }
    function boolText(value) {
      if (value === true) return 'yes';
      if (value === false) return 'no';
      return '';
    }
    function formatRate(value) {
      return value === null || value === undefined ? 'n/a' : `${(value * 100).toFixed(1)}%`;
    }
    function pillClass(row) {
      if (row.evaluationStatus === 'labeled' && row.top1WithinThreshold) return 'ok';
      if (row.evaluationStatus === 'labeled') return 'warn';
      if (row.evaluationStatus === 'invalid_label') return 'bad';
      return '';
    }
    loadData();
  </script>
</body>
</html>
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the page in the default browser.")
    args = parser.parse_args(argv)

    if not args.feedback.exists():
        print(f"error: feedback fixture not found: {args.feedback}", file=sys.stderr)
        return 2

    config = ServerConfig(
        feedback_path=args.feedback,
        report_path=args.report,
        host=args.host,
        port=args.port,
    )
    httpd = ThreadingHTTPServer((config.host, config.port), create_handler(config))
    url = f"http://{config.host}:{config.port}/"
    print(f"serving vertex labeler at {url}")
    print(f"feedback: {config.feedback_path}")
    print(f"report: {config.report_path}")
    if args.open:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
