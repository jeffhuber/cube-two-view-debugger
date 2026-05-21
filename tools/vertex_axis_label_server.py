#!/usr/bin/env python3
"""Local click-to-label UI for visible trihedral vertex + axes.

Diagnostics/data-only helper. Starts a tiny localhost server that lets a
human mark the true vertex and three outgoing cube-edge ray endpoints in
``tests/fixtures/vertex_axis_human_feedback_v0.json``.
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

from tools.vertex_axis_feedback import (  # noqa: E402
    DEFAULT_FEEDBACK,
    DEFAULT_REPORT,
    evaluate_feedback,
    render_report,
    update_feedback_row,
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


def create_handler(config: ServerConfig) -> type[BaseHTTPRequestHandler]:
    class VertexAxisLabelHandler(BaseHTTPRequestHandler):
        server_version = "VertexAxisLabelServer/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_text(INDEX_HTML, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/feedback":
                feedback = load_feedback(config.feedback_path)
                self._send_json(
                    {
                        "feedback": feedback,
                        "evaluation": evaluate_feedback(feedback),
                        "feedbackPath": str(config.feedback_path),
                        "reportPath": str(config.report_path),
                    }
                )
                return
            if parsed.path == "/image":
                raw_path = (parse_qs(parsed.query).get("path") or [""])[0]
                self._send_file(raw_path)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path == "/api/label":
                self._handle_label()
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}", file=sys.stderr)

        def _handle_label(self) -> None:
            try:
                payload = self._read_json_body()
                feedback = load_feedback(config.feedback_path)
                update_feedback_row(
                    feedback,
                    key=str(payload.get("key")),
                    status=str(payload.get("status") or "unlabeled"),
                    human_vertex_point=_parse_point(payload.get("humanVertexPoint")),
                    human_axis_endpoints=[
                        _parse_point(point)
                        for point in payload.get("humanAxisEndpoints", [])
                    ],
                    axis_label_quality=payload.get("axisLabelQuality"),
                    notes=str(payload.get("notes") or ""),
                )
                evaluation = save_feedback(config.feedback_path, config.report_path, feedback)
                self._send_json({"ok": True, "feedback": feedback, "evaluation": evaluation})
            except Exception as exc:
                self._send_json({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}, status=400)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            return json.loads(self.rfile.read(length).decode("utf-8"))

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

    return VertexAxisLabelHandler


def _parse_point(value: Any) -> Optional[Point]:
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return float(value[0]), float(value[1])
    return None


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trihedral Axis Labeler</title>
  <style>
    body { margin: 0; font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; background: #f5f6f8; }
    .app { min-height: 100vh; display: grid; grid-template-columns: 280px minmax(0, 1fr) 320px; }
    aside, .panel { background: #fff; border-color: #d7dde7; overflow: auto; max-height: 100vh; }
    aside { border-right: 1px solid #d7dde7; padding: 12px; }
    .panel { border-left: 1px solid #d7dde7; padding: 12px; }
    .stage { display: grid; place-items: center; min-height: 100vh; padding: 16px; overflow: auto; }
    .canvas-wrap { position: relative; background: #111; box-shadow: 0 4px 18px rgba(0,0,0,.18); }
    canvas { display: block; max-width: calc(100vw - 640px); max-height: calc(100vh - 32px); cursor: crosshair; }
    button, select, textarea { font: inherit; width: 100%; margin: 4px 0; }
    button { border: 1px solid #b7c1cf; background: #fff; padding: 8px; border-radius: 6px; cursor: pointer; }
    button.primary { background: #0f766e; color: white; border-color: #0f766e; }
    button.danger { color: #b3261e; }
    .row { padding: 8px; border: 1px solid #d7dde7; border-radius: 6px; margin-bottom: 6px; cursor: pointer; }
    .row.active { border-color: #0f766e; background: #edf8f6; }
    .muted { color: #647080; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .legend span { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h2>Rows</h2>
      <div id="summary" class="muted"></div>
      <div id="rows"></div>
    </aside>
    <main class="stage"><div class="canvas-wrap"><canvas id="canvas"></canvas></div></main>
    <section class="panel">
      <h2 id="title">Trihedral</h2>
      <div class="legend">
        <p><span style="background:#ff3b30"></span>vertex</p>
        <p><span style="background:#0a84ff"></span>axis endpoint 1</p>
        <p><span style="background:#34c759"></span>axis endpoint 2</p>
        <p><span style="background:#ffcc00"></span>axis endpoint 3</p>
      </div>
      <p class="muted">Click sequence: vertex, axis endpoint, axis endpoint, axis endpoint. Axis order does not matter.</p>
      <select id="status">
        <option value="vertex_labeled_axes_unlabeled">vertex only</option>
        <option value="labeled">labeled</option>
        <option value="ambiguous">ambiguous</option>
        <option value="not_visible">not visible</option>
        <option value="unlabeled">unlabeled</option>
      </select>
      <select id="quality">
        <option value="high">high</option>
        <option value="medium">medium</option>
        <option value="low">low</option>
      </select>
      <textarea id="notes" rows="5" placeholder="notes"></textarea>
      <button class="primary" id="save">Save</button>
      <button id="undo">Undo last point</button>
      <button class="danger" id="clear">Clear points</button>
      <pre id="debug" class="mono"></pre>
    </section>
  </div>
<script>
let payload, rows = [], activeIndex = 0, image = new Image(), scale = 1, points = [];
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

async function load() {
  payload = await (await fetch('/api/feedback')).json();
  rows = payload.feedback.rows || [];
  document.getElementById('summary').textContent =
    `${payload.evaluation.summary.trihedralLabeledRowCount}/${payload.evaluation.summary.rowCount} full labels`;
  renderRows();
  selectRow(activeIndex);
}

function renderRows() {
  const root = document.getElementById('rows');
  root.innerHTML = '';
  rows.forEach((row, i) => {
    const div = document.createElement('div');
    div.className = 'row' + (i === activeIndex ? ' active' : '');
    div.innerHTML = `<strong>${row.key}</strong><br><span class="muted">${row.status}</span>`;
    div.onclick = () => selectRow(i);
    root.appendChild(div);
  });
}

function selectRow(i) {
  activeIndex = Math.max(0, Math.min(rows.length - 1, i));
  const row = rows[activeIndex];
  document.getElementById('title').textContent = row.key;
  document.getElementById('status').value = row.status || 'unlabeled';
  document.getElementById('quality').value = row.axisLabelQuality || 'high';
  document.getElementById('notes').value = row.notes || '';
  points = [];
  if (row.humanVertexPoint) points.push(row.humanVertexPoint);
  (row.humanAxisEndpoints || []).forEach(p => { if (p) points.push(p); });
  image = new Image();
  image.onload = draw;
  image.src = '/image?path=' + encodeURIComponent(row.imagePath || '');
  renderRows();
}

function draw() {
  const maxW = Math.max(500, window.innerWidth - 640);
  const maxH = window.innerHeight - 32;
  scale = Math.min(1, maxW / image.naturalWidth, maxH / image.naturalHeight);
  canvas.width = image.naturalWidth * scale;
  canvas.height = image.naturalHeight * scale;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  const colors = ['#ff3b30', '#0a84ff', '#34c759', '#ffcc00'];
  if (points.length > 0) {
    for (let i = 1; i < points.length; i++) line(points[0], points[i], colors[i]);
  }
  points.forEach((p, i) => dot(p, colors[i], i === 0 ? 8 : 6));
  const m = rows[activeIndex].currentModel;
  if (m && m.vertexPoint) {
    dot(m.vertexPoint, '#fff', 5);
    (m.axes || []).forEach(a => {
      if (a.endpoint) line(m.vertexPoint, a.endpoint, 'rgba(255,255,255,.65)');
    });
  }
  document.getElementById('debug').textContent = JSON.stringify({points}, null, 2);
}

function dot(p, color, r) {
  ctx.beginPath();
  ctx.arc(p[0] * scale, p[1] * scale, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = '#111';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function line(a, b, color) {
  ctx.beginPath();
  ctx.moveTo(a[0] * scale, a[1] * scale);
  ctx.lineTo(b[0] * scale, b[1] * scale);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.stroke();
}

canvas.onclick = (event) => {
  if (points.length >= 4) points = [];
  const rect = canvas.getBoundingClientRect();
  points.push([(event.clientX - rect.left) / scale, (event.clientY - rect.top) / scale]);
  if (points.length === 4) document.getElementById('status').value = 'labeled';
  draw();
};

document.getElementById('undo').onclick = () => { points.pop(); draw(); };
document.getElementById('clear').onclick = () => { points = []; draw(); };
document.getElementById('save').onclick = async () => {
  const row = rows[activeIndex];
  const response = await fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      key: row.key,
      status: document.getElementById('status').value,
      humanVertexPoint: points[0] || null,
      humanAxisEndpoints: [points[1] || null, points[2] || null, points[3] || null],
      axisLabelQuality: document.getElementById('quality').value,
      notes: document.getElementById('notes').value
    })
  });
  const result = await response.json();
  if (!result.ok) { alert(result.error); return; }
  payload = result;
  rows = result.feedback.rows || rows;
  activeIndex = Math.min(activeIndex + 1, rows.length - 1);
  selectRow(activeIndex);
};

load();
</script>
</body>
</html>
"""


def run_server(config: ServerConfig, *, open_browser: bool) -> None:
    handler = create_handler(config)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    url = f"http://{config.host}:{config.port}/"
    print(f"vertex+axis labeler: {url}")
    print(f"feedback: {config.feedback_path}")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8778)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)
    run_server(
        ServerConfig(
            feedback_path=args.feedback,
            report_path=args.report,
            host=args.host,
            port=args.port,
        ),
        open_browser=args.open,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
