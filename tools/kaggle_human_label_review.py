#!/usr/bin/env python3
"""Run a local human-label review UI for the Kaggle curated eval set.

The curated Kaggle manifest is intentionally only a heuristic starter set.
This tool adds a human-label layer on top:

* ``write-template`` writes a small CSV with one row per curated image.
* ``serve`` starts a localhost-only review UI with hotkeys and writes updates
  back to the CSV through a tiny stdlib HTTP server.

No corpus images or generated labels should be committed unless a future PR
explicitly decides to add a tiny public fixture. The default label path lives
next to the local curated manifest.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import os
import sys
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.kaggle_curated_eval import DEFAULT_OUTPUT_DIR as CURATED_DEFAULT_OUTPUT_DIR  # noqa: E402


DEFAULT_MANIFEST = Path(
    os.environ.get("CTVD_KAGGLE_CURATED_MANIFEST", str(CURATED_DEFAULT_OUTPUT_DIR / "curated_manifest.json"))
)

LABEL_HEADERS = [
    "id",
    "relativePath",
    "keep",
    "role",
    "visible_faces",
    "quality",
    "expected_detector_result",
    "retake_reason",
    "notes",
]

KEEP_CHOICES = ["yes", "no", "maybe"]
VISIBLE_FACE_CHOICES = ["0", "1", "2", "3+"]
QUALITY_CHOICES = ["good", "okay", "bad"]
EXPECTED_DETECTOR_CHOICES = ["should_find_grid", "maybe_find_grid", "should_reject"]
RETAKE_REASON_CHOICES = [
    "none",
    "too_close",
    "cropped",
    "hand_occluded",
    "single_face",
    "bad_angle",
    "blurred",
    "not_cube",
    "other",
]

ROLE_DEFINITIONS: "OrderedDict[str, str]" = OrderedDict(
    [
        (
            "three_face_positive",
            "A cube photo with three visible faces and enough visible stickers that the detector should try to fit cube grids.",
        ),
        (
            "table_isometric",
            "A cube on a table or similar surface with an isometric-ish angle; useful for positive detector generalization even if not canonical.",
        ),
        (
            "single_face_negative",
            "Mostly one visible face or too little adjacent-face context; useful for retake/rejection behavior.",
        ),
        (
            "hand_occluded",
            "A hand, finger, or object visibly blocks meaningful cube stickers or edges; should generally ask for a retake.",
        ),
        (
            "cropped_close",
            "The cube is cut off by the frame or too close to inspect; useful for framing/retake feedback.",
        ),
        (
            "noncanonical_interesting",
            "A real cube image with unusual angle, lighting, background, or partial visibility that is worth keeping for exploratory diagnostics.",
        ),
        (
            "junk",
            "Not useful for this eval loop: not a cube, duplicate junk, extremely blurry/dark, or too ambiguous to label confidently.",
        ),
    ]
)

ROLE_HOTKEYS = {
    "1": "three_face_positive",
    "2": "table_isometric",
    "3": "single_face_negative",
    "4": "hand_occluded",
    "5": "cropped_close",
    "6": "noncanonical_interesting",
    "7": "junk",
}

CATEGORY_ROLE_SUGGESTIONS = {
    "usable_three_face": "three_face_positive",
    "usable_table_isometric": "table_isometric",
    "single_face_negative": "single_face_negative",
    "hand_occluded_negative": "hand_occluded",
    "cropped_close_negative": "cropped_close",
    "noncanonical_interesting": "noncanonical_interesting",
}

EXPECTED_BY_ROLE = {
    "three_face_positive": "should_find_grid",
    "table_isometric": "maybe_find_grid",
    "single_face_negative": "should_reject",
    "hand_occluded": "should_reject",
    "cropped_close": "should_reject",
    "noncanonical_interesting": "maybe_find_grid",
    "junk": "should_reject",
}

RETAKE_BY_ROLE = {
    "three_face_positive": "none",
    "table_isometric": "bad_angle",
    "single_face_negative": "single_face",
    "hand_occluded": "hand_occluded",
    "cropped_close": "cropped",
    "noncanonical_interesting": "other",
    "junk": "other",
}

VISIBLE_FACES_BY_ROLE = {
    "three_face_positive": "3+",
    "single_face_negative": "1",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def default_labels_path(manifest_path: Path) -> Path:
    return manifest_path.parent / "human_labels.csv"


def role_definitions_markdown() -> str:
    lines = ["# Kaggle Human Label Roles", ""]
    for index, (role, definition) in enumerate(ROLE_DEFINITIONS.items(), start=1):
        lines.append(f"{index}. `{role}`: {definition}")
    return "\n".join(lines) + "\n"


def manifest_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = list(manifest.get("images", []))
    missing = [index for index, entry in enumerate(entries, start=1) if not entry.get("id") or not entry.get("relativePath")]
    if missing:
        raise ValueError(f"manifest images missing id/relativePath at rows: {missing}")
    return entries


def default_label_row(entry: Mapping[str, Any]) -> dict[str, str]:
    role = CATEGORY_ROLE_SUGGESTIONS.get(str(entry.get("category", "")), "")
    return {
        "id": str(entry["id"]),
        "relativePath": str(entry["relativePath"]),
        "keep": "maybe",
        "role": role,
        "visible_faces": VISIBLE_FACES_BY_ROLE.get(role, ""),
        "quality": "",
        "expected_detector_result": EXPECTED_BY_ROLE.get(role, ""),
        "retake_reason": RETAKE_BY_ROLE.get(role, ""),
        "notes": "",
    }


def starter_label_rows(manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    return [default_label_row(entry) for entry in manifest_entries(manifest)]


def read_label_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != LABEL_HEADERS:
            raise ValueError(f"{path} must have headers {LABEL_HEADERS}; found {reader.fieldnames}")
        return [{header: row.get(header, "") for header in LABEL_HEADERS} for row in reader]


def merge_label_rows(manifest: Mapping[str, Any], existing_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    existing_by_path = {
        row["relativePath"]: {header: str(row.get(header, "")) for header in LABEL_HEADERS}
        for row in existing_rows
        if row.get("relativePath")
    }
    rows: list[dict[str, str]] = []
    for starter in starter_label_rows(manifest):
        existing = existing_by_path.get(starter["relativePath"])
        if existing:
            existing["id"] = starter["id"]
            existing["relativePath"] = starter["relativePath"]
            rows.append(existing)
        else:
            rows.append(starter)
    return rows


def validate_label_row(row: Mapping[str, str]) -> None:
    _require_choice(row, "keep", KEEP_CHOICES)
    _require_choice(row, "role", list(ROLE_DEFINITIONS))
    _require_choice(row, "visible_faces", VISIBLE_FACE_CHOICES)
    _require_choice(row, "quality", QUALITY_CHOICES)
    _require_choice(row, "expected_detector_result", EXPECTED_DETECTOR_CHOICES)
    _require_choice(row, "retake_reason", RETAKE_REASON_CHOICES)


def _require_choice(row: Mapping[str, str], field: str, choices: Sequence[str]) -> None:
    value = row.get(field, "")
    if value and value not in choices:
        raise ValueError(f"{row.get('id', '<unknown>')} has invalid {field}={value!r}; expected one of {choices}")


def write_label_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        validate_label_row(row)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in LABEL_HEADERS})


def write_template(manifest_path: Path, labels_path: Path, *, overwrite: bool = False) -> list[dict[str, str]]:
    manifest = load_json(manifest_path)
    if labels_path.exists() and not overwrite:
        rows = merge_label_rows(manifest, read_label_csv(labels_path))
    else:
        rows = starter_label_rows(manifest)
    write_label_csv(labels_path, rows)
    return rows


def label_summary(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": len(rows),
        "keep": {},
        "role": {},
        "expected_detector_result": {},
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    for field in ["keep", "role", "expected_detector_result"]:
        counts: dict[str, int] = {}
        for row in rows:
            value = row.get(field, "") or "<blank>"
            counts[value] = counts.get(value, 0) + 1
        summary[field] = dict(sorted(counts.items()))
    return summary


class LabelReviewServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], config: Mapping[str, Any]):
        super().__init__(server_address, LabelReviewHandler)
        self.config = dict(config)
        self.csv_lock = threading.Lock()


class LabelReviewHandler(BaseHTTPRequestHandler):
    server: LabelReviewServer

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            with self.server.csv_lock:
                config = config_with_latest_rows(self.server.config)
            self._send_text(build_review_html(config), content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/labels.csv":
            labels_path = Path(self.server.config["labels_path"])
            with self.server.csv_lock:
                body = labels_path.read_text() if labels_path.exists() else ""
            self._send_text(body, content_type="text/csv; charset=utf-8")
            return
        if parsed.path == "/roles.md":
            self._send_text(role_definitions_markdown(), content_type="text/markdown; charset=utf-8")
            return
        if parsed.path.startswith("/image/"):
            image_id = unquote(parsed.path.removeprefix("/image/"))
            self._send_image(image_id)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/labels":
            self.send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid JSON")
            return
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            self.send_error(HTTPStatus.BAD_REQUEST, "rows must be a list")
            return
        try:
            normalized = [{header: str(row.get(header, "")) for header in LABEL_HEADERS} for row in rows]
            validate_label_payload(normalized, self.server.config["entries"])
            with self.server.csv_lock:
                write_label_csv(Path(self.server.config["labels_path"]), normalized)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_json({"ok": True, "summary": label_summary(normalized)})

    def _send_image(self, image_id: str) -> None:
        entries_by_id = self.server.config["entries_by_id"]
        entry = entries_by_id.get(image_id)
        if not entry:
            self.send_error(HTTPStatus.NOT_FOUND, "unknown image id")
            return
        try:
            image_path = resolve_manifest_image_path(Path(self.server.config["corpus_root"]), entry["relativePath"])
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "path traversal blocked")
            return
        if not image_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, f"image not found: {image_id}")
            return
        content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
        data = image_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, body: str, *, content_type: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Mapping[str, Any]) -> None:
        self._send_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", content_type="application/json; charset=utf-8")


def server_config(manifest_path: Path, labels_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    rows = write_template(manifest_path, labels_path, overwrite=False)
    entries = manifest_entries(manifest)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "labels_path": str(labels_path),
        "corpus_root": str(manifest.get("sourceCorpusRoot", manifest_path.parent)),
        "entries": entries,
        "entries_by_id": {entry["id"]: entry for entry in entries},
        "rows": rows,
        "role_definitions": ROLE_DEFINITIONS,
        "role_hotkeys": ROLE_HOTKEYS,
        "choices": {
            "keep": KEEP_CHOICES,
            "role": list(ROLE_DEFINITIONS),
            "visible_faces": VISIBLE_FACE_CHOICES,
            "quality": QUALITY_CHOICES,
            "expected_detector_result": EXPECTED_DETECTOR_CHOICES,
            "retake_reason": RETAKE_REASON_CHOICES,
        },
    }


def resolve_manifest_image_path(corpus_root: Path, relative_path: str) -> Path:
    root = corpus_root.resolve()
    image_path = (root / relative_path).resolve()
    if not image_path.is_relative_to(root):
        raise ValueError(f"image path escapes corpus root: {relative_path}")
    return image_path


def validate_label_payload(rows: Sequence[Mapping[str, str]], entries: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != len(entries):
        raise ValueError(f"label row count {len(rows)} does not match manifest image count {len(entries)}")
    for index, (row, entry) in enumerate(zip(rows, entries), start=1):
        if row.get("id") != entry.get("id") or row.get("relativePath") != entry.get("relativePath"):
            raise ValueError(f"label row {index} does not match current manifest")


def config_with_latest_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    latest = dict(config)
    latest["rows"] = merge_label_rows(config["manifest"], read_label_csv(Path(str(config["labels_path"]))))
    return latest


def build_review_html(config: Mapping[str, Any]) -> str:
    safe_manifest = html.escape(str(config["manifest_path"]))
    safe_labels = html.escape(str(config["labels_path"]))
    app_payload = {
        "entries": config["entries"],
        "rows": config["rows"],
        "roleDefinitions": dict(config["role_definitions"]),
        "roleHotkeys": config["role_hotkeys"],
        "choices": config["choices"],
        "labelsPath": str(config["labels_path"]),
    }
    payload_json = json.dumps(app_payload).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kaggle Cube Human Label Review</title>
  <style>
    :root {{ color-scheme: light; --ink: #17202a; --muted: #697386; --line: #d7dce2; --accent: #155eef; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: #f7f8fa; }}
    header {{ display: flex; gap: 16px; align-items: center; justify-content: space-between; padding: 12px 16px; border-bottom: 1px solid var(--line); background: white; position: sticky; top: 0; z-index: 2; }}
    button, select, input, textarea {{ font: inherit; }}
    button {{ border: 1px solid var(--line); background: white; border-radius: 6px; padding: 7px 10px; cursor: pointer; }}
    button.primary {{ background: var(--accent); color: white; border-color: var(--accent); }}
    button.active {{ border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }}
    main {{ display: grid; grid-template-columns: minmax(420px, 1fr) 420px; gap: 16px; padding: 16px; }}
    .image-panel, .control-panel {{ background: white; border: 1px solid var(--line); border-radius: 8px; min-width: 0; }}
    .image-wrap {{ display: flex; align-items: center; justify-content: center; height: calc(100vh - 150px); min-height: 420px; padding: 12px; background: #111827; border-radius: 8px 8px 0 0; }}
    img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
    .meta {{ padding: 12px; display: grid; gap: 6px; color: var(--muted); font-size: 13px; }}
    .control-panel {{ padding: 14px; display: grid; gap: 16px; align-self: start; }}
    .field {{ display: grid; gap: 6px; }}
    .label {{ font-weight: 700; font-size: 13px; }}
    .choices {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    textarea {{ width: 100%; min-height: 72px; border: 1px solid var(--line); border-radius: 6px; padding: 8px; }}
    .role-defs {{ border-top: 1px solid var(--line); padding-top: 12px; display: grid; gap: 8px; font-size: 13px; }}
    .role-defs div {{ line-height: 1.35; }}
    .status {{ color: var(--muted); font-size: 13px; }}
    .progress {{ font-weight: 700; }}
    .kbd {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: #eef1f5; border: 1px solid #d3d8df; border-radius: 4px; padding: 1px 4px; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} .image-wrap {{ height: 52vh; }} }}
  </style>
</head>
<body>
  <header>
    <div>
      <div class="progress" id="progress"></div>
      <div class="status">Manifest: {safe_manifest}<br>Labels: {safe_labels}</div>
    </div>
    <div class="choices">
      <button id="prev">Previous <span class="kbd">k</span></button>
      <button id="next">Next <span class="kbd">j</span></button>
      <button class="primary" id="save">Save <span class="kbd">s</span></button>
      <a href="/labels.csv"><button type="button">Download CSV</button></a>
    </div>
  </header>
  <main>
    <section class="image-panel">
      <div class="image-wrap"><img id="image" alt=""></div>
      <div class="meta" id="meta"></div>
    </section>
    <section class="control-panel">
      <div class="field"><div class="label">Keep</div><div class="choices" data-field="keep"></div></div>
      <div class="field"><div class="label">Role <span class="status">(1-7)</span></div><div class="choices" data-field="role"></div></div>
      <div class="field"><div class="label">Visible Faces</div><div class="choices" data-field="visible_faces"></div></div>
      <div class="field"><div class="label">Quality</div><div class="choices" data-field="quality"></div></div>
      <div class="field"><div class="label">Expected Detector Result</div><div class="choices" data-field="expected_detector_result"></div></div>
      <div class="field"><div class="label">Retake Reason</div><div class="choices" data-field="retake_reason"></div></div>
      <div class="field"><div class="label">Notes</div><textarea id="notes" placeholder="Optional. Keep this short."></textarea></div>
      <div class="status" id="status"></div>
      <div class="role-defs" id="role-defs"></div>
    </section>
  </main>
  <script id="payload" type="application/json">{payload_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    const entries = payload.entries;
    const labels = new Map(payload.rows.map(row => [row.id, {{...row}}]));
    let index = 0;
    const status = document.getElementById('status');

    function row() {{
      const entry = entries[index];
      if (!labels.has(entry.id)) {{
        labels.set(entry.id, {{id: entry.id, relativePath: entry.relativePath, keep: 'maybe', role: '', visible_faces: '', quality: '', expected_detector_result: '', retake_reason: '', notes: ''}});
      }}
      return labels.get(entry.id);
    }}

    function renderButtons(field) {{
      const container = document.querySelector(`[data-field="${{field}}"]`);
      container.innerHTML = '';
      for (const choice of payload.choices[field]) {{
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = choice;
        button.dataset.value = choice;
        button.addEventListener('click', () => setField(field, choice));
        container.appendChild(button);
      }}
    }}

    for (const field of Object.keys(payload.choices)) renderButtons(field);

    function renderRoleDefs() {{
      const container = document.getElementById('role-defs');
      container.replaceChildren();
      const heading = document.createElement('div');
      heading.className = 'label';
      heading.textContent = 'Role definitions';
      container.appendChild(heading);
      let n = 1;
      for (const [role, definition] of Object.entries(payload.roleDefinitions)) {{
        const div = document.createElement('div');
        const key = document.createElement('span');
        key.className = 'kbd';
        key.textContent = String(n);
        const strong = document.createElement('strong');
        strong.textContent = ` ${{role}}`;
        div.append(key, strong, document.createTextNode(`: ${{definition}}`));
        container.appendChild(div);
        n += 1;
      }}
    }}

    function appendMetaLine(container, label, value, strong=false) {{
      const div = document.createElement('div');
      if (strong) {{
        const node = document.createElement('strong');
        node.textContent = value;
        div.appendChild(node);
      }} else {{
        div.textContent = `${{label}}: ${{value || ''}}`;
      }}
      container.appendChild(div);
    }}

    function render() {{
      const entry = entries[index];
      const current = row();
      document.getElementById('progress').textContent = `${{entry.id}} (${{index + 1}}/${{entries.length}})`;
      document.getElementById('image').src = `/image/${{encodeURIComponent(entry.id)}}`;
      document.getElementById('image').alt = entry.relativePath;
      const meta = document.getElementById('meta');
      meta.replaceChildren();
      appendMetaLine(meta, '', entry.relativePath, true);
      appendMetaLine(meta, 'category', entry.category || '');
      appendMetaLine(meta, 'source label', entry.sourceLabel || '');
      appendMetaLine(meta, 'tags', (entry.tags || []).join(', '));
      document.getElementById('notes').value = current.notes || '';
      for (const field of Object.keys(payload.choices)) {{
        for (const button of document.querySelectorAll(`[data-field="${{field}}"] button`)) {{
          button.classList.toggle('active', button.dataset.value === (current[field] || ''));
        }}
      }}
      status.textContent = '';
    }}

    function setField(field, value) {{
      row()[field] = value;
      if (field === 'role') applyRoleDefaults(value);
      render();
    }}

    function applyRoleDefaults(role) {{
      const current = row();
      const expected = {{
        three_face_positive: ['should_find_grid', 'none', '3+'],
        table_isometric: ['maybe_find_grid', 'bad_angle', ''],
        single_face_negative: ['should_reject', 'single_face', '1'],
        hand_occluded: ['should_reject', 'hand_occluded', ''],
        cropped_close: ['should_reject', 'cropped', ''],
        noncanonical_interesting: ['maybe_find_grid', 'other', ''],
        junk: ['should_reject', 'other', '0']
      }}[role];
      if (expected) {{
        current.expected_detector_result = expected[0];
        current.retake_reason = expected[1];
        current.visible_faces = expected[2];
      }}
    }}

    function move(delta) {{
      row().notes = document.getElementById('notes').value;
      index = Math.max(0, Math.min(entries.length - 1, index + delta));
      render();
    }}

    async function save() {{
      row().notes = document.getElementById('notes').value;
      const rows = entries.map(entry => labels.get(entry.id));
      const response = await fetch('/labels', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{rows}})
      }});
      if (!response.ok) {{
        status.textContent = `Save failed: ${{await response.text()}}`;
        return;
      }}
      const payload = await response.json();
      status.textContent = `Saved ${{payload.summary.total}} rows.`;
    }}

    document.getElementById('prev').addEventListener('click', () => move(-1));
    document.getElementById('next').addEventListener('click', () => move(1));
    document.getElementById('save').addEventListener('click', save);
    document.getElementById('notes').addEventListener('input', event => {{ row().notes = event.target.value; }});
    document.addEventListener('keydown', event => {{
      if (event.target.tagName === 'TEXTAREA') {{
        if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {{ event.preventDefault(); save(); }}
        return;
      }}
      const key = event.key.toLowerCase();
      if (key === 'j') move(1);
      else if (key === 'k') move(-1);
      else if (key === 's') save();
      else if (key === 'y') setField('keep', 'yes');
      else if (key === 'n') setField('keep', 'no');
      else if (key === 'm') setField('keep', 'maybe');
      else if (payload.roleHotkeys[event.key]) setField('role', payload.roleHotkeys[event.key]);
    }});

    renderRoleDefs();
    render();
  </script>
</body>
</html>
"""


def _resolve_labels_path(raw_labels: Path | None, manifest_path: Path) -> Path:
    return raw_labels if raw_labels is not None else default_labels_path(manifest_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    roles = subparsers.add_parser("roles", help="Print role definitions.")
    roles.add_argument("--json", action="store_true", help="Print definitions as JSON instead of Markdown.")

    template = subparsers.add_parser("write-template", help="Write or refresh a human_labels.csv template.")
    template.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    template.add_argument("--labels", type=Path)
    template.add_argument("--overwrite", action="store_true", help="Overwrite existing labels instead of preserving edits.")

    serve = subparsers.add_parser("serve", help="Start the localhost review UI and write labels to CSV.")
    serve.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    serve.add_argument("--labels", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "roles":
        if args.json:
            print(json.dumps(dict(ROLE_DEFINITIONS), indent=2, sort_keys=True))
        else:
            print(role_definitions_markdown(), end="")
        return 0

    if args.command == "write-template":
        labels_path = _resolve_labels_path(args.labels, args.manifest)
        rows = write_template(args.manifest, labels_path, overwrite=args.overwrite)
        print(f"wrote {labels_path}")
        print(f"rows: {len(rows)}")
        print(f"summary: {label_summary(rows)}")
        return 0

    if args.command == "serve":
        labels_path = _resolve_labels_path(args.labels, args.manifest)
        config = server_config(args.manifest, labels_path)
        server = LabelReviewServer((args.host, args.port), config)
        print(f"serving http://{args.host}:{args.port}/")
        print(f"manifest: {args.manifest}")
        print(f"labels: {labels_path}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down")
        return 0

    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
