from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import html
import io
import json
import mimetypes
import re
import subprocess
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote

import numpy as _np  # noqa: F401  -- needed for runtime diag block
import PIL as _PIL   # noqa: F401  -- ditto

from rubik_recognizer.dataset import (
    ImagePair,
    ImageUpload,
    evaluate_state,
    load_image_uploads_from_dir,
    normalize_set_id,
    pair_image_uploads,
    parse_ground_truth,
    parse_manifest_pairs,
)
from rubik_recognizer.recognizer import WhiteUpRecognizer, recognition_diagnostics


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
RUNS = ROOT / "runs"


# Origins permitted to call the recognizer cross-origin. The frontend
# integration we care about is cube-snap (jeffhuber.github.io and the
# Vite dev server). 127.0.0.1 is included for parity with localhost
# since browsers treat them as different origins.
_ALLOWED_ORIGIN_PATTERNS = (
    re.compile(r"^https://jeffhuber\.github\.io$"),
    re.compile(r"^http://localhost(?::\d+)?$"),
    re.compile(r"^http://127\.0\.0\.1(?::\d+)?$"),
)


def _origin_is_allowed(origin: Optional[str]) -> bool:
    if not origin:
        return False
    return any(pattern.match(origin) for pattern in _ALLOWED_ORIGIN_PATTERNS)


class RubikHandler(BaseHTTPRequestHandler):
    recognizer = WhiteUpRecognizer()

    def end_headers(self) -> None:
        # Inject CORS headers on every response, regardless of how it was
        # produced (our _send_json/_send_file paths AND the built-in
        # send_error path). Echo back the Origin if it's on the
        # allowlist; otherwise omit (browser will block, which is what
        # we want from an unknown origin).
        origin = self.headers.get("Origin") if hasattr(self, "headers") else None
        if _origin_is_allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Vary", "Origin")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        # Preflight. The CORS headers attached in end_headers() are what
        # the browser actually inspects; the body is empty.
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_file(STATIC / "index.html")
            return
        if path.startswith("/static/"):
            self._send_file(STATIC / path.removeprefix("/static/"))
            return
        if path == "/api/runs":
            self._send_json({"runs": list_saved_runs()})
            return
        if path == "/api/routes":
            # Self-describing route list so any agent (Claude / Codex /
            # human picking the project up cold) can discover the API
            # surface without grep-ing app.py. Kept short and accurate;
            # add new routes here whenever you add to dispatch.
            self._send_json({"routes": _api_routes()})
            return
        if path == "/api/diag":
            # Reports the runtime stack the recognizer is actually running
            # under: Python version, key library versions, git SHA, working
            # directory. Was added because subtle PIL / NumPy version drift
            # silently degrades recognition accuracy by ~20 stickers on
            # hard-lighting images, and the only way to debug that
            # post-hoc is to know what was running. See README "Pinned
            # dependencies" section.
            self._send_json(_runtime_diag())
            return
        if path.startswith("/runs/"):
            self._send_run_file(path)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/recognize":
            self._handle_recognize()
            return
        if path == "/api/recognize-batch":
            self._handle_batch()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _handle_recognize(self) -> None:
        try:
            fields = self._read_multipart()
            image_a = _first_field(fields, "imageA")
            image_b = _first_field(fields, "imageB")
            expected = _text_field(fields, "expectedState")
            set_id = _text_field(fields, "setId")
            # `?slim=1` returns a stripped-down payload with the heavy
            # debug fields (overlays + diagnostics) omitted. Used by
            # the cube-snap Fixer integration where the response is
            # several MB without it (mostly base64-encoded overlay
            # PNGs the embedded UI never consumes), and where the
            # browser tends to drop the connection on the long
            # download. Default unchanged so the debugger's own
            # static UI keeps the rich response.
            slim = self._query_flag("slim")
            if not image_a or not image_b:
                self._send_json(
                    {
                        "status": "rejected",
                        "reason": "Upload both image A and image B.",
                        "failedChecks": ["missing_upload"],
                    },
                    HTTPStatus.BAD_REQUEST,
                )
                return

            payload = recognize_and_persist(
                self.recognizer,
                ImagePair(
                    set_id=set_id or normalize_set_id(image_a[0] + " " + image_b[0]),
                    image_a=ImageUpload(image_a[0], image_a[1]),
                    image_b=ImageUpload(image_b[0], image_b[1]),
                ),
                expected_state=expected,
            )
            # Attach a per-request runtime block (input-image SHA256 +
            # byte size + decoded dimensions, and the same env info
            # /api/diag returns). When a saved recognition is later
            # questioned ("why did this score 26/54 instead of 50?"),
            # the JSON itself answers what env produced it.
            payload["runtime"] = _per_request_runtime(image_a[1], image_b[1])
            if slim:
                _strip_heavy_fields(payload)
            self._send_json(payload)
        except Exception as exc:  # Defensive API boundary for local debugging.
            # Print the full traceback to stderr so the operator running
            # the server can diagnose immediately. The HTTP response only
            # carries str(exc) (no traceback) to avoid leaking filesystem
            # paths to clients in case the server is ever exposed beyond
            # localhost. For local debugging the terminal log is the
            # primary surface.
            traceback.print_exc(file=sys.stderr)
            self._send_json(
                {
                    "status": "rejected",
                    "reason": "Recognizer failed before producing a cube state.",
                    "failedChecks": ["internal_error"],
                    "detail": str(exc),
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_batch(self) -> None:
        try:
            fields = self._read_multipart()
            uploads = [ImageUpload(filename, data) for filename, data in fields.get("images", []) if filename]
            truth_upload = _first_field(fields, "groundTruth")
            ground_truth = parse_ground_truth(truth_upload[1], truth_upload[0]) if truth_upload else {}
            pairs, unpaired = pair_image_uploads(uploads)
            batch = run_batch(self.recognizer, pairs, ground_truth, unpaired)
            self._send_json(batch)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            self._send_json(
                {
                    "status": "rejected",
                    "reason": "Batch recognizer failed before producing results.",
                    "failedChecks": ["internal_error"],
                    "detail": str(exc),
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _query_flag(self, name: str) -> bool:
        """Return True if `name` appears in the URL query string with a
        truthy value ("1", "true", "yes" — case-insensitive). Bare
        presence (`?slim`) also counts as true.
        """
        if "?" not in self.path:
            return False
        query = self.path.split("?", 1)[1]
        for pair in query.split("&"):
            if not pair:
                continue
            key, _, value = pair.partition("=")
            if key.lower() != name.lower():
                continue
            if value == "":
                return True
            return value.lower() in ("1", "true", "yes")
        return False

    def _send_run_file(self, request_path: str) -> None:
        relative = Path(unquote(request_path.removeprefix("/runs/")))
        path = (RUNS / relative).resolve()
        if RUNS.resolve() not in path.parents and path != RUNS.resolve():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(path)

    def _read_multipart(self) -> Dict[str, List[Tuple[str, bytes]]]:
        content_type = self.headers.get("Content-Type", "")
        marker = "boundary="
        if marker not in content_type:
            raise ValueError("Expected multipart/form-data.")

        boundary = content_type.split(marker, 1)[1].strip().strip('"')
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        delimiter = ("--" + boundary).encode("utf-8")
        fields: Dict[str, List[Tuple[str, bytes]]] = {}

        for chunk in raw.split(delimiter):
            chunk = chunk.strip()
            if not chunk or chunk == b"--":
                continue
            if chunk.endswith(b"--"):
                chunk = chunk[:-2].strip()
            header_blob, sep, body = chunk.partition(b"\r\n\r\n")
            if not sep:
                continue

            headers = header_blob.decode("utf-8", errors="replace").split("\r\n")
            disposition = next((h for h in headers if h.lower().startswith("content-disposition:")), "")
            params: Dict[str, str] = {}
            for item in disposition.split(";"):
                if "=" in item:
                    key, value = item.strip().split("=", 1)
                    params[key] = value.strip('"')
            name = params.get("name")
            filename = params.get("filename", name or "upload")
            if name:
                fields.setdefault(name, []).append((filename, body.rstrip(b"\r\n")))
        return fields

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        print("[rubik-app] " + fmt % args)


# ---------------------------------------------------------------------------
# Runtime diagnostics
#
# The recognizer's output is dependency-sensitive: PIL's libjpeg-backed
# decode and numpy's BLAS path can shift sticker classification by 20+
# stickers on hard-lighting images, with the same source code and same
# Python major version. So every API caller needs an easy way to ask
# "what stack is this server actually running?". Two surfaces:
#
#   GET /api/diag                       → standalone diagnostics
#   POST /api/recognize → response.runtime  → diagnostics for THIS call,
#                                              including image fingerprint
#
# Both share _runtime_diag(); _per_request_runtime() adds image fingerprints.

# Pinned in requirements.txt; we softly enforce here so a degraded env
# is loud rather than silent.
_MIN_PIL = (12, 2)
_MIN_NUMPY = (2, 3, 5)
_MIN_PYTHON = (3, 11)


def _version_tuple(s: str) -> Tuple[int, ...]:
    """Parse "12.2.0" / "2.3.5" / "12.2" into an int tuple. Anything
    non-numeric in a part (e.g. "1.2.0a1") is dropped."""
    out: List[int] = []
    for part in s.split("."):
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def _git_sha() -> Optional[str]:
    """Short git SHA of the recognizer's working tree, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _runtime_diag() -> Dict[str, Any]:
    """Stable JSON-serialisable description of the recognizer's runtime."""
    return {
        "python": {
            "version": ".".join(str(p) for p in sys.version_info[:3]),
            "executable": sys.executable,
            "implementation": sys.implementation.name,
        },
        "libraries": {
            "numpy": _np.__version__,
            "pillow": _PIL.__version__,
        },
        "minimums": {
            "python": ".".join(str(p) for p in _MIN_PYTHON),
            "numpy": ".".join(str(p) for p in _MIN_NUMPY),
            "pillow": ".".join(str(p) for p in _MIN_PIL),
        },
        "git": {
            "sha": _git_sha(),
            "cwd": str(ROOT),
        },
    }


def _api_routes() -> List[Dict[str, str]]:
    """Static list of public HTTP routes the server exposes. Update
    when adding/removing routes in do_GET/do_POST dispatch above.

    Intended for cold-start discovery by anyone (human or agent)
    picking the project up without prior context. The brief column
    explains the route's purpose, not its full schema — for that read
    the corresponding handler.
    """
    return [
        {"method": "GET",  "path": "/",                    "brief": "Static UI (index.html)."},
        {"method": "GET",  "path": "/static/*",            "brief": "Static assets (CSS, JS)."},
        {"method": "GET",  "path": "/api/routes",          "brief": "This route list."},
        {"method": "GET",  "path": "/api/diag",            "brief": "Runtime environment fingerprint (Python/NumPy/Pillow versions, git SHA)."},
        {"method": "GET",  "path": "/api/runs",            "brief": "List of recent saved recognition runs (per-pair summaries)."},
        {"method": "GET",  "path": "/runs/pairs/<id>/...", "brief": "Static access to a saved run's files (result.json, debug.json, overlays, samples.csv, original photos)."},
        {"method": "POST", "path": "/api/recognize",       "brief": "Recognize one pair. Multipart fields: imageA, imageB; optional setId, expectedState. Query: ?slim=1 to omit overlays/diagnostics. Persists a run under /runs/pairs/<id>/."},
        {"method": "POST", "path": "/api/recognize-batch", "brief": "Recognize multiple pairs in one call. Multipart field: images (multi-file); optional groundTruth (.csv/.tsv/.json). Pairs files by filename A/B markers or by drop order. Persists a batch under /runs/batches/<id>/."},
    ]


def _image_fingerprint(image_bytes: bytes) -> Dict[str, Any]:
    """SHA256 + size + decoded dims of an uploaded image. Lets a saved
    recognition be tied back to its exact input, even after the source
    file changes on disk."""
    fp: Dict[str, Any] = {
        "bytes": len(image_bytes),
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
    }
    try:
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as img:
            fp["decodedWidth"] = img.width
            fp["decodedHeight"] = img.height
            if img.format:
                fp["format"] = img.format
    except Exception as exc:  # Don't fail the recognition over diag.
        fp["decodeError"] = str(exc)
    return fp


def _per_request_runtime(image_a_bytes: bytes, image_b_bytes: bytes) -> Dict[str, Any]:
    """Same env block as /api/diag, plus per-image fingerprint metadata."""
    return {
        **_runtime_diag(),
        "imageA": _image_fingerprint(image_a_bytes),
        "imageB": _image_fingerprint(image_b_bytes),
    }


def _check_runtime_versions() -> List[str]:
    """Compare current Python/PIL/numpy versions to the pinned minimums.
    Returns a list of human-readable warnings (empty if all minimums met).
    Does NOT abort startup — degraded envs still work, just badly. The
    operator decides whether to upgrade."""
    warnings: List[str] = []
    py = sys.version_info[:3]
    if py < _MIN_PYTHON:
        warnings.append(
            f"Python {'.'.join(str(p) for p in py)} is below the recommended "
            f"{'.'.join(str(p) for p in _MIN_PYTHON)} — recognizer accuracy "
            "will silently degrade by ~20 stickers on hard-lighting images."
        )
    pil_v = _version_tuple(_PIL.__version__)
    if pil_v and pil_v < _MIN_PIL:
        warnings.append(
            f"Pillow {_PIL.__version__} is below the recommended "
            f"{'.'.join(str(p) for p in _MIN_PIL)} — see requirements.txt."
        )
    np_v = _version_tuple(_np.__version__)
    if np_v and np_v < _MIN_NUMPY:
        warnings.append(
            f"numpy {_np.__version__} is below the recommended "
            f"{'.'.join(str(p) for p in _MIN_NUMPY)} — see requirements.txt."
        )
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--analyze", nargs=2, metavar=("IMAGE_A", "IMAGE_B"))
    parser.add_argument("--batch", metavar="IMAGE_DIR")
    parser.add_argument("--batch-manifest", metavar="CSV_OR_TSV")
    parser.add_argument("--ground-truth", metavar="CSV_TSV_OR_JSON")
    args = parser.parse_args()

    if args.analyze:
        pair = ImagePair(
            set_id=normalize_set_id(Path(args.analyze[0]).name + " " + Path(args.analyze[1]).name),
            image_a=ImageUpload(Path(args.analyze[0]).name, Path(args.analyze[0]).read_bytes()),
            image_b=ImageUpload(Path(args.analyze[1]).name, Path(args.analyze[1]).read_bytes()),
        )
        payload = recognize_and_persist(WhiteUpRecognizer(), pair)
        payload.pop("overlays", None)
        print(json.dumps(payload, indent=2))
        return

    if args.batch or args.batch_manifest:
        ground_truth: Dict[str, str] = {}
        if args.batch_manifest:
            pairs, manifest_truth = parse_manifest_pairs(Path(args.batch_manifest).expanduser())
            ground_truth.update(manifest_truth)
            unpaired: List[str] = []
        else:
            uploads = load_image_uploads_from_dir(Path(args.batch).expanduser())
            pairs, unpaired = pair_image_uploads(uploads)
        if args.ground_truth:
            ground_truth.update(parse_ground_truth(Path(args.ground_truth).expanduser().read_bytes(), Path(args.ground_truth).name))
        print(json.dumps(run_batch(WhiteUpRecognizer(), pairs, ground_truth, unpaired), indent=2))
        return

    # Loud warning at startup if Python/PIL/numpy are below the pinned
    # minimums. These differences silently degrade recognition accuracy
    # — see requirements.txt commentary for the measured impact.
    for warning in _check_runtime_versions():
        print(f"[rubik-app] WARNING: {warning}", file=sys.stderr)

    server = ThreadingHTTPServer((args.host, args.port), RubikHandler)
    diag = _runtime_diag()
    print(
        f"Serving white-up Rubik recognizer at http://{args.host}:{args.port}/  "
        f"[python {diag['python']['version']}, "
        f"pillow {diag['libraries']['pillow']}, "
        f"numpy {diag['libraries']['numpy']}"
        + (f", git {diag['git']['sha']}" if diag['git']['sha'] else "")
        + "]"
    )
    server.serve_forever()


def _first_field(fields: Dict[str, List[Tuple[str, bytes]]], name: str) -> Optional[Tuple[str, bytes]]:
    values = fields.get(name) or []
    return values[0] if values else None


def _text_field(fields: Dict[str, List[Tuple[str, bytes]]], name: str) -> Optional[str]:
    value = _first_field(fields, name)
    if not value:
        return None
    return value[1].decode("utf-8", errors="replace").strip() or None


def recognize_and_persist(recognizer: WhiteUpRecognizer, pair: ImagePair, expected_state: Optional[str] = None) -> Dict:
    result = recognizer.recognize(pair.image_a.data, pair.image_b.data)
    payload = result.to_api_dict()
    if result.image_a and result.image_b:
        payload["diagnostics"] = recognition_diagnostics(result.image_a, result.image_b)
    evaluation = evaluate_state(result.state, expected_state)
    if evaluation.get("available"):
        payload["evaluation"] = evaluation
    run_info = save_run(pair, payload, result, expected_state)
    payload.update(run_info)
    return payload


def _strip_heavy_fields(payload: Dict) -> None:
    """Remove the bulkiest debug fields in place. Used by the slim API
    response mode (`?slim=1` on /api/recognize). The on-disk run files
    written by save_run() are NOT affected — only what's returned to
    the HTTP client is reduced.

    Removes:
      - overlays      base64-encoded PNGs of per-image diagnostic
                      visualizations (typically a few MB).
      - diagnostics   per-image grid/orientation breakdown that the
                      cube-snap Fixer doesn't consume.
      - imageA/imageB summaries — kept lightweight by upstream code
                      already (just sticker/grid counts) so retained.

    Leaves status / state / confidence / reason / recognitionCategory /
    recognitionCategoryReason / failedChecks / candidates / runId / runUrl /
    artifacts / evaluation intact — everything callers actually need to
    drive a UI flow.
    """
    for key in ("overlays", "diagnostics"):
        payload.pop(key, None)


def run_batch(
    recognizer: WhiteUpRecognizer,
    pairs: Iterable[ImagePair],
    ground_truth: Optional[Dict[str, str]] = None,
    unpaired: Optional[List[str]] = None,
) -> Dict:
    batch_id = _run_id("batch")
    batch_dir = RUNS / "batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    results = []
    truth = ground_truth or {}
    for pair in pairs:
        expected = _expected_for_pair(truth, pair.set_id)
        payload = recognize_and_persist(recognizer, pair, expected)
        results.append(_batch_item(pair, payload))

    summary = _batch_summary(batch_id, results, unpaired or [])
    summary["batchUrl"] = f"/runs/batches/{batch_id}/batch_report.html"
    (batch_dir / "batch_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (batch_dir / "batch_report.html").write_text(_batch_html(summary), encoding="utf-8")
    return summary


def save_run(pair: ImagePair, payload: Dict, result, expected_state: Optional[str]) -> Dict:
    run_id = _run_id(pair.set_id)
    run_dir = RUNS / "pairs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    image_a_name = _safe_filename(pair.image_a.name or "imageA.jpg")
    image_b_name = _safe_filename(pair.image_b.name or "imageB.jpg")
    (run_dir / image_a_name).write_bytes(pair.image_a.data)
    (run_dir / image_b_name).write_bytes(pair.image_b.data)

    payload_no_overlays = json.loads(json.dumps({key: value for key, value in payload.items() if key != "overlays"}))
    debug = _debug_payload(result, payload_no_overlays)
    if expected_state:
        debug["expectedState"] = expected_state

    overlay_paths = {}
    for label, data_url in (payload.get("overlays") or {}).items():
        if data_url:
            filename = f"{label}_overlay.png"
            _write_data_url(run_dir / filename, data_url)
            overlay_paths[label] = f"/runs/pairs/{run_id}/{filename}"

    (run_dir / "result.json").write_text(json.dumps(payload_no_overlays, indent=2), encoding="utf-8")
    (run_dir / "debug.json").write_text(json.dumps(debug, indent=2), encoding="utf-8")
    _write_samples_csv(run_dir / "samples.csv", result)

    summary = {
        "runId": run_id,
        "setId": pair.set_id,
        "createdAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": payload.get("status"),
        "state": payload.get("state"),
        "confidence": payload.get("confidence"),
        "recognitionCategory": payload.get("recognitionCategory"),
        "recognitionCategoryReason": payload.get("recognitionCategoryReason"),
        "reason": payload.get("reason"),
        "failedChecks": payload.get("failedChecks", []),
        "evaluation": payload.get("evaluation"),
        "artifacts": {
            "run": f"/runs/pairs/{run_id}/",
            "imageA": f"/runs/pairs/{run_id}/{image_a_name}",
            "imageB": f"/runs/pairs/{run_id}/{image_b_name}",
            "result": f"/runs/pairs/{run_id}/result.json",
            "debug": f"/runs/pairs/{run_id}/debug.json",
            "samples": f"/runs/pairs/{run_id}/samples.csv",
            "overlays": overlay_paths,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "runId": run_id,
        "runUrl": f"/runs/pairs/{run_id}/summary.json",
        "artifacts": summary["artifacts"],
    }


def list_saved_runs(limit: int = 80) -> List[Dict]:
    pair_root = RUNS / "pairs"
    if not pair_root.exists():
        return []
    summaries = []
    for path in pair_root.glob("*/summary.json"):
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    summaries.sort(key=lambda item: item.get("createdAt", ""), reverse=True)
    return summaries[:limit]


def _expected_for_pair(ground_truth: Dict[str, str], set_id: str) -> Optional[str]:
    key = normalize_set_id(set_id)
    if key in ground_truth:
        return ground_truth[key]
    matches = [state for truth_key, state in ground_truth.items() if key.startswith(truth_key) or truth_key.startswith(key)]
    return matches[0] if len(matches) == 1 else None


def _debug_payload(result, payload: Dict) -> Dict:
    return {
        "result": payload,
        "imageA": _analysis_debug(result.image_a) if result.image_a else None,
        "imageB": _analysis_debug(result.image_b) if result.image_b else None,
    }


def _analysis_debug(analysis) -> Dict:
    return {
        "summary": analysis.summary(),
        "stickers": [_sticker_debug(sticker) for sticker in analysis.stickers],
        "grids": [
            {
                "id": grid.id,
                "centerFace": grid.center_face,
                "matchedCount": grid.matched_count,
                "fitError": grid.fit_error,
                "points": grid.points,
                "cells": [[_sticker_debug(sticker) for sticker in row] for row in grid.stickers],
            }
            for grid in analysis.grids
        ],
    }


def _sticker_debug(sticker) -> Dict:
    return {
        "id": sticker.id,
        "source": sticker.source,
        "center": sticker.center,
        "bbox": sticker.bbox,
        "rgb": sticker.rgb,
        "color": sticker.match.color,
        "face": sticker.match.face,
        "distance": sticker.match.distance,
        "confidence": sticker.match.confidence,
        "alternatives": sticker.match.alternatives[:6],
        "shapeAngle": sticker.shape_angle,
    }


def _write_samples_csv(path: Path, result) -> None:
    rows = []
    for label, analysis in (("imageA", result.image_a), ("imageB", result.image_b)):
        if not analysis:
            continue
        seen = set()
        for sticker in analysis.stickers:
            rows.append(_sample_row(label, "component", sticker))
            seen.add(id(sticker))
        for grid in analysis.grids:
            for r, row in enumerate(grid.stickers):
                for c, sticker in enumerate(row):
                    if id(sticker) in seen:
                        continue
                    rows.append(_sample_row(label, f"grid_{grid.id}_{r}{c}", sticker))
                    seen.add(id(sticker))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image", "source", "sticker_id", "x", "y", "r", "g", "b", "color", "face", "confidence", "alternatives"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _sample_row(label: str, source: str, sticker) -> Dict[str, object]:
    return {
        "image": label,
        "source": source,
        "sticker_id": sticker.id,
        "x": round(sticker.center[0], 2),
        "y": round(sticker.center[1], 2),
        "r": sticker.rgb[0],
        "g": sticker.rgb[1],
        "b": sticker.rgb[2],
        "color": sticker.match.color,
        "face": sticker.match.face,
        "confidence": round(sticker.match.confidence, 4),
        "alternatives": " ".join(f"{color}:{distance:.2f}" for color, distance in sticker.match.alternatives[:6]),
    }


def _batch_item(pair: ImagePair, payload: Dict) -> Dict:
    return {
        "setId": pair.set_id,
        "status": payload.get("status"),
        "state": payload.get("state"),
        "confidence": payload.get("confidence"),
        "reason": payload.get("reason"),
        "failedChecks": payload.get("failedChecks", []),
        "runId": payload.get("runId"),
        "runUrl": payload.get("runUrl"),
        "artifacts": payload.get("artifacts"),
        "evaluation": payload.get("evaluation", {"available": False}),
    }


def _batch_summary(batch_id: str, results: List[Dict], unpaired: List[str]) -> Dict:
    with_truth = [item for item in results if item.get("evaluation", {}).get("available")]
    return {
        "batchId": batch_id,
        "createdAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "totalPairs": len(results),
        "successes": sum(1 for item in results if item["status"] == "success"),
        "rejections": sum(1 for item in results if item["status"] != "success"),
        "truthCount": len(with_truth),
        "exactMatches": sum(1 for item in with_truth if item["evaluation"].get("exact")),
        "unpaired": unpaired,
        "results": results,
    }


def _batch_html(summary: Dict) -> str:
    rows = []
    for item in summary["results"]:
        evaluation = item.get("evaluation", {})
        exact = "" if not evaluation.get("available") else ("yes" if evaluation.get("exact") else f"no ({evaluation.get('hamming')})")
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['setId'])}</td>"
            f"<td>{html.escape(item['status'])}</td>"
            f"<td>{html.escape(item.get('recognitionCategory') or '')}</td>"
            f"<td>{html.escape(str(item.get('confidence') or ''))}</td>"
            f"<td>{html.escape(exact)}</td>"
            f"<td><code>{html.escape(item.get('state') or '')}</code></td>"
            f"<td>{html.escape(item.get('reason') or '')}</td>"
            f"<td><a href=\"{html.escape(item.get('runUrl') or '#')}\">run</a></td>"
            "</tr>"
        )
    return (
        "<!doctype html><meta charset=\"utf-8\"><title>Rubik Batch Report</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px}table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ccd5dc;padding:6px;vertical-align:top}code{overflow-wrap:anywhere}</style>"
        f"<h1>{html.escape(summary['batchId'])}</h1>"
        f"<p>{summary['successes']} success, {summary['rejections']} rejected, {summary['exactMatches']} exact matches.</p>"
        "<table><thead><tr><th>Set</th><th>Status</th><th>Category</th><th>Confidence</th><th>Exact</th><th>State</th><th>Reason</th><th>Run</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _write_data_url(path: Path, data_url: str) -> None:
    _, _, encoded = data_url.partition(",")
    path.write_bytes(base64.b64decode(encoded))


def _run_id(set_id: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", set_id).strip("-").lower()[:64] or "run"
    return f"{stamp}-{slug}"


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^a-zA-Z0-9._ -]+", "_", name).strip() or "upload"
    return name[:160]


if __name__ == "__main__":
    main()
