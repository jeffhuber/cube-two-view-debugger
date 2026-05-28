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
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
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
from rubik_recognizer.recognizer import RecognitionResult, WhiteUpRecognizer, recognition_diagnostics
from tools.constrained_inference_gate import evaluate_runtime_payload_gate
from tools.hull_label_pair_selector import choose_guarded_pair, repair_rank, repair_valid


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
RUNS = ROOT / "runs"
LABELS = RUNS / "labels"
CONSTRAINED_SHADOW_LOG_ENV = "CUBE_CONSTRAINED_SHADOW_LOG"
CONSTRAINED_INFERENCE_MODE_ENV = "CUBE_CONSTRAINED_INFERENCE_MODE"
_CONSTRAINED_SHADOW_LOG_LOCK = threading.Lock()


# Origins permitted to call the recognizer cross-origin. The frontend
# integration we care about is cube-snap (jeffhuber.github.io and the
# Vite dev server). 127.0.0.1 is included for parity with localhost
# since browsers treat them as different origins.
_ALLOWED_ORIGIN_PATTERNS = (
    re.compile(r"^https://jeffhuber\.github\.io$"),
    re.compile(r"^https://cubesnap\.app$"),
    re.compile(r"^https://www\.cubesnap\.app$"),
    re.compile(r"^http://localhost(?::\d+)?$"),
    re.compile(r"^http://127\.0\.0\.1(?::\d+)?$"),
)


class LlmRectifiedYawInferenceError(RuntimeError):
    def __init__(self, message: str, yaw_inference: Dict[str, Any]):
        super().__init__(message)
        self.yaw_inference = yaw_inference


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
        if path == "/api/labels":
            self._send_json({"labels": list_saved_labels()})
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
        if path == "/api/llm-rectified-input":
            self._handle_llm_rectified_input()
            return
        if path == "/api/recognize-batch":
            self._handle_batch()
            return
        if path == "/api/labels":
            self._handle_label_save()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _handle_recognize(self) -> None:
        try:
            fields = self._read_multipart()
            image_a = _first_field(fields, "imageA")
            image_b = _first_field(fields, "imageB")
            expected = _text_field(fields, "expectedState")
            set_id = _text_field(fields, "setId")
            hull_label_tier1_mode = (
                self._query_param("hullLabelTier1")
                or self._query_param("hull_label_tier1")
            )
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
                hull_label_tier1_mode=hull_label_tier1_mode,
            )
            # `payload["runtime"]` is now set inside `recognize_and_persist`
            # before `save_run` writes the on-disk `result.json`, so both
            # the HTTP response and the saved file carry the same block
            # (input-image SHA256 + byte size + decoded dimensions, plus
            # the env info `/api/diag` returns including freshness flags).
            # Codex review on PR #70 caught the previous bug where this
            # was attached post-save and the on-disk audit trail missed
            # the runtime data entirely.
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

    def _handle_llm_rectified_input(self) -> None:
        try:
            fields = self._read_multipart()
            image_a = _first_field(fields, "imageA")
            image_b = _first_field(fields, "imageB")
            yaw_quarter_turns = _parse_llm_rectified_yaw(
                self._query_param("yawQuarterTurns")
                or _text_field(fields, "yawQuarterTurns")
            )
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

            payload = prepare_llm_rectified_input(
                image_a[1],
                image_b[1],
                yaw_quarter_turns=yaw_quarter_turns,
            )
            self._send_json(payload)
        except LlmRectifiedYawInferenceError as exc:
            self._send_json(
                {
                    "status": "rejected",
                    "reason": str(exc),
                    "failedChecks": ["yaw_inference_ambiguous"],
                    "yawInference": exc.yaw_inference,
                },
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        except ValueError as exc:
            self._send_json(
                {
                    "status": "rejected",
                    "reason": str(exc),
                    "failedChecks": ["invalid_rectified_input_request"],
                },
                HTTPStatus.BAD_REQUEST,
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            self._send_json(
                {
                    "status": "rejected",
                    "reason": "Rectified LLM input preparation failed.",
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

    def _handle_label_save(self) -> None:
        try:
            payload = self._read_json_body(max_bytes=2_000_000)
            saved = save_label_document(payload)
            self._send_json(saved)
        except ValueError as exc:
            self._send_json(
                {
                    "status": "rejected",
                    "reason": str(exc),
                    "failedChecks": ["invalid_label_payload"],
                },
                HTTPStatus.BAD_REQUEST,
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            self._send_json(
                {
                    "status": "rejected",
                    "reason": "Label save failed.",
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

    def _query_param(self, name: str) -> Optional[str]:
        if "?" not in self.path:
            return None
        query = self.path.split("?", 1)[1]
        for pair in query.split("&"):
            if not pair:
                continue
            key, _, value = pair.partition("=")
            if unquote(key) == name:
                return unquote(value) if value else ""
        return None

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

    def _read_json_body(self, *, max_bytes: int) -> Dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ValueError("Expected application/json.")
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Expected a JSON request body.")
        if length > max_bytes:
            raise ValueError("Label payload is too large.")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Could not parse label JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Label payload must be a JSON object.")
        return payload

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


def _git_branch() -> Optional[str]:
    """Current branch name, or None for detached HEAD or any git failure.
    Surfaced in /api/diag and the startup banner so operators can tell
    at a glance whether the running server is on main, on a WIP branch,
    or detached. See the 'Cv-local server identity' section in CLAUDE.md.
    """
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _git_dirty() -> Optional[bool]:
    """Whether tracked files had uncommitted changes at server start.

    Deliberately ignores untracked files: local agent worktrees and
    downloaded diagnostics can be useful but should not make the UI
    permanently say "-dirty". A tracked-file diff is the signal that
    the loaded server may not correspond exactly to a committed SHA.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


# --- Server-staleness machinery (Codex review on PR #70 expanded this) -----
#
# Three module-level caches, all populated in `main()` at server start:
#
# - `_IDENTITY_AT_START`: the sha + branch frozen at the moment the
#   Python process loaded the recognizer code. This is the
#   *authoritative* identity for "which code is running" because Python
#   imports happen once at process boot; `git pull` afterwards updates
#   the working tree but not the running code. `/api/diag` reports this
#   as `git.sha` / `git.branch`.
#
# - `_GIT_FRESHNESS_AT_START`: the {commitsBehind, checkedAt, fetched}
#   snapshot from server start. Frozen for the lifetime of the process.
#   Exposed as `git.commitsBehindAtStart` for historical audit.
#
# - `_GIT_FRESHNESS_CACHE` (+ `_GIT_FRESHNESS_CACHE_LOCK`): a
#   lazily-refreshed view of the same freshness data. `/api/diag` and
#   the per-request `_per_request_runtime()` read this cache through
#   `_git_freshness_current()`, which re-runs `git fetch` + rev-list if
#   the cached value is older than `_GIT_FRESHNESS_REFRESH_TTL_SECONDS`.
#   This is what catches the May-12 case: a server started fresh that
#   accumulates staleness over hours/days while never being restarted.
#
# All three default to `None` for test-import scenarios (where `main()`
# never runs). `_runtime_diag()` returns sensible defaults in that case.
_IDENTITY_AT_START: Optional[Dict[str, Any]] = None
_GIT_FRESHNESS_AT_START: Optional[Dict[str, Any]] = None
_GIT_FRESHNESS_CACHE: Optional[Dict[str, Any]] = None
_GIT_FRESHNESS_CACHE_LOCK = threading.Lock()
_GIT_FRESHNESS_REFRESH_TTL_SECONDS = 600  # 10 min; balances "fresh enough for staleness audit" vs network cost

# Canonical server log path (default port). The boot banner is appended here
# (in addition to stderr) so `grep '[rubik-app].*identity:'
# /tmp/cv-local-server.log | tail -1` answers "which code is currently
# running?" regardless of how the server was started — including stderr
# redirects that vary across agents/checkouts (the original cause of the
# May 14 stale-log incident). Override with the CV_LOCAL_SERVER_LOG env var.
#
# The convention assumes one server per host on port 8080 (see CLAUDE.md
# "Cv-local server identity"). Servers started on alternate ports get a
# port-suffixed path via `_default_log_path_for_port()` so they don't
# pollute the canonical file — flagged by Codex on PR #75 review:
# without per-port isolation, an `app.py --port 8085` boot would land
# in /tmp/cv-local-server.log and make `tail -1` misreport "which code
# is :8080 serving?".
_DEFAULT_SERVER_LOG_PATH = Path("/tmp/cv-local-server.log")
_DEFAULT_SERVER_PORT = 8080


def _default_log_path_for_port(port: int) -> Path:
    """Canonical log path for a given port.

    Port 8080 (the convention default) uses the bare canonical path so
    the documented `grep | tail -1` lookup keeps working. Any other port
    gets `/tmp/cv-local-server-<port>.log` so alternate-port servers
    cannot pollute the canonical file with their identity.
    """
    if port == _DEFAULT_SERVER_PORT:
        return _DEFAULT_SERVER_LOG_PATH
    return Path(f"/tmp/cv-local-server-{port}.log")


def _git_freshness(*, fetch: bool = False) -> Dict[str, Any]:
    """Check whether the working tree is behind its upstream branch.

    When `fetch=True`, runs `git fetch --quiet origin` first (10s
    timeout for the network call) to update the local view of origin.
    Without that, the comparison is only as fresh as the user's last
    manual fetch/pull.

    Returns a dict with three keys:

    - `commitsBehind`: int count, or `None` when the check couldn't
      run (no git, no upstream, detached HEAD, network failure with
      no cached origin, etc.). `0` means up-to-date.
    - `checkedAt`: ISO-8601 UTC timestamp of the *attempt*. Always
      present (string), regardless of whether the count came back
      `None`. Codex review on PR #70 v2 caught that a `None`-count
      cache with `checkedAt=None` would be considered "stale" by
      `_git_freshness_current()`'s TTL gate forever, causing
      every request to retry `git fetch`. Recording the attempt
      timestamp here is what fixes that — the TTL works the same way
      for known and unknown count states.
    - `fetched`: whether `git fetch` ran successfully during this
      call. Even when False, `commitsBehind` may still be useful (it
      reflects whatever the local view of origin already had).

    Compares against `@{upstream}` rather than hardcoded `origin/main`
    so a server running on a feature branch reports whether *that
    branch* has moved on origin, not the (irrelevant) distance from
    main. Practical for staleness detection; doesn't try to tell you
    "main has moved" on a feature-branch deployment.

    Never raises — every git error degrades to `commitsBehind=None`.
    See cube-two-view-debugger CLAUDE.md "Cv-local server identity"
    for the convention this supports.
    """
    result: Dict[str, Any] = {
        "commitsBehind": None,
        # Stamped here, BEFORE any subprocess call, so the timestamp
        # reflects the attempt regardless of whether git is even
        # available. Codex PR #70 v2 review.
        "checkedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "fetched": False,
    }
    if fetch:
        try:
            r = subprocess.run(
                ["git", "fetch", "--quiet", "origin"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=10,  # network op; capped so a slow link doesn't hang server start
            )
            result["fetched"] = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        r = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..@{upstream}"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=2,
        )
        if r.returncode == 0:
            result["commitsBehind"] = int(r.stdout.strip() or "0")
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return result


def _git_freshness_current() -> Dict[str, Any]:
    """Return the lazily-refreshed freshness, refreshing on TTL miss.

    Reads `_GIT_FRESHNESS_CACHE` under `_GIT_FRESHNESS_CACHE_LOCK`. If
    the cached value is older than `_GIT_FRESHNESS_REFRESH_TTL_SECONDS`
    (or has never been populated by `main()`), runs
    `_git_freshness(fetch=True)` to refresh from origin.

    Codex review on PR #70 flagged that startup-only freshness misses
    the May-12 case: server starts at commit A (0 behind), main moves
    25 commits, server keeps running with no signal. This helper is
    what closes that gap — every `/api/diag` hit and every saved
    `result.json` reflects the cached-at-request-time freshness, not
    just the boot-time snapshot.

    Thread-safe. Returns a default-shaped dict when the cache was
    never populated (test-import case where `main()` didn't run).
    """
    global _GIT_FRESHNESS_CACHE
    default = {"commitsBehind": None, "checkedAt": None, "fetched": False}
    with _GIT_FRESHNESS_CACHE_LOCK:
        cache = _GIT_FRESHNESS_CACHE
        # Test-import path: cache never populated. Don't trigger a
        # network call from a test that didn't ask for one.
        if cache is None:
            return default
        stale = True
        checked_at = cache.get("checkedAt")
        if isinstance(checked_at, str):
            try:
                checked = dt.datetime.fromisoformat(checked_at)
                now = dt.datetime.now(dt.timezone.utc)
                age_seconds = (now - checked).total_seconds()
                if age_seconds < _GIT_FRESHNESS_REFRESH_TTL_SECONDS:
                    stale = False
            except (ValueError, TypeError):
                stale = True
        if stale:
            _GIT_FRESHNESS_CACHE = _git_freshness(fetch=True)
        return _GIT_FRESHNESS_CACHE or default


def _runtime_diag() -> Dict[str, Any]:
    """Stable JSON-serialisable description of the recognizer's runtime.

    Identity (`git.sha`, `git.branch`) reflects the **loaded server
    code**, frozen at process start. Re-pulling the working tree
    without restarting does NOT change these — the loaded code is
    what actually serves requests. See `_IDENTITY_AT_START`.

    Freshness comes in two flavours:
      - `commitsBehindAtStart` / `commitsBehindCheckedAtStart`:
        snapshot from server start; useful for audit ("how stale was
        the server when this run was persisted?").
      - `commitsBehind` / `commitsBehindCheckedAt`: lazily refreshed
        on every diag call (TTL-gated); reflects the current view of
        origin.

    Warnings are driven by the CURRENT freshness, so a server that
    started clean and accumulated staleness over hours/days surfaces
    the staleness as soon as someone hits `/api/diag` or a recognition
    runs (because both call `_per_request_runtime` which goes through
    this function).
    """
    at_start = _GIT_FRESHNESS_AT_START or {
        "commitsBehind": None,
        "checkedAt": None,
        "fetched": False,
    }
    current = _git_freshness_current()
    identity = _IDENTITY_AT_START or {
        # Test-import fallback: when main() never ran, fall through
        # to current values so the field is still populated. The
        # banner-emission path doesn't use this fallback.
        "sha": _git_sha(),
        "branch": _git_branch(),
        "dirty": _git_dirty(),
    }
    warnings: List[str] = []
    current_behind = current.get("commitsBehind")
    if isinstance(current_behind, int) and current_behind > 0:
        # No "_at_start" suffix anymore: the warning reflects current
        # staleness, not just at-boot staleness. A server that started
        # fresh and drifted gets this warning as soon as a request
        # triggers `_git_freshness_current()` past the TTL.
        warnings.append(f"server_stale_by_{current_behind}_commits")
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
            # Authoritative identity = frozen at server start. The
            # banner uses these values too, so log-grep and /api/diag
            # always agree.
            "sha": identity.get("sha"),
            "branch": identity.get("branch"),
            "dirty": identity.get("dirty"),
            "dirtyScope": "tracked",
            "cwd": str(ROOT),
            # Lazy-refreshed: changes over the server's lifetime.
            "commitsBehind": current.get("commitsBehind"),
            "commitsBehindCheckedAt": current.get("checkedAt"),
            # Frozen at start: never updates. Useful for audit when
            # comparing a `result.json` from boot-time vs a later run.
            "commitsBehindAtStart": at_start.get("commitsBehind"),
            "commitsBehindCheckedAtStart": at_start.get("checkedAt"),
        },
        "warnings": warnings,
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
        {"method": "GET",  "path": "/api/labels",          "brief": "List of recently saved cube-geometry label JSON documents."},
        {"method": "GET",  "path": "/runs/pairs/<id>/...", "brief": "Static access to a saved run's files (result.json, debug.json, overlays, samples.csv, original photos)."},
        {"method": "GET",  "path": "/runs/labels/<id>.json", "brief": "Static access to saved cube-geometry label JSON."},
        {"method": "POST", "path": "/api/recognize",       "brief": "Recognize one pair. Multipart fields: imageA, imageB; optional setId, expectedState. Query: ?slim=1 to omit overlays/diagnostics; ?hullLabelTier1=shadow|prefer for the hidden hull-label Tier 1 candidate path, or constrained-shadow|constrained for the hidden constrained-inference gate path. Persists a run under /runs/pairs/<id>/."},
        {"method": "POST", "path": "/api/llm-rectified-input", "brief": "Prepare two Claude/GPT-ready rectified WCA contact-sheet JPEGs from imageA/imageB. Multipart fields: imageA, imageB; optional yawQuarterTurns=0..3 or auto. Does not call an LLM or persist a run."},
        {"method": "POST", "path": "/api/recognize-batch", "brief": "Recognize multiple pairs in one call. Multipart field: images (multi-file); optional groundTruth (.csv/.tsv/.json). Pairs files by filename A/B markers or by drop order. Persists a batch under /runs/batches/<id>/."},
        {"method": "POST", "path": "/api/labels",          "brief": "Persist one cube-geometry label JSON document under /runs/labels/."},
    ]


_LLM_RECTIFIED_WCA_GROUPS = {
    "imageA": ("U", "F", "R"),
    "imageB": ("D", "L", "B"),
}
_LLM_RECTIFIED_SESSION: Any = None
_LLM_RECTIFIED_SESSION_LOCK = threading.Lock()


def _parse_llm_rectified_yaw(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in ("", "auto", "infer", "center", "center-inference"):
        return None
    try:
        return int(value) % 4
    except ValueError:
        raise ValueError("yawQuarterTurns must be an integer 0..3 or auto")


def _llm_rectified_session() -> Any:
    global _LLM_RECTIFIED_SESSION
    with _LLM_RECTIFIED_SESSION_LOCK:
        if _LLM_RECTIFIED_SESSION is None:
            from rembg import new_session

            _LLM_RECTIFIED_SESSION = new_session("u2net")
        return _LLM_RECTIFIED_SESSION


def prepare_llm_rectified_input(
    image_a_bytes: bytes,
    image_b_bytes: bytes,
    *,
    yaw_quarter_turns: Optional[int] = None,
    max_side: int = 1600,
    panel_size: int = 300,
) -> Dict[str, Any]:
    """Create labeled WCA face contact sheets for LLM color reading.

    This is intentionally preparation-only: it does not call any LLM and it
    does not persist a recognizer run. CubeSnap's Fixer uses it as a local
    geometry-cleanup step before sending the resulting JPEGs to its normal
    cloud LLM endpoint with the rectified-face prompt. When
    ``yaw_quarter_turns`` is omitted, infer capture yaw from the six
    rectified slot-center colors before assigning panels to WCA faces.
    """

    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from rembg import remove

    from tools.corner_conventions import wca_face_by_slot
    from tools.global_cube_model import _slot_center_faces_from_rectified
    from tools.hull_label_color_repair import repair_from_hull_label_fits
    from tools.hull_label_assembly import convention_orientation_for_slot
    from tools.rectify_via_hull_labels import DEFAULT_MASK_THRESHOLDS, select_hull_label_threshold_fit

    def load_image(payload: bytes) -> Image.Image:
        with Image.open(io.BytesIO(payload)) as img:
            rgb = ImageOps.exif_transpose(img).convert("RGB")
        side = max(rgb.size)
        if side > max_side:
            scale = max_side / side
            rgb = rgb.resize(
                (round(rgb.width * scale), round(rgb.height * scale)),
                Image.Resampling.LANCZOS,
            )
        return rgb

    def apply_orientation(face: Image.Image, mirror: bool, rot_quarter: int) -> Image.Image:
        out = face
        if mirror:
            out = ImageOps.mirror(out)
        if rot_quarter % 4:
            out = out.rotate(90 * (rot_quarter % 4), expand=True)
        if out.size[0] != out.size[1]:
            side = min(out.size)
            out = ImageOps.fit(out, (side, side), method=Image.Resampling.BICUBIC)
        return out

    def label_panel(face: str, img: Image.Image) -> Image.Image:
        panel = Image.new("RGB", (panel_size, panel_size + 34), "white")
        draw = ImageDraw.Draw(panel)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
        draw.text((6, 4), face, fill=(30, 30, 30), font=font)
        resized = img.resize((panel_size, panel_size), Image.Resampling.BICUBIC)
        panel.paste(resized, (0, 34))
        try:
            number_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)
        except Exception:
            number_font = ImageFont.load_default()
        for i in (1, 2):
            x = round(i * panel_size / 3)
            y = 34 + round(i * panel_size / 3)
            draw.line([(x, 34), (x, 34 + panel_size)], fill=(255, 255, 255), width=2)
            draw.line([(0, y), (panel_size, y)], fill=(255, 255, 255), width=2)
        cell = panel_size / 3.0
        for index in range(9):
            row, col = divmod(index, 3)
            draw.text(
                (round(col * cell + 8), round(34 + row * cell + 6)),
                f"{face}{index + 1}",
                fill=(20, 20, 20),
                font=number_font,
                stroke_width=2,
                stroke_fill=(255, 255, 255),
            )
        draw.rectangle((0, 34, panel_size - 1, 34 + panel_size - 1), outline=(40, 40, 40), width=2)
        return panel

    def make_face_sheet(group_name: str, faces: Iterable[str]) -> Image.Image:
        panels = [label_panel(face, panels_by_face[face]) for face in faces]
        margin = 14
        title_h = 36
        width = len(panels) * panel_size + (len(panels) - 1) * margin
        sheet = Image.new("RGB", (width, title_h + panel_size + 34), "white")
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
        draw.text((0, 4), f"{group_name}: read each WCA facelet label exactly", fill=(30, 30, 30), font=font)
        x = 0
        for panel in panels:
            sheet.paste(panel, (x, title_h))
            x += panel_size + margin
        return sheet

    def jpeg_base64(image: Image.Image) -> Tuple[str, int]:
        out = io.BytesIO()
        image.convert("RGB").save(out, "JPEG", quality=90, optimize=True)
        data = out.getvalue()
        return base64.b64encode(data).decode("ascii"), len(data)

    def fit_threshold_candidates(
        side: str,
        image: Image.Image,
        alpha: _np.ndarray,
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
        accepted: Dict[int, Dict[str, Any]] = {}
        candidates: List[Dict[str, Any]] = []
        for threshold in DEFAULT_MASK_THRESHOLDS:
            selection = select_hull_label_threshold_fit(
                image,
                alpha,
                side,
                thresholds=[int(threshold)],
                face_size_px=panel_size,
            )
            candidate_rows = selection.trace.get("threshold_candidates") or []
            candidate = dict(candidate_rows[0] if candidate_rows else selection.trace)
            candidates.append(candidate)
            if selection.fit is None:
                continue
            trace = dict(selection.trace)
            trace["slot_center_faces"] = _slot_center_faces_from_rectified(selection.fit.rectified_faces)
            accepted[int(threshold)] = {
                "fit": selection.fit,
                "image": image,
                "trace": trace,
            }
        aggregate_trace = {
            "thresholds": [int(value) for value in DEFAULT_MASK_THRESHOLDS],
            "threshold_candidates": candidates,
            "accepted_thresholds": sorted(accepted),
        }
        return accepted, aggregate_trace

    def current_entry(entries: Mapping[int, Mapping[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        threshold, entry = min(
            entries.items(),
            key=lambda item: (
                float((item[1].get("trace") or {}).get("sticker_score_total") or 999999.0),
                int(item[0]),
            ),
        )
        return int(threshold), dict(entry)

    def evaluate_pair(side_entries: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
        fits_for_yaw = {side: side_entries[side]["fit"] for side in ("A", "B")}
        yaw_inference = _infer_yaw_from_rectified_fits(fits_for_yaw)
        if yaw_quarter_turns is None:
            if not yaw_inference.get("accepted"):
                return {
                    "status": "yaw_unavailable",
                    "yawInference": yaw_inference,
                    "yawQuarterTurns": None,
                    "yawSource": "center-inference",
                }
            selected = int(yaw_inference["yawQuarterTurns"])
            source = "center-inference"
        else:
            selected = yaw_quarter_turns % 4
            source = "explicit"
        try:
            repair = repair_from_hull_label_fits(
                side_fits=side_entries,
                yaw_quarter_turns=selected,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "repair_error",
                "yawInference": yaw_inference,
                "yawQuarterTurns": selected,
                "yawSource": source,
                "repair": {
                    "schema": "hull_label_color_repair_v1",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            }
        return {
            "status": "assembled",
            "yawInference": yaw_inference,
            "yawQuarterTurns": selected,
            "yawSource": source,
            "repair": repair,
        }

    def compact_combo(
        thresholds: Mapping[str, int],
        side_entries: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        evaluation = evaluate_pair(side_entries)
        score_total = sum(
            float((side_entries[side].get("trace") or {}).get("sticker_score_total") or 0.0)
            for side in ("A", "B")
        )
        return {
            "thresholds": {"A": int(thresholds["A"]), "B": int(thresholds["B"])},
            "sideFits": {side: dict(side_entries[side]) for side in ("A", "B")},
            "evaluation": evaluation,
            "productionRank": list(repair_rank(evaluation)),
            "stickerScoreTotal": round(score_total, 2),
        }

    image_a = load_image(image_a_bytes)
    image_b = load_image(image_b_bytes)
    session = _llm_rectified_session()
    threshold_entries_by_side: Dict[str, Dict[int, Dict[str, Any]]] = {}
    threshold_diagnostics_by_side: Dict[str, Any] = {}
    for side, image in (("A", image_a), ("B", image_b)):
        rgba = remove(image, session=session).convert("RGBA")
        alpha = _np.asarray(rgba.split()[-1], dtype=_np.uint8)
        entries, threshold_diagnostics = fit_threshold_candidates(side, image, alpha)
        if not entries:
            raise RuntimeError(
                f"hull-label rectification failed for side {side}: "
                "no alpha threshold candidate accepted"
            )
        threshold_entries_by_side[side] = entries
        threshold_diagnostics_by_side[side] = threshold_diagnostics

    current_thresholds: Dict[str, int] = {}
    current_side_entries: Dict[str, Dict[str, Any]] = {}
    for side in ("A", "B"):
        threshold, entry = current_entry(threshold_entries_by_side[side])
        current_thresholds[side] = threshold
        current_side_entries[side] = entry

    current_combo = compact_combo(current_thresholds, current_side_entries)
    pair_candidates: List[Dict[str, Any]] = [current_combo]
    if not repair_valid(current_combo["evaluation"]):
        pair_candidates = []
        for threshold_a, entry_a in sorted(threshold_entries_by_side["A"].items()):
            for threshold_b, entry_b in sorted(threshold_entries_by_side["B"].items()):
                pair_candidates.append(compact_combo(
                    {"A": threshold_a, "B": threshold_b},
                    {"A": entry_a, "B": entry_b},
                ))
    selected_combo = choose_guarded_pair(
        current_combo=current_combo,
        candidates=pair_candidates,
        fallback_to_current_without_alternative=True,
    )
    selected_eval = selected_combo["evaluation"]
    yaw_inference = selected_eval.get("yawInference") or {}
    if selected_eval.get("status") == "yaw_unavailable" and yaw_quarter_turns is None:
        raise LlmRectifiedYawInferenceError(
            "could not infer capture yaw from rectified center colors "
            f"(best={yaw_inference.get('bestYawQuarterTurns')}, "
            f"score={yaw_inference.get('bestScore')}, "
            f"second={yaw_inference.get('secondScore')}, "
            f"margin={yaw_inference.get('margin')})",
            dict(yaw_inference),
        )
    selected_yaw = int(selected_eval.get("yawQuarterTurns") or 0)
    yaw_source = str(selected_eval.get("yawSource") or "center-inference")
    selected_side_entries = selected_combo["sideFits"]
    fits_by_side = {side: selected_side_entries[side]["fit"] for side in ("A", "B")}
    threshold_traces_by_side = {
        side: dict(selected_side_entries[side].get("trace") or {})
        for side in ("A", "B")
    }
    for side in ("A", "B"):
        threshold_traces_by_side[side]["threshold_candidates"] = threshold_diagnostics_by_side[side]["threshold_candidates"]
        threshold_traces_by_side[side]["accepted_thresholds"] = threshold_diagnostics_by_side[side]["accepted_thresholds"]
        threshold_traces_by_side[side]["pair_selected_mask_threshold"] = selected_combo["thresholds"][side]
    pair_threshold_selection = {
        "strategy": "guarded_pair_when_current_invalid",
        "selectionReason": selected_combo.get("selectionReason"),
        "currentThresholds": current_thresholds,
        "selectedThresholds": selected_combo.get("thresholds"),
        "currentRepairValid": repair_valid(current_combo["evaluation"]),
        "selectedRepairValid": repair_valid(selected_eval),
        "currentProductionRank": current_combo.get("productionRank"),
        "selectedProductionRank": selected_combo.get("productionRank"),
        "evaluatedPairCount": len(pair_candidates),
        "possiblePairCount": (
            len(threshold_entries_by_side["A"]) * len(threshold_entries_by_side["B"])
        ),
    }

    panels_by_face: Dict[str, Image.Image] = {}
    panel_metadata: List[Dict[str, Any]] = []
    for side, fit in (("A", fits_by_side["A"]), ("B", fits_by_side["B"])):
        assignments = wca_face_by_slot(side, selected_yaw)
        for slot, wca_face in assignments.items():
            orientation = convention_orientation_for_slot(
                side=side,
                slot=slot,
                yaw_quarter_turns=selected_yaw,
                wca_face=wca_face,
                quad=fit.face_quads[slot],
            )
            if orientation is None:
                raise RuntimeError(f"could not orient side {side} slot {slot} as WCA face {wca_face}")
            panels_by_face[wca_face] = apply_orientation(fit.rectified_faces[slot], *orientation)
            panel_metadata.append({
                "side": side,
                "image": "imageA" if side == "A" else "imageB",
                "slot": slot,
                "wcaFace": wca_face,
                "yawQuarterTurns": selected_yaw,
                "mirror": orientation[0],
                "rotQuarter": orientation[1],
            })

    missing = sorted(set("URFDLB") - set(panels_by_face))
    if missing:
        raise RuntimeError(f"rectified panels missing WCA faces: {', '.join(missing)}")
    image_a_sheet = make_face_sheet("Image 1", _LLM_RECTIFIED_WCA_GROUPS["imageA"])
    image_b_sheet = make_face_sheet("Image 2", _LLM_RECTIFIED_WCA_GROUPS["imageB"])
    image_a_b64, image_a_size = jpeg_base64(image_a_sheet)
    image_b_b64, image_b_size = jpeg_base64(image_b_sheet)
    panel_metadata.sort(key=lambda item: "URFDLB".index(str(item["wcaFace"])))
    deterministic_repair = selected_eval.get("repair") or {
        "schema": "hull_label_color_repair_v1",
        "status": selected_eval.get("status", "unavailable"),
    }
    promotion_gate = evaluate_runtime_payload_gate(
        repair=deterministic_repair,
        pair_threshold_selection=pair_threshold_selection,
        side_traces=threshold_traces_by_side,
        yaw_inference=yaw_inference,
    )
    return {
        "status": "success",
        "prompt": "rectified",
        "imageA": image_a_b64,
        "imageB": image_b_b64,
        "imageABytes": image_a_size,
        "imageBBytes": image_b_size,
        "yawQuarterTurns": selected_yaw,
        "yawSource": yaw_source,
        "yawInference": yaw_inference,
        "panels": panel_metadata,
        "hullLabelMaskThresholds": threshold_traces_by_side,
        "hullLabelPairThresholdSelection": pair_threshold_selection,
        "deterministicColorRepair": deterministic_repair,
        "constrainedInferencePromotionGate": promotion_gate,
    }


def _infer_yaw_from_rectified_fits(fits_by_side: Dict[str, Any]) -> Dict[str, Any]:
    from tools.hull_label_yaw import infer_yaw_from_rectified_fits

    return infer_yaw_from_rectified_fits(fits_by_side)


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


def _write_boot_record(host: str, port: int, diag: Dict[str, Any]) -> None:
    """Emit the startup identity banner to stderr AND a canonical log file.

    The stderr path matches historical behavior (nohup-redirected logs get
    the banner immediately because stderr is unbuffered). The file path is
    new: writing to a fixed location means `grep '[rubik-app].*identity:'
    /tmp/cv-local-server.log | tail -1` reliably reports the running
    server's identity even when different invocations use different stderr
    redirects. Append mode (not truncate) preserves any stderr-redirected
    content that may already be in the file.

    Path resolution:
      1. `CV_LOCAL_SERVER_LOG` env var (test harnesses, custom setups)
      2. `/tmp/cv-local-server.log` when port == 8080 (canonical)
      3. `/tmp/cv-local-server-<port>.log` for any other port (per-port
         isolation so an `app.py --port 8085` boot can't pollute the
         canonical file's `tail -1` identity)

    Configurable via CV_LOCAL_SERVER_LOG; failures to write the file are
    swallowed so a read-only /tmp or missing parent dir can't crash the
    server.

    Note on stream collision: if the operator redirects stderr to the
    same path that this function writes (e.g.
    `python app.py > /tmp/cv-local-server.log 2>&1` with the canonical
    path), two file descriptors point at the same file — one with
    `O_APPEND` (this function's `open(...,'a')`), one without (the
    shell-inherited stderr fd, opened by `>`). Concurrent writes can
    overlap and partially overwrite the boot record. The CLAUDE.md
    convention separates stderr from the canonical log to avoid this;
    if you must share the path, use `>>` so the shell-inherited fd
    also has `O_APPEND`. See Devin's review on PR #75.
    """
    sha = diag["git"]["sha"] or "unknown"
    if diag["git"].get("dirty") is True:
        sha = f"{sha}-dirty"
    branch = diag["git"]["branch"] or "detached"
    lines = [
        f"[rubik-app] Serving http://{host}:{port}/",
        f"[rubik-app]   identity: {diag['git']['cwd']} @ {sha} ({branch})",
        (
            f"[rubik-app]   env:      "
            f"python {diag['python']['version']}, "
            f"pillow {diag['libraries']['pillow']}, "
            f"numpy {diag['libraries']['numpy']}"
        ),
    ]
    # Staleness warning: if the working tree was behind its upstream
    # branch at startup, surface a loud line in the banner. Caught the
    # 32-hour-stale May 12 server postmortem; the goal is "anyone
    # reading the server log notices immediately instead of finding
    # out via a misclassified recognition hours later."
    behind = diag["git"].get("commitsBehindAtStart")
    if isinstance(behind, int) and behind > 0:
        upstream = branch if branch != "detached" else "upstream"
        lines.append(
            f"[rubik-app]   WARNING:  {behind} commit(s) behind origin/{upstream}. "
            f"Pull + restart to refresh."
        )

    for line in lines:
        print(line, file=sys.stderr)

    log_path_str = os.environ.get(
        "CV_LOCAL_SERVER_LOG", str(_default_log_path_for_port(port))
    )
    try:
        with open(log_path_str, "a") as f:
            # Separator + timestamp delimit each boot record so callers
            # can `tail -1` reliably even if the file accumulates many
            # boots, and so unrelated stderr-redirected content stays
            # visually separated.
            f.write("\n" + "=" * 40 + "\n")
            f.write(f"[rubik-app]   booted:   {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
            for line in lines:
                f.write(line + "\n")
    except OSError:
        # Don't crash the server because we couldn't write the audit
        # file. The stderr banner still went out.
        pass


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

    # Freeze identity (sha, branch) at server start. This is the
    # *authoritative* "which code is running" — Python imports load
    # the recognizer module once at process boot, so any subsequent
    # `git pull` updates the working tree but not the loaded code.
    # `/api/diag` reports these as `git.sha` / `git.branch`. The same
    # values feed the startup banner so log-grep and HTTP API agree.
    global _IDENTITY_AT_START
    _IDENTITY_AT_START = {
        "sha": _git_sha(),
        "branch": _git_branch(),
        "dirty": _git_dirty(),
    }

    # Populate the freshness cache before the banner so (1) the banner
    # can surface a "behind origin" warning at startup, and (2) the
    # lazy-refresh path has an initial cached value to age from. Runs
    # `git fetch` once (10s cap); silent on network failure.
    #
    # Two separate module-level variables intentionally:
    # - `_GIT_FRESHNESS_AT_START` is frozen for the lifetime of the
    #   process (exposed as `git.commitsBehindAtStart` for audit).
    # - `_GIT_FRESHNESS_CACHE` is lazily refreshed by
    #   `_git_freshness_current()` on every diag/recognition call
    #   when older than `_GIT_FRESHNESS_REFRESH_TTL_SECONDS`.
    global _GIT_FRESHNESS_AT_START, _GIT_FRESHNESS_CACHE
    _GIT_FRESHNESS_AT_START = _git_freshness(fetch=True)
    _GIT_FRESHNESS_CACHE = dict(_GIT_FRESHNESS_AT_START)  # seed; lazily updated thereafter

    server = ThreadingHTTPServer((args.host, args.port), RubikHandler)
    diag = _runtime_diag()
    # Identity banner. Surfaces which repo/branch/SHA is serving so a
    # grep on the server log answers "which code is running?" without
    # hitting /api/diag. Critical when multiple repo clones on the
    # same host compete for port 8080 — see CLAUDE.md "Cv-local server
    # identity" for the full convention. Writes to stderr AND a
    # canonical log file so the grep works regardless of how each
    # agent/checkout redirects stderr.
    _write_boot_record(args.host, args.port, diag)
    server.serve_forever()


def _first_field(fields: Dict[str, List[Tuple[str, bytes]]], name: str) -> Optional[Tuple[str, bytes]]:
    values = fields.get(name) or []
    return values[0] if values else None


def _text_field(fields: Dict[str, List[Tuple[str, bytes]]], name: str) -> Optional[str]:
    value = _first_field(fields, name)
    if not value:
        return None
    return value[1].decode("utf-8", errors="replace").strip() or None


def recognize_and_persist(
    recognizer: WhiteUpRecognizer,
    pair: ImagePair,
    expected_state: Optional[str] = None,
    hull_label_tier1_mode: Optional[str] = None,
) -> Dict:
    effective_mode = _effective_hull_label_tier1_mode(hull_label_tier1_mode)
    constrained_mode = _normalize_constrained_inference_mode(effective_mode)
    if constrained_mode:
        result = _recognize_with_constrained_inference_mode(
            recognizer,
            pair.image_a.data,
            pair.image_b.data,
            constrained_mode,
            expected_state=expected_state,
        )
    else:
        result = recognizer.recognize(
            pair.image_a.data,
            pair.image_b.data,
            hull_label_tier1_mode=effective_mode,
        )
    payload = result.to_api_dict()
    if result.image_a and result.image_b:
        payload["diagnostics"] = recognition_diagnostics(result.image_a, result.image_b)
    evaluation = evaluate_state(result.state, expected_state)
    if evaluation.get("available"):
        payload["evaluation"] = evaluation
    # Attach the runtime block BEFORE save_run so the persisted
    # `result.json` carries the same identity + freshness data as the
    # HTTP response. Codex review on PR #70 caught that the previous
    # location (after recognize_and_persist returned) wrote to disk
    # *before* runtime was set, so the on-disk audit trail was missing
    # the staleness signal. Per-request runtime includes input image
    # fingerprints (SHA256, byte size, decoded dimensions), so a saved
    # run self-documents both what input bytes produced it AND what
    # env / freshness the server was in.
    payload["runtime"] = _per_request_runtime(pair.image_a.data, pair.image_b.data)
    run_info = save_run(pair, payload, result, expected_state)
    payload.update(run_info)
    if constrained_mode:
        _append_constrained_shadow_event(pair, payload, constrained_mode)
    return payload


def _effective_hull_label_tier1_mode(explicit_mode: Optional[str]) -> Optional[str]:
    if explicit_mode is not None and str(explicit_mode).strip():
        return explicit_mode
    raw = os.environ.get(CONSTRAINED_INFERENCE_MODE_ENV)
    value = str(raw or "").strip().lower().replace("_", "-")
    if value in {"", "0", "false", "off", "none"}:
        return explicit_mode
    if value in {"shadow", "trace", "diagnostic"}:
        return "constrained-shadow"
    if value in {"prefer", "candidate"}:
        return "constrained"
    return raw


def _normalize_constrained_inference_mode(raw: Optional[str]) -> Optional[str]:
    value = str(raw or "").strip().lower().replace("_", "-")
    if value in {"constrained-shadow", "constrained-trace", "constrained-diagnostic"}:
        return "shadow"
    if value in {"constrained", "constrained-prefer", "constrained-candidate"}:
        return "prefer"
    return None


def _candidate_evaluation_from_payload(
    payload: Mapping[str, Any],
    expected_state: Optional[str],
) -> Dict[str, Any]:
    if not expected_state:
        return {"available": False}
    repair = payload.get("deterministicColorRepair")
    repair_payload = repair if isinstance(repair, Mapping) else {}
    recommended = repair_payload.get("recommended")
    recommended_payload = recommended if isinstance(recommended, Mapping) else {}
    state = recommended_payload.get("state")
    evaluation = evaluate_state(state if isinstance(state, str) else None, expected_state)
    return {
        key: evaluation.get(key)
        for key in ("available", "exact", "hamming", "expectedValid", "expectedErrors")
        if key in evaluation
    }


def _compact_cubie_consistency_signal(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return None
    return {
        key: raw.get(key)
        for key in (
            "totalCubies",
            "consistentCount",
            "inconsistentCount",
            "inconsistentSplitCount",
            "inconsistentInImageCount",
            "inconsistentNames",
        )
        if key in raw
    }


def _two_view_consistency_signal_from_repair(repair: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    methods = repair.get("methods")
    if not isinstance(methods, Mapping):
        return None
    two_view = methods.get("two_view_consistency_repaired")
    if not isinstance(two_view, Mapping):
        return None
    gate = two_view.get("gate")
    gate_payload = gate if isinstance(gate, Mapping) else {}
    return {
        "status": two_view.get("status"),
        "validState": two_view.get("validState"),
        "countBalanced": two_view.get("countBalanced"),
        "repairCost": two_view.get("repairCost"),
        "repairChanges": two_view.get("repairChanges"),
        "gate": {
            "accepted": gate_payload.get("accepted"),
            "reasons": gate_payload.get("reasons"),
            "stateDeltaFromCanonical": gate_payload.get("stateDeltaFromCanonical"),
            "baselineCubieConsistency": _compact_cubie_consistency_signal(
                gate_payload.get("baselineCubieConsistency")
            ),
            "candidateCubieConsistency": _compact_cubie_consistency_signal(
                gate_payload.get("candidateCubieConsistency")
            ),
        },
    }


def _constrained_signal_from_payload(
    payload: Mapping[str, Any],
    *,
    selected: bool,
    expected_state: Optional[str] = None,
) -> Dict[str, Any]:
    repair = payload.get("deterministicColorRepair")
    repair_payload = repair if isinstance(repair, Mapping) else {}
    recommended = repair_payload.get("recommended")
    recommended_payload = recommended if isinstance(recommended, Mapping) else {}
    return {
        "schema": "constrained_inference_recognize_signal_v1",
        "selected": selected,
        "fallbackToLegacy": not selected,
        "status": payload.get("status"),
        "yawQuarterTurns": payload.get("yawQuarterTurns"),
        "yawSource": payload.get("yawSource"),
        "yawInference": payload.get("yawInference"),
        "pairThresholdSelection": payload.get("hullLabelPairThresholdSelection"),
        "promotionGate": payload.get("constrainedInferencePromotionGate"),
        "recommendedMethod": repair_payload.get("recommendedMethod"),
        "recommended": {
            "validState": recommended_payload.get("validState"),
            "countBalanced": recommended_payload.get("countBalanced"),
            "confidence": recommended_payload.get("confidence"),
            "repairMoveCount": recommended_payload.get("repairMoveCount"),
            "repairCost": recommended_payload.get("repairCost"),
            "repairChanges": recommended_payload.get("repairChanges"),
            "stateDeltaFromCanonical": recommended_payload.get("stateDeltaFromCanonical"),
        },
        "twoViewConsistencyRepair": _two_view_consistency_signal_from_repair(repair_payload),
        "candidateEvaluation": _candidate_evaluation_from_payload(payload, expected_state),
    }


def _constrained_candidate_result(
    payload: Mapping[str, Any],
    *,
    expected_state: Optional[str] = None,
) -> Optional[RecognitionResult]:
    gate = payload.get("constrainedInferencePromotionGate")
    if not isinstance(gate, Mapping) or gate.get("accepted") is not True:
        return None
    repair = payload.get("deterministicColorRepair")
    if not isinstance(repair, Mapping):
        return None
    recommended = repair.get("recommended")
    if not isinstance(recommended, Mapping):
        return None
    state = recommended.get("state")
    if not isinstance(state, str) or len(state) != 54:
        return None
    confidence_label = str(recommended.get("confidence") or "")
    confidence = 0.90 if confidence_label == "high" else 0.80
    return RecognitionResult(
        status="success",
        state=state,
        confidence=confidence,
        reason="Recognized a unique legal white-up cube state via constrained hull-label inference.",
        failed_checks=[],
        candidates=1,
        recognition_signals={
            "constrainedInference": _constrained_signal_from_payload(
                payload,
                selected=True,
                expected_state=expected_state,
            ),
        },
    )


def _attach_constrained_shadow_signal(
    result: RecognitionResult,
    payload: Mapping[str, Any],
    *,
    selected: bool,
    expected_state: Optional[str] = None,
) -> None:
    signals = dict(result.recognition_signals or {})
    signals["constrainedInference"] = _constrained_signal_from_payload(
        payload,
        selected=selected,
        expected_state=expected_state,
    )
    result.recognition_signals = signals


def _attach_constrained_error_signal(result: RecognitionResult, exc: Exception, *, mode: str) -> None:
    signals = dict(result.recognition_signals or {})
    signals["constrainedInference"] = {
        "schema": "constrained_inference_recognize_signal_v1",
        "selected": False,
        "fallbackToLegacy": True,
        "mode": mode,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
    }
    result.recognition_signals = signals


def _constrained_shadow_log_path() -> Optional[Path]:
    raw = os.environ.get(CONSTRAINED_SHADOW_LOG_ENV)
    if raw is not None and raw.strip().lower() in {"", "0", "false", "off", "none"}:
        return None
    if raw:
        return Path(raw).expanduser()
    return RUNS / "constrained_inference_shadow.jsonl"


def _compact_constrained_shadow_event(
    pair: ImagePair,
    payload: Mapping[str, Any],
    mode: str,
) -> Optional[Dict[str, Any]]:
    signals = payload.get("recognitionSignals")
    if not isinstance(signals, Mapping):
        return None
    signal = signals.get("constrainedInference")
    if not isinstance(signal, Mapping):
        return None

    gate = signal.get("promotionGate") if isinstance(signal.get("promotionGate"), Mapping) else {}
    recommended = signal.get("recommended") if isinstance(signal.get("recommended"), Mapping) else {}
    candidate_evaluation = (
        signal.get("candidateEvaluation")
        if isinstance(signal.get("candidateEvaluation"), Mapping)
        else {}
    )
    pair_selection = (
        signal.get("pairThresholdSelection")
        if isinstance(signal.get("pairThresholdSelection"), Mapping)
        else {}
    )
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), Mapping) else {}
    inputs = runtime.get("inputs") if isinstance(runtime.get("inputs"), Mapping) else {}
    evaluation = payload.get("evaluation") if isinstance(payload.get("evaluation"), Mapping) else None
    return {
        "schema": "constrained_inference_shadow_event_v1",
        "createdAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": mode,
        "runId": payload.get("runId"),
        "runUrl": payload.get("runUrl"),
        "setId": pair.set_id,
        "result": {
            "status": payload.get("status"),
            "recognitionCategory": payload.get("recognitionCategory"),
            "recognitionCategoryReason": payload.get("recognitionCategoryReason"),
            "failedChecks": payload.get("failedChecks", []),
            "confidence": payload.get("confidence"),
        },
        "constrainedInference": {
            "selected": signal.get("selected"),
            "fallbackToLegacy": signal.get("fallbackToLegacy"),
            "status": signal.get("status"),
            "yawQuarterTurns": signal.get("yawQuarterTurns"),
            "yawSource": signal.get("yawSource"),
            "recommendedMethod": signal.get("recommendedMethod"),
            "recommended": {
                "validState": recommended.get("validState"),
                "countBalanced": recommended.get("countBalanced"),
                "confidence": recommended.get("confidence"),
                "repairMoveCount": recommended.get("repairMoveCount"),
                "repairCost": recommended.get("repairCost"),
                "repairChanges": recommended.get("repairChanges"),
                "stateDeltaFromCanonical": recommended.get("stateDeltaFromCanonical"),
            },
            "candidateEvaluation": {
                "available": candidate_evaluation.get("available"),
                "exact": candidate_evaluation.get("exact"),
                "hamming": candidate_evaluation.get("hamming"),
                "expectedValid": candidate_evaluation.get("expectedValid"),
                "expectedErrors": candidate_evaluation.get("expectedErrors"),
            },
            "promotionGate": {
                "accepted": gate.get("accepted"),
                "decision": gate.get("decision"),
                "rejectReasons": list(gate.get("rejectReasons") or []),
                "productionRank": gate.get("productionRank"),
            },
            "pairThresholdSelection": {
                "selectionReason": pair_selection.get("selectionReason"),
                "currentThresholds": pair_selection.get("currentThresholds"),
                "selectedThresholds": pair_selection.get("selectedThresholds"),
            },
        },
        "inputs": {
            "imageA": inputs.get("imageA"),
            "imageB": inputs.get("imageB"),
        },
        **({"evaluation": evaluation} if evaluation else {}),
    }


def _append_constrained_shadow_event(
    pair: ImagePair,
    payload: Mapping[str, Any],
    mode: str,
) -> None:
    event = _compact_constrained_shadow_event(pair, payload, mode)
    if event is None:
        return
    path = _constrained_shadow_log_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, sort_keys=True)
        with _CONSTRAINED_SHADOW_LOG_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:  # noqa: BLE001
        print(
            f"[rubik-app] warning: failed to append constrained shadow event to {path}: {exc}",
            file=sys.stderr,
        )


def _recognize_with_constrained_inference_mode(
    recognizer: WhiteUpRecognizer,
    image_a: bytes,
    image_b: bytes,
    mode: str,
    *,
    expected_state: Optional[str] = None,
) -> RecognitionResult:
    legacy = recognizer.recognize(image_a, image_b, hull_label_tier1_mode="off")
    try:
        payload = prepare_llm_rectified_input(image_a, image_b)
    except Exception as exc:  # noqa: BLE001
        _attach_constrained_error_signal(legacy, exc, mode=mode)
        return legacy

    candidate = _constrained_candidate_result(payload, expected_state=expected_state)
    if mode == "prefer" and candidate is not None:
        return candidate

    _attach_constrained_shadow_signal(legacy, payload, selected=False, expected_state=expected_state)
    return legacy


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


def save_label_document(payload: Dict[str, Any]) -> Dict[str, Any]:
    _validate_label_document(payload)
    set_id = str(payload.get("setId") or "unlabelled").strip() or "unlabelled"
    image = payload.get("image") if isinstance(payload.get("image"), dict) else {}
    image_side = str(payload.get("imageSide") or image.get("side") or "image").strip() or "image"
    label_id = _run_id(f"{set_id}-{image_side}-geometry-label")
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    document = json.loads(json.dumps(payload))
    document.update(
        {
            "schemaVersion": int(document.get("schemaVersion") or 1),
            "labelId": label_id,
            "savedAt": created_at,
            "labelUrl": f"/runs/labels/{label_id}.json",
        }
    )
    LABELS.mkdir(parents=True, exist_ok=True)
    path = LABELS / f"{label_id}.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return _label_summary(document)


def list_saved_labels(limit: int = 80) -> List[Dict[str, Any]]:
    if not LABELS.exists():
        return []
    summaries: List[Dict[str, Any]] = []
    for path in LABELS.glob("*.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        summaries.append(_label_summary(document))
    summaries.sort(key=lambda item: item.get("savedAt", ""), reverse=True)
    return summaries[:limit]


def _validate_label_document(payload: Dict[str, Any]) -> None:
    labels = payload.get("labels")
    image = payload.get("image")
    if not isinstance(image, dict):
        raise ValueError("Label payload must include an image object.")
    if not isinstance(labels, dict):
        raise ValueError("Label payload must include a labels object.")
    face_quads = labels.get("faceQuads") or {}
    cube_hull = labels.get("cubeHull") or []
    if not isinstance(face_quads, dict):
        raise ValueError("labels.faceQuads must be an object.")
    for face, points in face_quads.items():
        if face not in {"U", "R", "F", "D", "L", "B"}:
            raise ValueError(f"Unknown face label: {face}.")
        if not _is_point_list(points, expected_len=4):
            raise ValueError(f"Face {face} must contain exactly four points.")
    if cube_hull and not _is_point_list(cube_hull, min_len=3):
        raise ValueError("labels.cubeHull must contain at least three points.")
    if not face_quads and not cube_hull:
        raise ValueError("Add at least one face quad or cube hull before saving.")


def _is_point_list(value: Any, *, expected_len: Optional[int] = None, min_len: Optional[int] = None) -> bool:
    if not isinstance(value, list):
        return False
    if expected_len is not None and len(value) != expected_len:
        return False
    if min_len is not None and len(value) < min_len:
        return False
    for point in value:
        if not isinstance(point, dict):
            return False
        if not isinstance(point.get("x"), (int, float)) or not isinstance(point.get("y"), (int, float)):
            return False
    return True


def _label_summary(document: Dict[str, Any]) -> Dict[str, Any]:
    image = document.get("image") if isinstance(document.get("image"), dict) else {}
    labels = document.get("labels") if isinstance(document.get("labels"), dict) else {}
    face_quads = labels.get("faceQuads") if isinstance(labels.get("faceQuads"), dict) else {}
    cube_hull = labels.get("cubeHull") if isinstance(labels.get("cubeHull"), list) else []
    label_id = document.get("labelId") or Path(str(document.get("labelUrl") or "")).stem
    return {
        "labelId": label_id,
        "labelUrl": document.get("labelUrl") or (f"/runs/labels/{label_id}.json" if label_id else None),
        "savedAt": document.get("savedAt"),
        "setId": document.get("setId"),
        "imageSide": document.get("imageSide") or image.get("side"),
        "imageName": image.get("name"),
        "imageSha256": image.get("sha256"),
        "imageWidth": image.get("width"),
        "imageHeight": image.get("height"),
        "faceLabels": sorted(face_quads),
        "faceQuadCount": len(face_quads),
        "cubeHullPointCount": len(cube_hull),
    }


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
