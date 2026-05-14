"""
Tests for the HTTP API surface in app.py.

These are intentionally focused on the route shape + the most
load-bearing behaviors (setId pass-through, route self-description,
saved-run listing), not on the recognizer itself which has its own
suite. They use the recognizer behind the scenes so they pay full
recognition cost — keep the fixtures small and synthetic.
"""
from __future__ import annotations

import io
import json
import sys
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import Tuple

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import RubikHandler, _api_routes  # noqa: E402


def test_api_routes_lists_known_endpoints():
    """The route table should cover every dispatch arm exposed in
    do_GET / do_POST. If a new route is added, update _api_routes()
    so cold-start agents can discover it."""
    routes = _api_routes()
    paths = {entry["path"] for entry in routes}

    # Spot-check the routes that exist in dispatch.
    assert "/api/routes" in paths
    assert "/api/diag" in paths
    assert "/api/runs" in paths
    assert "/api/recognize" in paths
    assert "/api/recognize-batch" in paths
    assert "/runs/pairs/<id>/..." in paths
    assert "/" in paths
    assert "/static/*" in paths

    # Every entry has the expected three keys.
    for entry in routes:
        assert set(entry.keys()) >= {"method", "path", "brief"}
        assert entry["method"] in {"GET", "POST"}
        assert isinstance(entry["brief"], str) and entry["brief"], "non-empty brief"


@pytest.fixture(scope="module")
def server() -> Tuple[str, int]:
    """Spin up RubikHandler on an ephemeral port for the integration
    tests that need real HTTP responses. Tests share one server (module
    scope) to avoid paying the per-test bind cost."""
    from http.server import ThreadingHTTPServer

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), RubikHandler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield host, port
    finally:
        httpd.shutdown()
        thread.join(timeout=2)


def _get_json(server, path):
    host, port = server
    conn = HTTPConnection(host, port, timeout=5)
    conn.request("GET", path)
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response.status, json.loads(body or b"{}")


def test_api_routes_http(server):
    """The /api/routes endpoint should return the same data as the helper."""
    status, payload = _get_json(server, "/api/routes")
    assert status == HTTPStatus.OK
    assert "routes" in payload
    assert payload["routes"] == _api_routes()


def test_api_runs_returns_list(server):
    """/api/runs should always return a {runs: [...]} object, even when
    the runs/ directory doesn't exist or is empty."""
    status, payload = _get_json(server, "/api/runs")
    assert status == HTTPStatus.OK
    assert "runs" in payload
    assert isinstance(payload["runs"], list)


def test_api_diag_smoke(server):
    """/api/diag should report python + library versions. Tests we can
    keep cheap because the schema rarely changes; this catches the
    obvious regression of the endpoint being broken."""
    status, payload = _get_json(server, "/api/diag")
    assert status == HTTPStatus.OK
    assert "python" in payload
    assert "libraries" in payload
    assert "numpy" in payload["libraries"]
    assert "pillow" in payload["libraries"]


def test_api_diag_exposes_git_identity(server):
    """The git identity block must include sha + branch + cwd so
    operators can answer "which code is :8080 serving?" without
    grep-ing the server log. See the 'Cv-local server identity'
    section in CLAUDE.md for the convention."""
    status, payload = _get_json(server, "/api/diag")
    assert status == HTTPStatus.OK
    git = payload.get("git") or {}
    assert "sha" in git, f"expected git.sha in /api/diag, got keys: {sorted(git.keys())}"
    assert "branch" in git, f"expected git.branch in /api/diag, got keys: {sorted(git.keys())}"
    assert "cwd" in git, f"expected git.cwd in /api/diag, got keys: {sorted(git.keys())}"
    # cwd should be the repo root (where app.py lives).
    assert git["cwd"] == str(ROOT), f"git.cwd should equal ROOT ({ROOT}), got {git['cwd']!r}"
    # branch may be None for detached HEAD; otherwise it's a non-empty string.
    if git["branch"] is not None:
        assert isinstance(git["branch"], str) and git["branch"], "branch should be non-empty when present"


def test_runtime_diag_includes_branch_field():
    """Direct unit test for _runtime_diag's branch field — covers the
    case where the test process happens to be on a detached HEAD or
    fresh clone with no checked-out branch. branch == None is allowed;
    branch == '' is NOT (would be a regression of _git_branch's
    empty-string-to-None conversion)."""
    from app import _runtime_diag

    diag = _runtime_diag()
    assert "git" in diag
    assert "branch" in diag["git"], "branch field must be present even when None"
    branch = diag["git"]["branch"]
    assert branch is None or (isinstance(branch, str) and branch), (
        f"branch must be None or non-empty str, got {branch!r}"
    )


def test_runtime_diag_includes_freshness_fields():
    """`commitsBehindAtStart` + `commitsBehindCheckedAt` carry the
    staleness signal that the startup banner emits to stderr, so any
    saved result.json or /api/diag hit can prove the server's age
    after-the-fact. Tested separately because not every call path
    populates the freshness cache (e.g. test imports), so the fields
    must still exist with None values then."""
    from app import _runtime_diag

    diag = _runtime_diag()
    git = diag.get("git") or {}
    assert "commitsBehindAtStart" in git, (
        f"expected git.commitsBehindAtStart in diag, got: {sorted(git.keys())}"
    )
    assert "commitsBehindCheckedAt" in git, (
        f"expected git.commitsBehindCheckedAt in diag, got: {sorted(git.keys())}"
    )
    # Values can be None (when freshness wasn't populated this run)
    # OR (int, ISO8601-string-or-None). Tests can't assume which.
    behind = git["commitsBehindAtStart"]
    assert behind is None or (isinstance(behind, int) and behind >= 0), (
        f"commitsBehindAtStart must be None or non-negative int, got {behind!r}"
    )


def test_runtime_diag_warnings_field_present_and_empty_by_default():
    """`warnings` is the top-level audit-trail field. Empty list means
    'nothing flagged at start'; non-empty means downstream consumers
    (saved result.json, the Fixer, the bench harness) should surface
    them. Default state in unit-test imports is no freshness cache, so
    the list must be empty."""
    from app import _runtime_diag

    diag = _runtime_diag()
    assert "warnings" in diag, f"expected top-level 'warnings' field, got: {sorted(diag.keys())}"
    assert isinstance(diag["warnings"], list)
    # No cache populated by tests → no behind-count → empty warnings.
    assert diag["warnings"] == []


def test_runtime_diag_warnings_populated_when_cache_says_behind(monkeypatch):
    """When the module-level freshness cache (normally populated by
    `main()` at server start) records a non-zero behind-count, the
    diag adds a `server_stale_by_N_commits_at_start` warning. This is
    the audit-trail mechanism for the postmortem case: a result.json
    persisted by a stale server should carry the staleness flag so a
    reviewer can detect the issue after the fact."""
    import app as app_module

    # Inject a synthetic cache. Mirrors what _git_freshness(fetch=True)
    # would return on a 7-commits-behind working tree.
    monkeypatch.setattr(
        app_module,
        "_GIT_FRESHNESS_AT_START",
        {
            "commitsBehind": 7,
            "checkedAt": "2026-05-14T04:30:00+00:00",
            "fetched": True,
        },
        raising=False,
    )
    diag = app_module._runtime_diag()
    assert diag["git"]["commitsBehindAtStart"] == 7
    assert diag["git"]["commitsBehindCheckedAt"] == "2026-05-14T04:30:00+00:00"
    assert "server_stale_by_7_commits_at_start" in diag["warnings"], (
        f"expected stale warning, got: {diag['warnings']}"
    )


def test_runtime_diag_no_warning_when_cache_says_fresh(monkeypatch):
    """commitsBehind=0 is the up-to-date case — no warning emitted.
    Distinguishes 'verified fresh' from 'couldn't verify' (None)."""
    import app as app_module

    monkeypatch.setattr(
        app_module,
        "_GIT_FRESHNESS_AT_START",
        {
            "commitsBehind": 0,
            "checkedAt": "2026-05-14T04:30:00+00:00",
            "fetched": True,
        },
        raising=False,
    )
    diag = app_module._runtime_diag()
    assert diag["git"]["commitsBehindAtStart"] == 0
    assert diag["warnings"] == []


def test_runtime_diag_no_warning_when_cache_says_unknown(monkeypatch):
    """commitsBehind=None means the check couldn't run — no upstream,
    no network, git not installed, etc. Don't emit a stale warning in
    that case (we can't tell). Distinguishing this from
    commitsBehind=0 is what prevents false alarms on offline servers."""
    import app as app_module

    monkeypatch.setattr(
        app_module,
        "_GIT_FRESHNESS_AT_START",
        {"commitsBehind": None, "checkedAt": None, "fetched": False},
        raising=False,
    )
    diag = app_module._runtime_diag()
    assert diag["git"]["commitsBehindAtStart"] is None
    assert diag["warnings"] == []


def test_git_freshness_returns_well_formed_shape():
    """`_git_freshness()` always returns the three documented keys.
    Without `fetch=True` this is just a `git rev-list` call against the
    local view of origin — fast, no network. Values may be None on
    detached HEAD or no-upstream branches, but the keys must exist."""
    from app import _git_freshness

    result = _git_freshness(fetch=False)
    assert set(result.keys()) == {"commitsBehind", "checkedAt", "fetched"}
    # fetched=False because we passed fetch=False
    assert result["fetched"] is False
    behind = result["commitsBehind"]
    assert behind is None or (isinstance(behind, int) and behind >= 0)
    checked_at = result["checkedAt"]
    assert checked_at is None or isinstance(checked_at, str)


def _multipart_body(fields, boundary="testboundary123"):
    """Build a minimal multipart/form-data body. `fields` is a list of
    (name, filename_or_None, value_bytes) tuples. Returns (content_type, body)."""
    chunks = []
    for name, filename, value in fields:
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        if filename is not None:
            chunks.append(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8")
            )
            chunks.append(b"Content-Type: application/octet-stream\r\n\r\n")
        else:
            chunks.append(
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
            )
        if isinstance(value, str):
            chunks.append(value.encode("utf-8"))
        else:
            chunks.append(value)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(chunks)
    return f"multipart/form-data; boundary={boundary}", body


def test_recognize_requires_both_images(server):
    """Missing imageA/imageB should fast-fail with a 400 + a descriptive
    failedChecks entry. Catches the regression of the form accepting
    half-submitted requests."""
    host, port = server
    content_type, body = _multipart_body(
        [("imageA", "a.jpg", b"not-actually-a-jpeg")],
        boundary="b1",
    )
    conn = HTTPConnection(host, port, timeout=5)
    conn.request(
        "POST",
        "/api/recognize",
        body=body,
        headers={"Content-Type": content_type, "Content-Length": str(len(body))},
    )
    response = conn.getresponse()
    payload = json.loads(response.read())
    conn.close()
    assert response.status == HTTPStatus.BAD_REQUEST
    assert payload.get("status") == "rejected"
    assert "missing_upload" in (payload.get("failedChecks") or [])


def _solid_jpeg(color=(255, 255, 255)) -> bytes:
    """Return a tiny solid-color JPEG. The recognizer can't produce a
    legal cube from this, so the test asserts on the FAILURE path —
    which is fine because we're testing setId pass-through, not
    recognition accuracy."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def test_recognize_persists_user_supplied_set_id(server, tmp_path):
    """When the form supplies a setId, the run id should include it
    (slugified). Catches the regression of the setId form field being
    ignored — which would silently break the UI's set-name input."""
    host, port = server
    content_type, body = _multipart_body(
        [
            ("imageA", "a.jpg", _solid_jpeg((255, 255, 255))),
            ("imageB", "b.jpg", _solid_jpeg((255, 255, 0))),
            ("setId", None, "Set 9999 (api test)"),
        ],
        boundary="b2",
    )
    conn = HTTPConnection(host, port, timeout=180)
    conn.request(
        "POST",
        "/api/recognize",
        body=body,
        headers={"Content-Type": content_type, "Content-Length": str(len(body))},
    )
    response = conn.getresponse()
    payload = json.loads(response.read())
    conn.close()

    assert response.status == HTTPStatus.OK
    # The run id should carry a slugified form of the supplied set id.
    run_id = payload.get("runId") or ""
    assert "set-9999" in run_id, f"runId did not include slugified setId: {run_id!r}"
    # The on-disk run dir should exist.
    run_dir = ROOT / "runs" / "pairs" / run_id
    assert run_dir.exists(), f"saved run dir missing: {run_dir}"
