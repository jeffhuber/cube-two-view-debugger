"""
Tests for the HTTP API surface in app.py.

These are intentionally focused on the route shape + the most
load-bearing behaviors (setId pass-through, route self-description,
saved-run listing), not on the recognizer itself which has its own
suite. They use the recognizer behind the scenes so they pay full
recognition cost — keep the fixtures small and synthetic.
"""
from __future__ import annotations

import datetime as dt
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


def test_runtime_diag_includes_both_at_start_and_current_freshness_fields():
    """Diag carries TWO freshness snapshots (Codex review on PR #70
    expanded this from one): `commitsBehindAtStart` is frozen for the
    process lifetime; `commitsBehind` is lazily refreshed on a TTL.
    Both must be present (each may be None when not populated)."""
    from app import _runtime_diag

    diag = _runtime_diag()
    git = diag.get("git") or {}
    for key in (
        "commitsBehind",
        "commitsBehindCheckedAt",
        "commitsBehindAtStart",
        "commitsBehindCheckedAtStart",
    ):
        assert key in git, (
            f"expected git.{key} in diag, got keys: {sorted(git.keys())}"
        )
    for value in (git["commitsBehind"], git["commitsBehindAtStart"]):
        assert value is None or (isinstance(value, int) and value >= 0), (
            f"behind count must be None or non-negative int, got {value!r}"
        )


def test_runtime_diag_warnings_field_present_and_empty_by_default():
    """`warnings` is the top-level audit-trail field. Empty list means
    'nothing flagged'; non-empty means downstream consumers should
    surface them. Default state in unit-test imports is no cache, so
    the list must be empty."""
    from app import _runtime_diag

    diag = _runtime_diag()
    assert "warnings" in diag, f"expected top-level 'warnings' field, got: {sorted(diag.keys())}"
    assert isinstance(diag["warnings"], list)
    assert diag["warnings"] == []


def test_runtime_diag_warnings_driven_by_current_not_at_start(monkeypatch):
    """**Codex review on PR #70, finding #1**: warnings must reflect
    CURRENT staleness (so a server that started fresh and accumulated
    25 commits of staleness while running gets flagged), not the
    frozen at-start value (which would stay 0 forever on a
    started-fresh server).
    """
    import app as app_module

    now_iso = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    # At-start: server booted fresh, 0 behind. Frozen.
    monkeypatch.setattr(
        app_module,
        "_GIT_FRESHNESS_AT_START",
        {"commitsBehind": 0, "checkedAt": now_iso, "fetched": True},
        raising=False,
    )
    # Current: 25 commits accumulated since boot. The lazy-refresh
    # cache reflects this; warnings + audit trail must follow.
    monkeypatch.setattr(
        app_module,
        "_GIT_FRESHNESS_CACHE",
        {"commitsBehind": 25, "checkedAt": now_iso, "fetched": True},
        raising=False,
    )
    diag = app_module._runtime_diag()
    assert diag["git"]["commitsBehindAtStart"] == 0
    assert diag["git"]["commitsBehind"] == 25
    # Warning string format changed (no "_at_start" suffix) — the
    # warning is about *current* staleness, not just at-boot.
    assert "server_stale_by_25_commits" in diag["warnings"], (
        f"expected current-staleness warning, got: {diag['warnings']}"
    )


def test_runtime_diag_no_warning_when_current_fresh(monkeypatch):
    """commitsBehind=0 in the current cache → no warning, regardless
    of at-start."""
    import app as app_module

    now_iso = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    monkeypatch.setattr(
        app_module,
        "_GIT_FRESHNESS_CACHE",
        {"commitsBehind": 0, "checkedAt": now_iso, "fetched": True},
        raising=False,
    )
    diag = app_module._runtime_diag()
    assert diag["git"]["commitsBehind"] == 0
    assert diag["warnings"] == []


def test_runtime_diag_no_warning_when_current_unknown(monkeypatch):
    """commitsBehind=None means the check couldn't run (no upstream,
    no network, git not installed). No warning — we can't tell.
    Distinct from commitsBehind=0 (verified up-to-date).

    Note: the cache must carry a fresh `checkedAt` so the TTL helper
    returns it verbatim rather than triggering a real refresh. In
    practice this scenario is "we tried recently, got None back."""
    import app as app_module

    now_iso = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    monkeypatch.setattr(
        app_module,
        "_GIT_FRESHNESS_CACHE",
        {"commitsBehind": None, "checkedAt": now_iso, "fetched": False},
        raising=False,
    )
    diag = app_module._runtime_diag()
    assert diag["git"]["commitsBehind"] is None
    assert diag["warnings"] == []


def test_runtime_diag_identity_uses_at_start_when_populated(monkeypatch):
    """**Codex review on PR #70, finding #3**: `git.sha` and
    `git.branch` reflect the **loaded server code** (frozen at server
    start), not the live working tree. Pulling the repo while the
    process keeps running updates HEAD but NOT the loaded code; the
    API must report the loaded code's identity.
    """
    import app as app_module

    monkeypatch.setattr(
        app_module,
        "_IDENTITY_AT_START",
        {"sha": "abc1234", "branch": "main"},
        raising=False,
    )
    diag = app_module._runtime_diag()
    assert diag["git"]["sha"] == "abc1234", (
        f"expected loaded-code sha, got: {diag['git']['sha']}"
    )
    assert diag["git"]["branch"] == "main"


def test_runtime_diag_identity_falls_back_when_not_started_through_main():
    """When `_IDENTITY_AT_START` is None (test-import scenario), the
    diag falls back to live `_git_sha()` / `_git_branch()`. Keeps
    unit tests working without requiring them to populate the cache."""
    from app import _runtime_diag

    diag = _runtime_diag()
    sha = diag["git"]["sha"]
    assert sha is None or (isinstance(sha, str) and sha), (
        f"sha must be None or non-empty str, got: {sha!r}"
    )


def test_git_freshness_returns_well_formed_shape():
    """`_git_freshness()` always returns the three documented keys.
    Without `fetch=True` this is just a `git rev-list` call against the
    local view of origin — fast, no network."""
    from app import _git_freshness

    result = _git_freshness(fetch=False)
    assert set(result.keys()) == {"commitsBehind", "checkedAt", "fetched"}
    assert result["fetched"] is False
    behind = result["commitsBehind"]
    assert behind is None or (isinstance(behind, int) and behind >= 0)
    checked_at = result["checkedAt"]
    assert checked_at is None or isinstance(checked_at, str)


def test_git_freshness_current_returns_cache_when_fresh(monkeypatch):
    """`_git_freshness_current()` returns the cached value verbatim
    when its `checkedAt` is within the TTL — no subprocess call.
    Validates the TTL-gate that keeps per-request latency low."""
    import app as app_module

    now_iso = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    cache = {"commitsBehind": 3, "checkedAt": now_iso, "fetched": True}
    monkeypatch.setattr(app_module, "_GIT_FRESHNESS_CACHE", cache, raising=False)

    # Spy: if `_git_freshness` is called the cache was deemed stale.
    called = {"count": 0}
    def spy(*args, **kwargs):
        called["count"] += 1
        return {"commitsBehind": 999, "checkedAt": now_iso, "fetched": True}
    monkeypatch.setattr(app_module, "_git_freshness", spy)

    result = app_module._git_freshness_current()
    assert result == cache
    assert called["count"] == 0, "should NOT refresh while cache is within TTL"


def test_git_freshness_current_refreshes_when_cache_is_stale(monkeypatch):
    """When `checkedAt` is older than the TTL, the helper re-runs
    `_git_freshness(fetch=True)`. This is the mechanism that catches
    the May-12 case: a server with a stale cache picks up freshness
    changes on the next request."""
    import app as app_module

    old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=app_module._GIT_FRESHNESS_REFRESH_TTL_SECONDS + 60)
    monkeypatch.setattr(
        app_module,
        "_GIT_FRESHNESS_CACHE",
        {"commitsBehind": 0, "checkedAt": old.isoformat(timespec="seconds"), "fetched": True},
        raising=False,
    )
    fresh_value = {
        "commitsBehind": 7,
        "checkedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "fetched": True,
    }
    monkeypatch.setattr(
        app_module,
        "_git_freshness",
        lambda *, fetch=False: fresh_value,
    )

    result = app_module._git_freshness_current()
    assert result == fresh_value, "stale cache should be replaced by refresh result"
    assert app_module._GIT_FRESHNESS_CACHE == fresh_value


def test_git_freshness_current_returns_default_when_cache_never_populated(monkeypatch):
    """The helper must NOT trigger a real network call when the cache
    was never populated (test-import scenario). Prevents tests from
    hanging on `git fetch`."""
    import app as app_module

    monkeypatch.setattr(app_module, "_GIT_FRESHNESS_CACHE", None, raising=False)
    called = {"count": 0}
    def fail_if_called(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("must not call _git_freshness when cache is None")
    monkeypatch.setattr(app_module, "_git_freshness", fail_if_called)

    result = app_module._git_freshness_current()
    assert result == {"commitsBehind": None, "checkedAt": None, "fetched": False}
    assert called["count"] == 0


def test_recognize_and_persist_writes_runtime_to_disk(tmp_path, monkeypatch):
    """**Codex review on PR #70, finding #2**: `payload["runtime"]`
    must be attached BEFORE `save_run` writes `result.json` so the
    on-disk audit trail carries the freshness/identity block. The
    previous PR location (after `recognize_and_persist` returned)
    persisted result.json BEFORE runtime was set, leaving the on-disk
    audit trail empty.
    """
    import app as app_module
    from rubik_recognizer.dataset import ImagePair, ImageUpload
    from PIL import Image

    monkeypatch.setattr(app_module, "RUNS", tmp_path / "runs", raising=False)

    # Tiny solid-color JPEGs. The recognizer will fast-reject (can't
    # form a legal cube from a solid color), but that's fine —
    # save_run still writes result.json on rejection paths.
    def jpeg(color):
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), color).save(buf, format="JPEG", quality=70)
        return buf.getvalue()

    pair = ImagePair(
        set_id="set-runtime-disk-test",
        image_a=ImageUpload("a.jpg", jpeg((255, 255, 255))),
        image_b=ImageUpload("b.jpg", jpeg((255, 255, 0))),
    )
    app_module.recognize_and_persist(app_module.WhiteUpRecognizer(), pair)

    saved_dirs = list((tmp_path / "runs" / "pairs").glob("*"))
    assert saved_dirs, f"expected a saved run under {tmp_path / 'runs' / 'pairs'}"
    saved_result = json.loads((saved_dirs[0] / "result.json").read_text())

    # The fix: on-disk file carries the runtime block.
    assert "runtime" in saved_result, (
        "Persisted result.json must include the runtime block (Codex review "
        "PR #70 finding #2). Got keys: " + ", ".join(sorted(saved_result.keys()))
    )
    rt = saved_result["runtime"]
    assert "git" in rt
    assert "sha" in rt["git"]
    assert "warnings" in rt
    # And imageA/imageB fingerprints (the original audit-trail purpose).
    assert "imageA" in rt and "imageB" in rt


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
