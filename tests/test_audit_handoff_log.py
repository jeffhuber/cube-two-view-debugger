from __future__ import annotations

import json
import os
from pathlib import Path

from tools import audit_handoff_log as log


def _events(tmp_path: Path):
    path = tmp_path / log.EVENTS_FILE
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_start_refuses_duplicate_live_lock(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AUDIT_HANDOFF_LOG_DIR", str(tmp_path))
    argv = [
        "start",
        "--lane", "codex-audit",
        "--repo", "owner/repo",
        "--pr", "12",
        "--head", "abcdef1234567890",
        "--pid", str(os.getpid()),
        "--actor", "codex",
    ]

    assert log.main(argv) == 0
    duplicate_rc = log.main(argv)

    assert duplicate_rc == log.ACTIVE_EXIT_CODE
    captured = capsys.readouterr()
    assert "active audit already running" in captured.err
    events = _events(tmp_path)
    assert events[-1]["event"] == "duplicate_refused"


def test_finish_removes_lock_and_logs_exit_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AUDIT_HANDOFF_LOG_DIR", str(tmp_path))
    assert log.main([
        "start",
        "--lane", "codex-audit",
        "--repo", "owner/repo",
        "--pr", "13",
        "--head", "123456abcdef",
        "--pid", str(os.getpid()),
    ]) == 0
    lock_id = capsys.readouterr().out.strip()

    assert (tmp_path / log.LOCKS_DIR / f"{lock_id}.json").exists()
    assert log.main(["finish", "--lock-id", lock_id, "--status", "completed", "--exit-code", "0"]) == 0

    assert not (tmp_path / log.LOCKS_DIR / f"{lock_id}.json").exists()
    events = _events(tmp_path)
    assert events[-1]["event"] == "finished"
    assert events[-1]["status"] == "completed"
    assert events[-1]["exitCode"] == 0


def test_stale_lock_is_reaped_and_replaced(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AUDIT_HANDOFF_LOG_DIR", str(tmp_path))
    lock_id = log.lock_id_for(lane="codex-audit", repo="owner/repo", pr=14, head="feedface1234")
    stale_path = tmp_path / log.LOCKS_DIR / f"{lock_id}.json"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text(json.dumps({
        "lockId": lock_id,
        "lane": "codex-audit",
        "repo": "owner/repo",
        "pr": 14,
        "head": "feedface1234",
        "pid": 0,
    }))

    assert log.main([
        "start",
        "--lane", "codex-audit",
        "--repo", "owner/repo",
        "--pr", "14",
        "--head", "feedface1234",
        "--pid", str(os.getpid()),
    ]) == 0

    assert capsys.readouterr().out.strip() == lock_id
    events = _events(tmp_path)
    assert [event["event"] for event in events[:2]] == ["stale_lock_reaped", "started"]


def test_status_filters_active_locks(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AUDIT_HANDOFF_LOG_DIR", str(tmp_path))
    assert log.main([
        "start",
        "--lane", "codex-audit",
        "--repo", "owner/repo",
        "--pr", "15",
        "--head", "cab005e",
        "--pid", str(os.getpid()),
    ]) == 0
    capsys.readouterr()

    assert log.main(["status", "--repo", "owner/repo", "--pr", "15"]) == 0
    assert "codex-audit owner/repo#15" in capsys.readouterr().out

    assert log.main(["status", "--repo", "owner/repo", "--pr", "999"]) == 0
    assert "no active audit locks" in capsys.readouterr().out
