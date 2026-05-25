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


def test_record_appends_event_without_lock(tmp_path, monkeypatch):
    """`record` writes a single event with no lock side-effects."""
    monkeypatch.setenv("AUDIT_HANDOFF_LOG_DIR", str(tmp_path))
    assert log.main([
        "record",
        "--lane", "claude-review",
        "--repo", "owner/repo",
        "--pr", "172",
        "--head", "6abe1c7abcdef",
        "--event", "finished",
        "--verdict", "pass",
        "--actor", "claude",
        "--notes", "PR PASS via post_review.sh",
    ]) == 0

    events = _events(tmp_path)
    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "finished"
    assert ev["lane"] == "claude-review"
    assert ev["repo"] == "owner/repo"
    assert ev["pr"] == 172
    assert ev["head"] == "6abe1c7abcdef"
    assert ev["verdict"] == "pass"
    assert ev["actor"] == "claude"
    assert ev["notes"] == "PR PASS via post_review.sh"
    # No lock should have been created — record is lock-free.
    locks_dir = tmp_path / log.LOCKS_DIR
    assert not locks_dir.exists() or not list(locks_dir.glob("*.json"))


def test_record_omits_optional_fields_when_not_supplied(tmp_path, monkeypatch):
    """Optional fields (verdict, notes) are absent from the event when not set,
    keeping the JSONL minimal for the common case."""
    monkeypatch.setenv("AUDIT_HANDOFF_LOG_DIR", str(tmp_path))
    assert log.main([
        "record",
        "--lane", "codex-review",
        "--repo", "owner/repo",
        "--pr", "200",
        "--head", "deadbeef1234",
        "--event", "started",
    ]) == 0

    events = _events(tmp_path)
    ev = events[0]
    assert ev["event"] == "started"
    assert "verdict" not in ev
    assert "notes" not in ev


def test_record_finished_event_matches_monitor_filter_shape(tmp_path, monkeypatch):
    """The on-disk JSON for a `record --event finished` line must contain the
    literal substring `"event": "finished"` so that the Monitor filter used in
    practice (`grep -E '"event": ?"(finished|duplicate_refused)"'`) catches it
    transparently — same shape as `finish_lock`'s output."""
    monkeypatch.setenv("AUDIT_HANDOFF_LOG_DIR", str(tmp_path))
    assert log.main([
        "record",
        "--lane", "claude-review",
        "--repo", "owner/repo",
        "--pr", "1",
        "--head", "abcdef",
        "--event", "finished",
    ]) == 0

    raw = (tmp_path / log.EVENTS_FILE).read_text(encoding="utf-8")
    assert '"event": "finished"' in raw, (
        "Monitor grep pattern depends on this exact substring; if you "
        "change the serializer format, update the filter in the queue-sweep "
        "protocol section of CLAUDE.md."
    )
