from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools import request_review


def test_build_body_wraps_validation_without_shell_execution():
    body = request_review.build_body(
        reviewer="Claude",
        head="abc123",
        scope="Check literal markdown handling.",
        validations=["echo `do-not-run` && echo $(do-not-run-either)"],
        notes=["This is generated inside Python, not a shell heredoc."],
    )

    assert "Claude review requested for current head `abc123`" in body
    assert "- `echo `do-not-run` && echo $(do-not-run-either)`" in body
    assert "This is generated inside Python" in body


def test_request_review_posts_comment_adds_label_and_logs_event(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_HANDOFF_LOG_DIR", str(tmp_path / "audit-log"))
    posted = {}
    gh_calls = []

    def fake_post_comment(repo: str, pr: int, body: str) -> None:
        posted["repo"] = repo
        posted["pr"] = pr
        posted["body"] = body

    def fake_run(args, **kwargs):
        gh_calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    request_review.request_review(
        lane="claude-review",
        repo="jeffhuber/cube-two-view-debugger",
        pr=365,
        head="abc123",
        label="needs-claude-review",
        reviewer="Claude",
        scope="Review the safe helper.",
        validations=["pytest tests/test_request_review.py"],
        notes=[],
        actor="codex",
        run=fake_run,
        post_comment=fake_post_comment,
    )

    assert posted["repo"] == "jeffhuber/cube-two-view-debugger"
    assert posted["pr"] == 365
    assert "Review the safe helper" in posted["body"]
    assert gh_calls == [
        [
            "gh",
            "issue",
            "edit",
            "365",
            "--repo",
            "jeffhuber/cube-two-view-debugger",
            "--add-label",
            "needs-claude-review",
        ]
    ]
    events = [
        json.loads(line)
        for line in Path(tmp_path / "audit-log" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["event"] == "review_requested"
    assert events[-1]["lane"] == "claude-review"
    assert events[-1]["actor"] == "codex"
    assert events[-1]["head"] == "abc123"
