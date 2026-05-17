import json

from tools import devin_audit_bridge
from tools.devin_audit_bridge import (
    NEEDS_LABEL,
    build_payload,
    devin_already_reviewed_sha,
    resolve_audit_request,
)


def make_pr(*, labels=None, sha="abc123", number=17, state="open"):
    return {
        "number": number,
        "html_url": f"https://github.com/jeffhuber/cube-two-view-debugger/pull/{number}",
        "title": "Test PR",
        "state": state,
        "labels": [{"name": label} for label in (labels or [])],
        "head": {
            "sha": sha,
            "ref": "codex/test",
            "repo": {"full_name": "jeffhuber/cube-two-view-debugger"},
        },
        "base": {"ref": "main"},
    }


def fetcher(pr):
    def _fetch(_number):
        return pr

    return _fetch


def test_labeled_needs_devin_audit_dispatches():
    pr = make_pr(labels=[NEEDS_LABEL])
    request, reason = resolve_audit_request(
        event_name="pull_request_target",
        action="labeled",
        actor="codex",
        event={"label": {"name": NEEDS_LABEL}, "pull_request": pr},
        fetch_pull_request=fetcher(pr),
    )

    assert reason == "dispatch"
    assert request is not None
    assert request.pull_request["number"] == 17


def test_synchronize_dispatches_when_label_is_still_present():
    pr = make_pr(labels=[NEEDS_LABEL])
    request, reason = resolve_audit_request(
        event_name="pull_request_target",
        action="synchronize",
        actor="codex",
        event={"pull_request": pr},
        fetch_pull_request=fetcher(pr),
    )

    assert reason == "dispatch"
    assert request is not None


def test_synchronize_skips_without_needs_devin_audit_label():
    pr = make_pr(labels=["devin-audit-done"])
    request, reason = resolve_audit_request(
        event_name="pull_request_target",
        action="synchronize",
        actor="codex",
        event={"pull_request": pr},
        fetch_pull_request=fetcher(pr),
    )

    assert request is None
    assert "label is absent" in reason


def test_trusted_devin_audit_comment_dispatches():
    pr = make_pr(labels=[])
    request, reason = resolve_audit_request(
        event_name="issue_comment",
        action="created",
        actor="jeffhuber",
        event={
            "issue": {"number": 17, "pull_request": {"url": "https://api.github.com/pr"}},
            "comment": {
                "body": "Please run @devin audit again.",
                "author_association": "OWNER",
            },
        },
        fetch_pull_request=fetcher(pr),
    )

    assert reason == "dispatch"
    assert request is not None
    assert request.trigger["event"] == "issue_comment"
    assert request.force is False


def test_trusted_devin_audit_force_comment_dispatches_with_force():
    pr = make_pr(labels=[])
    request, reason = resolve_audit_request(
        event_name="issue_comment",
        action="created",
        actor="jeffhuber",
        event={
            "issue": {"number": 17, "pull_request": {"url": "https://api.github.com/pr"}},
            "comment": {
                "body": "Please run @devin audit force.",
                "author_association": "OWNER",
            },
        },
        fetch_pull_request=fetcher(pr),
    )

    assert reason == "dispatch"
    assert request is not None
    assert request.force is True


def test_devin_audit_comment_requires_trusted_commenter():
    pr = make_pr(labels=[])
    request, reason = resolve_audit_request(
        event_name="issue_comment",
        action="created",
        actor="internet-stranger",
        event={
            "issue": {"number": 17, "pull_request": {"url": "https://api.github.com/pr"}},
            "comment": {
                "body": "@devin audit",
                "author_association": "NONE",
            },
        },
        fetch_pull_request=fetcher(pr),
    )

    assert request is None
    assert "untrusted commenter" in reason


def test_ignored_bot_actor_skips():
    pr = make_pr(labels=[NEEDS_LABEL])
    request, reason = resolve_audit_request(
        event_name="pull_request_target",
        action="labeled",
        actor="devin-ai-integration[bot]",
        event={"label": {"name": NEEDS_LABEL}, "pull_request": pr},
        fetch_pull_request=fetcher(pr),
    )

    assert request is None
    assert "ignored actor" in reason


def test_payload_contains_sha_dedupe_key_and_review_instructions():
    pr = make_pr(labels=[NEEDS_LABEL], sha="def456", number=42)
    request, _reason = resolve_audit_request(
        event_name="pull_request_target",
        action="labeled",
        actor="codex",
        event={"label": {"name": NEEDS_LABEL}, "pull_request": pr},
        fetch_pull_request=fetcher(pr),
    )

    payload = build_payload("jeffhuber/cube-two-view-debugger", request)

    assert payload["dedupe_key"] == "jeffhuber/cube-two-view-debugger#42@def456"
    assert payload["pull_request"]["head_sha"] == "def456"
    assert "On pass, remove needs-devin-audit" in payload["instructions"]


def test_devin_already_reviewed_sha_detects_same_sha_in_devin_comment():
    comments = [
        {
            "user": {"login": "devin-ai-integration"},
            "body": "Audit pass on latest head abc123: no blockers.",
        }
    ]

    assert devin_already_reviewed_sha("abc123", comments)


def test_devin_already_reviewed_sha_ignores_different_sha():
    comments = [
        {
            "user": {"login": "devin-ai-integration"},
            "body": "Audit pass on latest head oldsha: no blockers.",
        }
    ]

    assert not devin_already_reviewed_sha("abc123", comments)


def test_devin_already_reviewed_sha_ignores_non_devin_comments():
    comments = [
        {
            "user": {"login": "jeffhuber"},
            "body": "Mentioning abc123 should not dedupe Devin.",
        }
    ]

    assert not devin_already_reviewed_sha("abc123", comments)


def test_run_skips_webhook_when_devin_already_reviewed_current_sha(tmp_path, monkeypatch):
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "label": {"name": NEEDS_LABEL},
                "pull_request": make_pr(labels=[NEEDS_LABEL], sha="abc123"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request_target")
    monkeypatch.setenv("GITHUB_EVENT_ACTION", "labeled")
    monkeypatch.setenv("GITHUB_REPOSITORY", "jeffhuber/cube-two-view-debugger")
    monkeypatch.setenv("GITHUB_ACTOR", "codex")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("DEVIN_WEBHOOK_URL", "https://devin.example/webhook")
    monkeypatch.setenv("DEVIN_WEBHOOK_SECRET", "secret")

    def fake_paginated(path, token):
        if path.endswith("/issues/17/comments"):
            return [
                {
                    "user": {"login": "devin-ai-integration"},
                    "body": "Audit pass on latest head abc123: no blockers.",
                }
            ]
        return []

    def fail_post_webhook(_url, _secret, _payload):
        raise AssertionError("webhook should not be posted for an already reviewed SHA")

    monkeypatch.setattr(devin_audit_bridge, "github_api_paginated", fake_paginated)
    monkeypatch.setattr(devin_audit_bridge, "post_webhook", fail_post_webhook)

    assert devin_audit_bridge.run() == 0


def test_run_force_comment_posts_even_when_devin_already_reviewed_current_sha(
    tmp_path, monkeypatch
):
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps(
            {
                "issue": {
                    "number": 17,
                    "pull_request": {"url": "https://api.github.com/pr"},
                },
                "comment": {
                    "body": "@devin audit force",
                    "author_association": "OWNER",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_EVENT_NAME", "issue_comment")
    monkeypatch.setenv("GITHUB_EVENT_ACTION", "created")
    monkeypatch.setenv("GITHUB_REPOSITORY", "jeffhuber/cube-two-view-debugger")
    monkeypatch.setenv("GITHUB_ACTOR", "jeffhuber")
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("DEVIN_WEBHOOK_URL", "https://devin.example/webhook")
    monkeypatch.setenv("DEVIN_WEBHOOK_SECRET", "secret")

    calls = []

    def fake_github_api_json(path, token):
        assert path == "/repos/jeffhuber/cube-two-view-debugger/pulls/17"
        return make_pr(labels=[NEEDS_LABEL], sha="abc123")

    def fake_paginated(_path, _token):
        raise AssertionError("force dispatch should skip dedupe lookups")

    def fake_post_webhook(url, secret, payload):
        calls.append((url, secret, payload))
        return 202, "accepted"

    monkeypatch.setattr(devin_audit_bridge, "github_api_json", fake_github_api_json)
    monkeypatch.setattr(devin_audit_bridge, "github_api_paginated", fake_paginated)
    monkeypatch.setattr(devin_audit_bridge, "post_webhook", fake_post_webhook)

    assert devin_audit_bridge.run() == 0
    assert len(calls) == 1
    assert calls[0][2]["dedupe_key"] == "jeffhuber/cube-two-view-debugger#17@abc123"
