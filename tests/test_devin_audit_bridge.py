from tools.devin_audit_bridge import (
    NEEDS_LABEL,
    build_payload,
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
