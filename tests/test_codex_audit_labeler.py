"""Tests for tools/codex_audit_labeler.py.

Specifically guards the two Codex-meta-review fixes for #234:
  - empty CODEX_BOT_AUTHORS env var must fall back to defaults (P2)
  - non-PR issue_comment events must not 404 trying to fetch /pulls/{n} (P3)
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict

import pytest


def test_default_authors_used_when_env_var_unset(monkeypatch):
    """Direct unset env: defaults apply."""
    monkeypatch.delenv("CODEX_BOT_AUTHORS", raising=False)
    # Need to re-import so the module-level CODEX_COMMENT_AUTHORS picks up
    # the changed env.
    from tools import codex_audit_labeler
    importlib.reload(codex_audit_labeler)
    assert "codex-audit-bot" in codex_audit_labeler.CODEX_COMMENT_AUTHORS
    assert "codex-audit-bot[bot]" in codex_audit_labeler.CODEX_COMMENT_AUTHORS


def test_default_authors_used_when_env_var_empty_string(monkeypatch):
    """Codex meta-review #234 P2: GitHub Actions exports an empty string
    when the repo var is unset. `os.environ.get(name, default)` returns
    "" not the default. Fix uses `... or _DEFAULT_AUTHORS` to fall back.
    Without the fix, the trust set would be empty and the labeler would
    silently ignore every Codex-bot comment.
    """
    monkeypatch.setenv("CODEX_BOT_AUTHORS", "")
    from tools import codex_audit_labeler
    importlib.reload(codex_audit_labeler)
    # Bug repro check: this should NOT be the empty set.
    assert codex_audit_labeler.CODEX_COMMENT_AUTHORS != set()
    assert "codex-audit-bot" in codex_audit_labeler.CODEX_COMMENT_AUTHORS


def test_custom_authors_via_env_override_defaults(monkeypatch):
    """Non-empty env value still wins over defaults."""
    monkeypatch.setenv("CODEX_BOT_AUTHORS", "jeffhuber,custom-bot[bot]")
    from tools import codex_audit_labeler
    importlib.reload(codex_audit_labeler)
    assert codex_audit_labeler.CODEX_COMMENT_AUTHORS == {"jeffhuber", "custom-bot[bot]"}
    # Defaults are NOT in the set when overridden
    assert "codex-audit-bot" not in codex_audit_labeler.CODEX_COMMENT_AUTHORS


def test_main_skips_pr_head_fetch_on_non_pr_comment(monkeypatch, tmp_path):
    """Codex meta-review #234 P3: when the event is a comment on a regular
    (non-PR) issue, `main()` must NOT call `/pulls/{n}` — that 404s and
    fails the workflow. The fix checks `issue.pull_request` before
    fetching.
    """
    import json
    monkeypatch.setenv("CODEX_BOT_AUTHORS", "codex-audit-bot")
    from tools import codex_audit_labeler
    importlib.reload(codex_audit_labeler)

    # Event: a comment on a regular issue (no `pull_request` key under `issue`)
    event = {
        "action": "created",
        "issue": {"number": 999, "title": "Regular issue"},
        "comment": {"body": "irrelevant", "user": {"login": "anyone"}},
    }
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event))

    fetch_calls = []

    def track_fetch(*args, **kwargs):
        fetch_calls.append(args)
        # If reached, return a dummy PR — but we expect this NOT to be called
        return {"head": {"sha": "dummy"}}

    monkeypatch.setattr(codex_audit_labeler, "fetch_pull_request", track_fetch)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    rc = codex_audit_labeler.main()
    # Should exit 0 (no label change needed) without calling the PR fetch.
    assert rc == 0
    assert fetch_calls == [], "must not call /pulls/{n} on non-PR comments"


def test_classify_uses_last_trailer_when_body_quotes_an_earlier_one(monkeypatch):
    """Codex round 3 of #234 — P2: when a comment body quotes the trailer
    text earlier (e.g., the comment was reviewing the protocol doc or
    an earlier review comment), `search()` would return the FIRST match.
    A real BLOCKED verdict could be mislabeled as done because the
    quoted trailer earlier in the body was `codex-audit-done`.

    Fix uses `finditer` and takes the last match — which is what
    `format_comment` actually appends as the final line.
    """
    monkeypatch.setenv("CODEX_BOT_AUTHORS", "codex-audit-bot")
    from tools import codex_audit_labeler
    importlib.reload(codex_audit_labeler)

    # Body that quotes a "done" trailer earlier (e.g., in a code block
    # showing example trailers), with the REAL trailer at the end
    # marking BLOCKED.
    body = """## Codex audit

Verdict: BLOCKED

Some example trailers documented in the protocol:
- `<!-- CODEX_AUDIT_STATE: codex-audit-done -->` for PASS
- `<!-- CODEX_AUDIT_STATE: codex-audit-blocked -->` for BLOCKED

[P2] Real finding here.

<!-- CODEX_AUDIT_STATE: codex-audit-blocked -->
"""
    result = codex_audit_labeler.classify_audit_comment(body)
    assert result == "blocked", (
        f"trailer-match-first bug regressed: got {result!r}, expected 'blocked'"
    )


def test_classify_uses_last_trailer_with_three_trailers(monkeypatch):
    """Stress test: three trailers in the body — the LAST one wins."""
    monkeypatch.setenv("CODEX_BOT_AUTHORS", "codex-audit-bot")
    from tools import codex_audit_labeler
    importlib.reload(codex_audit_labeler)

    body = (
        "<!-- CODEX_AUDIT_STATE: codex-audit-done -->\n"
        "<!-- CODEX_AUDIT_STATE: needs-codex-audit -->\n"
        "<!-- CODEX_AUDIT_STATE: codex-audit-blocked -->\n"
    )
    assert codex_audit_labeler.classify_audit_comment(body) == "blocked"


def test_main_does_fetch_pr_head_on_pr_comment(monkeypatch, tmp_path):
    """Inverse: when the comment IS on a PR, the fetch still happens."""
    import json
    monkeypatch.setenv("CODEX_BOT_AUTHORS", "codex-audit-bot")
    from tools import codex_audit_labeler
    importlib.reload(codex_audit_labeler)

    event = {
        "action": "created",
        "issue": {
            "number": 234,
            "title": "Codex audit lane",
            "pull_request": {"url": "..."},  # marker that this IS a PR
        },
        "comment": {"body": "irrelevant", "user": {"login": "anyone"}},
    }
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event))

    fetch_calls = []

    def track_fetch(repo, issue_number, *, token=None, tokens=None):
        fetch_calls.append((repo, issue_number))
        assert token is None
        assert tokens == (codex_audit_labeler.GitHubToken("GITHUB_TOKEN", "test-token"),)
        return {"head": {"sha": "deadbeef"}}

    monkeypatch.setattr(codex_audit_labeler, "fetch_pull_request", track_fetch)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    codex_audit_labeler.main()
    assert fetch_calls == [("owner/repo", 234)]


def test_apply_label_decision_falls_back_when_label_pat_is_forbidden(monkeypatch):
    """If CODEX_AUDIT_LABEL_TOKEN exists but lacks Issues:write, the
    labeler should retry with the built-in GITHUB_TOKEN instead of failing
    before the labels can reflect the audit comment.
    """
    from tools import codex_audit_labeler

    calls = []

    def fake_request(method, path, *, token, body=None, allow_missing=False):
        calls.append((method, path, token, body, allow_missing))
        if token == "bad-pat":
            raise codex_audit_labeler.GitHubRequestError(method, path, 403, "forbidden")
        return None

    monkeypatch.setattr(codex_audit_labeler, "github_request", fake_request)

    decision = codex_audit_labeler.LabelDecision(
        issue_number=279,
        add_label=codex_audit_labeler.BLOCKED_LABEL,
        remove_labels=(),
        reviewed_sha="deadbeef",
        reason="audit blocked",
    )
    codex_audit_labeler.apply_label_decision(
        "owner/repo",
        decision,
        tokens=(
            codex_audit_labeler.GitHubToken("CODEX_AUDIT_LABEL_TOKEN", "bad-pat"),
            codex_audit_labeler.GitHubToken("GITHUB_TOKEN", "github-token"),
        ),
    )

    assert calls == [
        (
            "POST",
            "/repos/owner/repo/issues/279/labels",
            "bad-pat",
            {"labels": [codex_audit_labeler.BLOCKED_LABEL]},
            False,
        ),
        (
            "POST",
            "/repos/owner/repo/issues/279/labels",
            "github-token",
            {"labels": [codex_audit_labeler.BLOCKED_LABEL]},
            False,
        ),
    ]


def test_fallback_warning_says_when_no_tokens_remain(monkeypatch, capsys):
    from tools import codex_audit_labeler

    def fake_request(method, path, *, token, body=None, allow_missing=False):
        raise codex_audit_labeler.GitHubRequestError(method, path, 403, "forbidden")

    monkeypatch.setattr(codex_audit_labeler, "github_request", fake_request)

    with pytest.raises(codex_audit_labeler.GitHubRequestError):
        codex_audit_labeler.github_request_with_fallback(
            "GET",
            "/repos/owner/repo/pulls/1",
            tokens=(codex_audit_labeler.GitHubToken("GITHUB_TOKEN", "bad"),),
        )

    stderr = capsys.readouterr().err
    assert "GITHUB_TOKEN; no more tokens" in stderr
    assert "GITHUB_TOKEN; trying next token" not in stderr


def test_env_tokens_prefer_label_pat_then_github_token(monkeypatch):
    from tools import codex_audit_labeler

    monkeypatch.setenv("CODEX_AUDIT_LABEL_TOKEN", "pat")
    monkeypatch.setenv("GITHUB_TOKEN", "builtin")
    assert codex_audit_labeler.github_tokens_from_env() == (
        codex_audit_labeler.GitHubToken("CODEX_AUDIT_LABEL_TOKEN", "pat"),
        codex_audit_labeler.GitHubToken("GITHUB_TOKEN", "builtin"),
    )
