"""Tests for tools/greptile_audit_labeler.py.

Pure-function tests of severity parsing + verdict classification +
decision logic. The CLI entry point (`main`) is exercised by the
GitHub Action itself; we use the test cases here to pin the gates.

Fixture: the inline comment body used in `_REAL_P1_BODY` is captured
from `ssvlabs/ssv` PR #2835 (an open-source repo with Greptile
installed) — that's the only real Greptile payload I had captured
before the audit lane was built. Other test bodies are synthesized
to exercise specific code paths (P0, P2/P3-only, format drift,
opt-in gate, stale-HEAD gate).
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

from tools import greptile_audit_labeler as g


# ----- Real Greptile inline comment body (from ssvlabs/ssv#2835) -----

_REAL_P1_BODY = (
    '<a href="#"><img alt="P1" '
    'src="https://greptile-static-assets.s3.amazonaws.com/badges/p1.svg?v=7" '
    'align="top"></a> **Round timer never reset for future proposals**\n\n'
    'After `bumpToRound(msgRound)` is called on line 42, `i.State.Round` '
    'is already set to `msgRound`, so the condition `msgRound > i.State.Round` '
    'is always `false`. The `roundTimer.TimeoutForRound(msgRound)` call on '
    'line 45 is dead code — the timer is never reset when a future-round '
    'proposal is accepted, which can cause premature timeouts for the new '
    'round.\n\n'
    '`currentRound` is already captured at line 32 before the bump, so the '
    'fix is to use it in the comparison:\n\n'
    '```suggestion\n'
    '\ti.bumpToRound(msgRound)\n'
    '\ti.State.ProposalAcceptedForCurrentRound = msg\n'
    '\tif msgRound > currentRound {\n'
    '\t\ti.roundTimer.TimeoutForRound(msgRound)\n'
    '\t}\n'
    '```'
)


# ----- Severity parsing -----


def test_severity_of_real_p1_comment_via_alt():
    """The real Greptile P1 finding has both alt='P1' AND the badge
    URL — our parser prefers alt and gets the right answer."""
    assert g.severity_of(_REAL_P1_BODY) == 1


def test_severity_of_synthetic_p0_p2_p3():
    """Synthesize each severity by patching the badge tier."""
    for tier in (0, 1, 2, 3):
        synthetic = _REAL_P1_BODY.replace('alt="P1"', f'alt="P{tier}"').replace(
            'badges/p1.svg', f'badges/p{tier}.svg'
        )
        assert g.severity_of(synthetic) == tier, f"failed for tier {tier}"


def test_severity_of_alt_only_no_url():
    """If alt is present but URL is missing (e.g., the img is broken
    or unusual structure), still parse via alt."""
    body = '<img alt="P0"> finding text'
    assert g.severity_of(body) == 0


def test_severity_of_url_only_no_alt():
    """If alt is missing (custom Greptile theme or whatever) but the
    badge URL is there, fall back to URL pattern."""
    body = ('<img src="https://greptile-static-assets.s3.amazonaws.com/'
            'badges/p2.svg?v=7"> finding')
    assert g.severity_of(body) == 2


def test_severity_of_no_markers_returns_none():
    """A regular review comment with no badge → None (= format unknown).
    The classifier uses this to fail closed."""
    body = "This is just a comment with no severity markers."
    assert g.severity_of(body) is None


def test_severity_of_prefers_alt_when_both_present_disagree():
    """If alt says P1 but URL says P3 (Greptile bug?), trust alt
    because that's more semantic. Test contrived but pins the behavior."""
    body = (
        '<img alt="P1" '
        'src="https://greptile-static-assets.s3.amazonaws.com/badges/p3.svg">'
    )
    assert g.severity_of(body) == 1


# ----- Verdict classification -----


def test_classify_no_findings_is_done():
    """Zero inline comments → done (Greptile signaled clean)."""
    v = g.parse_review_comments([])
    assert v.classify() == "done"
    assert v.blocker_count == 0


def test_classify_p3_only_is_done():
    """P3 nit-only findings → done. Concerns are non-blocking."""
    v = g.parse_review_comments([{"body": '<img alt="P3"> nit'}])
    assert v.classify() == "done"
    assert v.p3_count == 1
    assert v.blocker_count == 0


def test_classify_p1_is_blocked():
    v = g.parse_review_comments([{"body": _REAL_P1_BODY}])
    assert v.classify() == "blocked"
    assert v.p1_count == 1
    assert v.blocker_count == 1


def test_classify_p0_is_blocked():
    """P0 (critical) also counts as a blocker — Codex's tightening."""
    v = g.parse_review_comments([{"body": '<img alt="P0"> critical'}])
    assert v.classify() == "blocked"
    assert v.p0_count == 1
    assert v.blocker_count == 1


def test_classify_mixed_severities_blocked_when_any_p0_or_p1():
    """Mixed: 1 P1 + 2 P2 + 3 P3 → BLOCKED (P1 trips it)."""
    comments = [
        {"body": '<img alt="P1"> blocker'},
        {"body": '<img alt="P2"> concern'},
        {"body": '<img alt="P2"> concern'},
        {"body": '<img alt="P3"> nit'},
        {"body": '<img alt="P3"> nit'},
        {"body": '<img alt="P3"> nit'},
    ]
    v = g.parse_review_comments(comments)
    assert v.classify() == "blocked"
    assert v.p1_count == 1 and v.p2_count == 2 and v.p3_count == 3


def test_classify_format_drift_fails_closed():
    """Codex's tightening — Gate 3: if inline comments exist but NO
    severity markers can be parsed (e.g., Greptile changed its badge
    format), fail closed to `needs` rather than auto-PASS."""
    comments = [
        {"body": "Some new format we don't recognize"},
        {"body": "Another comment with no badge"},
    ]
    v = g.parse_review_comments(comments)
    assert v.classify() == "needs"
    assert v.unparsed_count == 2


def test_classify_mixed_parsed_and_unparsed_fails_closed():
    """Codex review of PR #235 — P2 (round 1): ANY unparsed comment
    triggers fail-closed, not only the all-unparsed case. Rationale:
    an unparsed comment could be a malformed P0/P1 the regex missed;
    requeueing is conservative; trusting the parseable subset alone
    could silently ignore a drifted blocker.

    Updated from the original test (which asserted 'use what we got')
    — the new policy is stricter."""
    comments = [
        {"body": _REAL_P1_BODY},  # parseable P1
        {"body": "Unrecognizable format"},  # unparsed
    ]
    v = g.parse_review_comments(comments)
    # New policy: ANY unparsed → fail closed (needs).
    assert v.classify() == "needs"
    assert v.p1_count == 1
    assert v.unparsed_count == 1


def test_classify_one_unparsed_alone_still_fails_closed():
    """Single unparsed comment → needs (was already covered; restates
    after the policy change)."""
    v = g.parse_review_comments([{"body": "unknown format"}])
    assert v.classify() == "needs"


def test_classify_unparsed_alongside_p3_only_fails_closed():
    """P3-only + 1 unparsed → needs (the unparsed could be a hidden
    P0/P1). Used to be 'done' under old policy."""
    comments = [
        {"body": '<img alt="P3"> nit'},
        {"body": "drifted-format comment"},
    ]
    v = g.parse_review_comments(comments)
    assert v.classify() == "needs"


# ----- Decision logic (gates 1-4) -----


def _make_event(
    *,
    action: str = "submitted",
    author: str = "greptile-apps[bot]",
    pr_number: int = 17,
    review_commit: str = "deadbeef00000000",
    pr_labels: List[str] = None,
) -> Dict[str, Any]:
    return {
        "action": action,
        "review": {
            "id": 12345,
            "user": {"login": author},
            "commit_id": review_commit,
            "state": "commented",
            "body": "",
        },
        "pull_request": {
            "number": pr_number,
            "head": {"sha": review_commit},
            "labels": [{"name": l} for l in (pr_labels or [])],
        },
    }


def test_decision_skips_non_submitted_actions():
    """Only 'submitted' and 'edited' review actions trigger labels."""
    event = _make_event(action="dismissed")
    decision, reason = g.resolve_label_decision(
        event, pr_labels=[g.NEEDS_LABEL], current_head_sha="deadbeef00000000",
        review_comments=[],
    )
    assert decision is None
    assert "unsupported" in reason


def test_decision_skips_non_greptile_authors():
    """Codex authors, random user comments, etc. don't get labeled."""
    event = _make_event(author="codex-audit-bot")
    decision, reason = g.resolve_label_decision(
        event, pr_labels=[g.NEEDS_LABEL], current_head_sha="deadbeef00000000",
        review_comments=[],
    )
    assert decision is None
    assert "ignored review author" in reason


def test_decision_opt_in_gate_skips_pr_without_any_greptile_label():
    """Gate 1: only opted-in PRs participate. Greptile auto-reviews every
    PR but only ones with any greptile-audit-* label get processed."""
    event = _make_event()
    decision, reason = g.resolve_label_decision(
        event, pr_labels=[], current_head_sha="deadbeef00000000",
        review_comments=[{"body": _REAL_P1_BODY}],
    )
    assert decision is None
    assert "does not carry any greptile-audit-* label" in reason


def test_decision_opt_in_gate_accepts_done_label_too(monkeypatch):
    """Codex PR #235 — P2 (round 2): an already-processed PR keeps its
    `greptile-audit-done` (or `-blocked`) after the first review. A
    subsequent Greptile review on a new push MUST still be processed
    — otherwise the done/blocked label goes stale after a regression.
    """
    event = _make_event()
    decision, reason = g.resolve_label_decision(
        event,
        pr_labels=[g.DONE_LABEL],  # previously processed, no needs-
        current_head_sha="deadbeef00000000",
        review_comments=[{"body": _REAL_P1_BODY}],
    )
    # Must NOT be None — the PR is opted-in via the done label.
    assert decision is not None
    # Real P1 → flips to blocked.
    assert decision.add_label == g.BLOCKED_LABEL


def test_decision_opt_in_gate_accepts_blocked_label_too():
    """Symmetric: previously-blocked PR can flip to done on a fix."""
    event = _make_event()
    decision, reason = g.resolve_label_decision(
        event,
        pr_labels=[g.BLOCKED_LABEL],  # previously blocked
        current_head_sha="deadbeef00000000",
        review_comments=[],  # clean review now
    )
    assert decision is not None
    assert decision.add_label == g.DONE_LABEL


def test_decision_stale_head_gate_requeues():
    """Gate 2: review.commit_id differs from PR head → re-queue."""
    event = _make_event(review_commit="oldsha")
    decision, reason = g.resolve_label_decision(
        event, pr_labels=[g.NEEDS_LABEL], current_head_sha="newsha",
        review_comments=[{"body": _REAL_P1_BODY}],
    )
    assert decision is not None
    assert decision.add_label == g.NEEDS_LABEL
    assert decision.remove_labels == (g.DONE_LABEL, g.BLOCKED_LABEL)
    assert "stale review" in reason


def test_decision_clean_review_applies_done_label():
    """Greptile review with zero inline comments → greptile-audit-done."""
    event = _make_event()
    decision, reason = g.resolve_label_decision(
        event, pr_labels=[g.NEEDS_LABEL], current_head_sha="deadbeef00000000",
        review_comments=[],
    )
    assert decision.add_label == g.DONE_LABEL
    assert decision.remove_labels == (g.NEEDS_LABEL, g.BLOCKED_LABEL)
    # The "clean" wording is in `decision.reason`, not the function-return
    # `reason` (which is a short categorical label like "label done").
    assert "clean" in decision.reason.lower()


def test_decision_blocker_applies_blocked_label():
    """Real P1 → greptile-audit-blocked."""
    event = _make_event()
    decision, reason = g.resolve_label_decision(
        event, pr_labels=[g.NEEDS_LABEL], current_head_sha="deadbeef00000000",
        review_comments=[{"body": _REAL_P1_BODY}],
    )
    assert decision.add_label == g.BLOCKED_LABEL
    assert decision.remove_labels == (g.NEEDS_LABEL, g.DONE_LABEL)


def test_decision_format_drift_fails_closed_to_needs():
    """Gate 3: inline comments with no recognizable badges → re-queue."""
    event = _make_event()
    decision, reason = g.resolve_label_decision(
        event, pr_labels=[g.NEEDS_LABEL], current_head_sha="deadbeef00000000",
        review_comments=[{"body": "unrecognized format"}],
    )
    assert decision.add_label == g.NEEDS_LABEL
    assert "format unknown" in reason.lower() or "drifted" in reason.lower()


def test_decision_p2_p3_only_is_done():
    """P2/P3-only review → done. Concerns surfaced but don't gate."""
    event = _make_event()
    decision, reason = g.resolve_label_decision(
        event, pr_labels=[g.NEEDS_LABEL], current_head_sha="deadbeef00000000",
        review_comments=[
            {"body": '<img alt="P2"> concern A'},
            {"body": '<img alt="P3"> nit'},
        ],
    )
    assert decision.add_label == g.DONE_LABEL
    assert "P2=1" in decision.reason
    assert "P3=1" in decision.reason


# ----- Pagination -----


def test_fetch_review_comments_paginates_until_empty(monkeypatch):
    """Codex PR #235 P2 (round 1): fetch_review_comments must paginate.
    Without pagination, a PR with >100 inline comments could have its
    target Greptile review comments on page 2+ — and the labeler would
    label `greptile-audit-done` for a P0/P1 it never saw."""
    # Simulate two full pages + a partial third page.
    page_1 = [{"pull_request_review_id": 999, "body": "comment a"}] * 100
    page_2 = [{"pull_request_review_id": 999, "body": "comment b"}] * 100
    page_3 = [{"pull_request_review_id": 999, "body": "comment c"}] * 30
    pages = [page_1, page_2, page_3]

    fetch_calls: List[str] = []

    def fake_github_request(method, path, *, token, **_kw):
        fetch_calls.append(path)
        # Extract `&page=N` from the query string (regex anchored to `&`
        # so it doesn't accidentally match `per_page=100`'s `page=100`).
        import re
        m = re.search(r"&page=(\d+)", path)
        page_num = int(m.group(1)) if m else 1
        return pages[page_num - 1] if page_num - 1 < len(pages) else []

    monkeypatch.setattr(g, "github_request", fake_github_request)
    result = g.fetch_review_comments("owner/repo", 42, 999, token="x")
    # All 230 comments fetched + filtered (all matched review 999).
    assert len(result) == 230
    # Made 3 page fetches (stopped after partial page 3).
    assert len(fetch_calls) == 3
    assert all("per_page=100" in c for c in fetch_calls)
    assert "page=1" in fetch_calls[0]
    assert "page=2" in fetch_calls[1]
    assert "page=3" in fetch_calls[2]


def test_fetch_review_comments_raises_on_pagination_cap(monkeypatch):
    """Codex PR #235 — P2 (round 2): when the safety page cap is hit
    without exhausting the list, raise `ReviewCommentsTruncated`
    rather than silently returning partial data. The orchestrator
    catches it and applies `needs-greptile-audit` (fail closed).
    """
    import pytest
    # Return 100 entries for every page request — never reach a partial page
    monkeypatch.setattr(
        g, "github_request",
        lambda method, path, *, token=None, **_kw:
            [{"pull_request_review_id": 999, "body": "x"}] * 100,
    )
    with pytest.raises(g.ReviewCommentsTruncated) as exc_info:
        g.fetch_review_comments("o/r", 1, 999, token="x")
    assert "pagination cap" in str(exc_info.value).lower()


def test_fetch_review_comments_filters_by_review_id(monkeypatch):
    """Verify the filter still works with the new paginated fetch."""
    page_1 = [
        {"pull_request_review_id": 111, "body": "from review 111"},
        {"pull_request_review_id": 222, "body": "from review 222"},
        {"pull_request_review_id": 999, "body": "the one we want"},
    ]
    monkeypatch.setattr(g, "github_request",
                        lambda *a, **k: page_1 if "page=1" in a[1] else [])
    result = g.fetch_review_comments("owner/repo", 1, 999, token="x")
    assert len(result) == 1
    assert result[0]["body"] == "the one we want"


# ----- Fail-closed helper (missing review.id + truncation paths) -----


def _bootstrap_fail_closed_call(monkeypatch, *, review_user_login, pr_labels):
    """Drive `_fail_closed_requeue` with the same gates the production
    path runs, capturing whether `apply_label_decision` was called.

    Returns (rc, called) where rc is the function return code and
    called is True iff a label POST was issued.
    """
    called: List[bool] = []
    monkeypatch.setattr(
        g, "apply_label_decision",
        lambda repo, decision, token: called.append(True),
    )
    event = {
        "pull_request": {"number": 42},
        "review": {
            "id": 1234,
            "user": {"login": review_user_login},
            "commit_id": "abc123",
        },
    }
    rc = g._fail_closed_requeue(
        repo="o/r",
        event=event,
        review=event["review"],
        pr_labels=list(pr_labels),
        reason="test",
        token="x",
    )
    return rc, bool(called)


def test_fail_closed_requeue_applies_label_when_gates_pass(monkeypatch):
    """Greptile bot + opted-in PR: fail-closed re-applies needs-greptile-audit."""
    rc, called = _bootstrap_fail_closed_call(
        monkeypatch,
        review_user_login="greptile-apps[bot]",
        pr_labels=[g.NEEDS_LABEL],
    )
    assert rc == 0
    assert called is True


def test_fail_closed_requeue_skips_when_not_greptile_author(monkeypatch):
    """Non-Greptile reviewer: fail-closed does NOT touch labels even
    if the PR is opted in. Prevents arbitrary reviewers from forcing
    re-queue via a malformed/missing review.id event."""
    rc, called = _bootstrap_fail_closed_call(
        monkeypatch,
        review_user_login="some-human",
        pr_labels=[g.NEEDS_LABEL],
    )
    assert rc == 0
    assert called is False


def test_fail_closed_requeue_skips_when_not_opted_in(monkeypatch):
    """PR carries no greptile-audit-* label: fail-closed is a no-op.
    Greptile auto-fires on every PR but only opted-in PRs participate."""
    rc, called = _bootstrap_fail_closed_call(
        monkeypatch,
        review_user_login="greptile-apps[bot]",
        pr_labels=[],
    )
    assert rc == 0
    assert called is False


def test_fail_closed_requeue_accepts_done_or_blocked_as_opt_in(monkeypatch):
    """After the first Greptile review flips needs→done, a subsequent
    truncated/missing-id review should still requeue via fail-closed.
    Mirror of the Gate-1 fix that accepts any greptile-audit-* label."""
    for prior in (g.DONE_LABEL, g.BLOCKED_LABEL):
        rc, called = _bootstrap_fail_closed_call(
            monkeypatch,
            review_user_login="greptile-apps[bot]",
            pr_labels=[prior],
        )
        assert rc == 0, f"prior={prior}"
        assert called is True, f"prior={prior}"


# ----- Author authority (env var override) -----


def test_default_authors_when_env_var_unset(monkeypatch):
    monkeypatch.delenv("GREPTILE_BOT_AUTHORS", raising=False)
    importlib.reload(g)
    assert "greptile-apps[bot]" in g.GREPTILE_REVIEW_AUTHORS
    assert "greptile-apps" in g.GREPTILE_REVIEW_AUTHORS


def test_default_authors_when_env_var_empty_string(monkeypatch):
    """Same Codex-meta-review-#234 fix pattern — empty env doesn't
    defeat default."""
    monkeypatch.setenv("GREPTILE_BOT_AUTHORS", "")
    importlib.reload(g)
    assert "greptile-apps[bot]" in g.GREPTILE_REVIEW_AUTHORS


def test_custom_authors_via_env_override(monkeypatch):
    monkeypatch.setenv("GREPTILE_BOT_AUTHORS", "my-greptile-org/reviewer")
    importlib.reload(g)
    assert g.GREPTILE_REVIEW_AUTHORS == {"my-greptile-org/reviewer"}
    assert "greptile-apps[bot]" not in g.GREPTILE_REVIEW_AUTHORS
