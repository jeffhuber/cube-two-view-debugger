from __future__ import annotations

from tools import gitar_audit_labeler as gitar


# Real Gitar "Code Review" dashboard comment (captured from cube-snap
# #270). The verdict lives in the <kbd> badge after "Code Review".
APPROVED_COMMENT = """<details>
<summary><b>Code Review</b> <kbd>✅ Approved</kbd></summary>

Introduces an advisory-only rule configuration for Gitar to ensure
human-led code reviews. No issues found.

</details>

<sub>Was this helpful? React with 👍 / 👎 | [Gitar](https://gitar.ai)</sub>
"""

# Real in-progress prefix (captured from ctvd #411) — Gitar edits the
# comment to add the Code Review block once done. On its own this is NOT
# a verdict comment and must be skipped.
IN_PROGRESS_COMMENT = (
    '<kbd><img src="https://raw.githubusercontent.com/gitarcode/.github'
    '/main/assets/gitar-spin.svg" align="center"> Reviewing your code</kbd>'
)

# A hypothetical non-approved verdict (exact badge text TBD until a real
# one is observed; the classifier treats any non-"Approved" badge as
# blocked).
CHANGES_COMMENT = (
    "<details>\n<summary><b>Code Review</b> <kbd>❌ Changes Requested</kbd>"
    "</summary>\n\nFound a null deref.\n\n</details>"
)

# Same approved comment but Gitar also appended our authoritative trailer.
APPROVED_WITH_TRAILER = APPROVED_COMMENT + "\n<!-- GITAR_AUDIT_STATE: gitar-audit-done -->"


def _event(body, *, author="gitar-bot", action="created", is_pr=True, number=42):
    issue = {"number": number}
    if is_pr:
        issue["pull_request"] = {"url": "https://api.github.com/.../pulls/42"}
    return {
        "action": action,
        "issue": issue,
        "comment": {"user": {"login": author}, "body": body},
    }


# ----- badge parsing -----


def test_parse_badge_approved():
    assert gitar.parse_code_review_badge(APPROVED_COMMENT) == "✅ Approved"


def test_parse_badge_strips_nested_tags():
    body = "<summary><b>Code Review</b> <kbd><img src='x'> Changes Requested</kbd></summary>"
    assert gitar.parse_code_review_badge(body) == "Changes Requested"


def test_parse_badge_none_for_in_progress_only():
    # The "Reviewing your code" badge is not preceded by "Code Review</b>".
    assert gitar.parse_code_review_badge(IN_PROGRESS_COMMENT) is None


def test_parse_badge_none_for_non_review_comment():
    assert gitar.parse_code_review_badge("just a normal reply, thanks!") is None


def test_parse_badge_empty_when_tag_empty():
    assert gitar.parse_code_review_badge("<b>Code Review</b> <kbd></kbd>") == ""


def test_classify_badge_values():
    assert gitar.classify_badge(None) is None
    assert gitar.classify_badge("") == "needs"
    assert gitar.classify_badge("✅ Approved") == "done"
    assert gitar.classify_badge("Approved") == "done"
    assert gitar.classify_badge("❌ Changes Requested") == "blocked"
    assert gitar.classify_badge("Commented") == "blocked"


# ----- trailer parsing -----


def test_parse_trailer_values():
    assert gitar.parse_state_trailer("x <!-- GITAR_AUDIT_STATE: gitar-audit-done -->") == "done"
    assert gitar.parse_state_trailer("<!-- GITAR_AUDIT_STATE: gitar-audit-blocked -->") == "blocked"
    assert gitar.parse_state_trailer("<!-- GITAR_AUDIT_STATE: needs-gitar-audit -->") == "needs"


def test_parse_trailer_last_wins():
    body = (
        "<!-- GITAR_AUDIT_STATE: gitar-audit-blocked -->\n"
        "<!-- GITAR_AUDIT_STATE: gitar-audit-done -->"
    )
    assert gitar.parse_state_trailer(body) == "done"


def test_parse_trailer_malformed_is_none():
    # A non-canonical value must NOT match (falls through to badge parse).
    assert gitar.parse_state_trailer("<!-- GITAR_AUDIT_STATE: maybe -->") is None
    assert gitar.parse_state_trailer("no trailer here") is None


# ----- combined classifier (trailer-first) -----


def test_classify_prefers_trailer_over_badge():
    # Badge says Approved but trailer says blocked → trailer wins.
    body = APPROVED_COMMENT + "\n<!-- GITAR_AUDIT_STATE: gitar-audit-blocked -->"
    status, detail = gitar.classify_gitar_comment(body)
    assert status == "blocked"
    assert "trailer" in detail


def test_classify_falls_back_to_badge():
    status, detail = gitar.classify_gitar_comment(APPROVED_COMMENT)
    assert status == "done"
    assert "badge" in detail


def test_classify_none_when_no_verdict():
    status, _ = gitar.classify_gitar_comment(IN_PROGRESS_COMMENT)
    assert status is None


# ----- resolve_label_decision -----


def test_resolve_done_via_badge():
    decision, reason = gitar.resolve_label_decision(_event(APPROVED_COMMENT))
    assert decision is not None
    assert decision.add_label == gitar.DONE_LABEL
    assert decision.remove_labels == (gitar.NEEDS_LABEL, gitar.BLOCKED_LABEL)
    assert decision.issue_number == 42


def test_resolve_done_via_trailer():
    decision, _ = gitar.resolve_label_decision(_event(APPROVED_WITH_TRAILER))
    assert decision is not None
    assert decision.add_label == gitar.DONE_LABEL


def test_resolve_blocked():
    decision, _ = gitar.resolve_label_decision(_event(CHANGES_COMMENT))
    assert decision is not None
    assert decision.add_label == gitar.BLOCKED_LABEL


def test_resolve_needs_on_empty_badge():
    decision, _ = gitar.resolve_label_decision(
        _event("<b>Code Review</b> <kbd></kbd>")
    )
    assert decision is not None
    assert decision.add_label == gitar.NEEDS_LABEL


def test_resolve_skip_non_gitar_author():
    decision, reason = gitar.resolve_label_decision(
        _event(APPROVED_COMMENT, author="somebody-else")
    )
    assert decision is None
    assert "author" in reason


def test_resolve_skip_issue_not_pr():
    decision, reason = gitar.resolve_label_decision(
        _event(APPROVED_COMMENT, is_pr=False)
    )
    assert decision is None
    assert "pull request" in reason


def test_resolve_skip_in_progress_comment():
    decision, reason = gitar.resolve_label_decision(_event(IN_PROGRESS_COMMENT))
    assert decision is None
    assert "no Gitar verdict" in reason


def test_resolve_skip_unsupported_action():
    decision, reason = gitar.resolve_label_decision(
        _event(APPROVED_COMMENT, action="deleted")
    )
    assert decision is None
    assert "action" in reason


def test_resolve_handles_edited_action():
    # Gitar edits its comment from in-progress to the final verdict.
    decision, _ = gitar.resolve_label_decision(
        _event(APPROVED_COMMENT, action="edited")
    )
    assert decision is not None
    assert decision.add_label == gitar.DONE_LABEL


def test_bot_author_override(monkeypatch):
    assert gitar.is_gitar_comment_author("gitar-bot")
    assert not gitar.is_gitar_comment_author("random-user")
