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


def test_parse_badge_tolerates_kbd_attributes():
    body = "<b>Code Review</b> <kbd class='verdict'>✅ Approved</kbd>"
    assert gitar.parse_code_review_badge(body) == "✅ Approved"


def test_has_code_review_block():
    assert gitar.has_code_review_block(APPROVED_COMMENT)
    assert not gitar.has_code_review_block(IN_PROGRESS_COMMENT)
    assert not gitar.has_code_review_block("just a normal reply")


def test_classify_badge_values():
    assert gitar.classify_badge(None) is None
    assert gitar.classify_badge("") == "needs"
    assert gitar.classify_badge("✅ Approved") == "done"
    assert gitar.classify_badge("Approved") == "done"
    assert gitar.classify_badge("APPROVED") == "done"
    assert gitar.classify_badge("❌ Changes Requested") == "blocked"
    assert gitar.classify_badge("Commented") == "blocked"


def test_classify_badge_substring_approved_is_blocked():
    # "approved" as a substring must NOT classify as done (Codex P2):
    # these are non-approvals and must fail closed to blocked.
    assert gitar.classify_badge("Not Approved") == "blocked"
    assert gitar.classify_badge("Unapproved") == "blocked"
    assert gitar.classify_badge("⚠️ Approved with concerns") == "blocked"


# ----- code stripping (badge-spoof defense) -----


def test_strip_code_blanks_fences_and_spans():
    # Code is blanked to spaces (newlines preserved) so quoted markup
    # inside it cannot be read as Gitar's own verdict.
    assert "Approved" not in gitar._strip_code("`✅ Approved`")
    assert "Approved" not in gitar._strip_code("``✅ Approved``")
    assert "Approved" not in gitar._strip_code("```\n✅ Approved\n```")
    assert "Approved" not in gitar._strip_code("~~~\n✅ Approved\n~~~")


# ----- combined classifier (native badge) -----


def test_classify_uses_badge():
    status, detail = gitar.classify_gitar_comment(APPROVED_COMMENT)
    assert status == "done"
    assert "badge" in detail


def test_classify_blocked_badge():
    status, _ = gitar.classify_gitar_comment(CHANGES_COMMENT)
    assert status == "blocked"


def test_classify_none_when_no_verdict():
    status, _ = gitar.classify_gitar_comment(IN_PROGRESS_COMMENT)
    assert status is None


def test_classify_fail_closed_on_badge_drift():
    # Code Review block present but the <kbd> badge markup drifted (here a
    # <span> instead of <kbd>): must fail closed to needs, not silently
    # skip and leave a stale label (Codex P2 on cube-snap#271).
    body = "<details>\n<summary><b>Code Review</b> <span>✅ Approved</span></summary>\n</details>"
    status, detail = gitar.classify_gitar_comment(body)
    assert status == "needs"
    assert "unparseable" in detail


def test_classify_ignores_fenced_badge_spoof():
    # A blocked review whose body quotes a fake approved Code Review badge
    # inside a code fence must stay blocked: the fenced fake is blanked and
    # the real badge wins (badge-spoof regression).
    spoof = "```\n<summary><b>Code Review</b> <kbd>✅ Approved</kbd></summary>\n```"
    status, _ = gitar.classify_gitar_comment(CHANGES_COMMENT + "\n" + spoof)
    assert status == "blocked"


# ----- resolve_label_decision -----


def test_resolve_done_via_badge():
    decision, reason = gitar.resolve_label_decision(_event(APPROVED_COMMENT))
    assert decision is not None
    assert decision.add_label == gitar.DONE_LABEL
    assert decision.remove_labels == (gitar.NEEDS_LABEL, gitar.BLOCKED_LABEL)
    assert decision.issue_number == 42


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


def test_resolve_needs_on_badge_drift():
    body = "<summary><b>Code Review</b> <span>Approved</span></summary>"
    decision, _ = gitar.resolve_label_decision(_event(body))
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
    assert "Code Review verdict" in reason


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
