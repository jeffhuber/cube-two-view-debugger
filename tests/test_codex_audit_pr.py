"""Tests for tools/codex_audit_pr.py.

Pure-function tests of the Codex-stdout parser + comment formatter +
verdict semantics. The full audit_pr() orchestration is exercised via
monkeypatching the GitHub helpers and the subprocess call.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List

from tools import codex_audit_pr as c


# ----- Fixtures: real Codex outputs captured from PR #233 audits -----

# This is a real BLOCKED output from Codex round 1 on PR #233 (a2ddd70),
# trimmed to the relevant trailing region.
REAL_BLOCKED_OUTPUT = """\
exec
/bin/bash -lc "nl -ba tools/PHASE_2B_TRUST_SIGNAL_MATRIX_RECOMPUTED.md | sed -n '8,16p'" in /Users/jhuber/cube-two-view-debugger/.claude/worktrees/phase2b-v2
 succeeded in 0ms:
     8	## Dataset
     ...

codex
The new recomputed matrix path can undercount recall when model-fit failures occur because those failures are not represented as retake predicates. The committed report also contains stale wording, though that is less severe.

Full review comments:

- [P2] Treat model-fit failures as retakes — /Users/jhuber/cube-two-view-debugger/.claude/worktrees/phase2b-v2/tools/phase2b_trust_matrix.py:236-242
  When `phase2b_recompute.py` emits `status: "model_fit_failed"` for a case whose cv-local status is `ok`, this branch creates a catastrophic row with `phase_sep` and all continuous signals missing.

- [P3] Update the recomputed report headline — /Users/jhuber/cube-two-view-debugger/.claude/worktrees/phase2b-v2/tools/PHASE_2B_TRUST_SIGNAL_MATRIX_RECOMPUTED.md:76-76
  In the recomputed report this headline still says only `phase_sep + cv-local` rules were evaluated.
"""

REAL_PASS_OUTPUT = """\
exec
/bin/bash -lc "grep -n \\"rembg\\\\|numpy\\\\|Pillow\\" requirements.txt" in /Users/jhuber/cube-two-view-debugger/.claude/worktrees/phase2b-v2
 succeeded in 0ms:
3:# numpy combos silently degrade by ~20 stickers on hard-lighting images
...

codex
The changes add the recomputed signal workflow and associated fixtures/tests without introducing a clear correctness, runtime, or schema issue in the modified code paths. The generated matrices match the committed outputs and the targeted tests pass when run with the repository on PYTHONPATH.
"""


# ----- Parser tests -----


def test_parse_blocked_output_extracts_p_tags():
    parsed = c.parse_codex_output(REAL_BLOCKED_OUTPUT)
    assert parsed.verdict == "BLOCKED"
    assert parsed.p2_count == 1
    assert parsed.p3_count == 1
    assert parsed.p0_count == 0
    assert parsed.p1_count == 0
    assert parsed.blocker_count == 1  # P2 counts; P3 doesn't
    assert len(parsed.findings) == 2
    assert "Treat model-fit failures as retakes" in parsed.findings[0]


def test_parse_pass_output_no_findings():
    parsed = c.parse_codex_output(REAL_PASS_OUTPUT)
    assert parsed.verdict == "PASS"
    assert parsed.blocker_count == 0
    assert parsed.p0_count == 0 and parsed.p1_count == 0
    assert parsed.p2_count == 0 and parsed.p3_count == 0
    assert parsed.findings == []


def test_parse_handles_duplicated_summary():
    """Codex CLI sometimes streams the same final summary twice. The
    parser should de-dupe rather than double-count."""
    duplicated = REAL_PASS_OUTPUT.rstrip() + "\n" + REAL_PASS_OUTPUT.split("\ncodex\n", 1)[1]
    parsed = c.parse_codex_output(duplicated)
    # Duplicated prose shouldn't introduce phantom findings
    p_total = parsed.p0_count + parsed.p1_count + parsed.p2_count + parsed.p3_count
    assert p_total == 0


def test_parse_no_codex_marker_returns_unknown():
    """Codex round 3 of #234 — P2: if the stdout doesn't have a `codex`
    final-verdict marker, parser MUST NOT return PASS (which would
    silently mark unaudited PRs as done). Instead returns UNKNOWN so
    the orchestrator emits the requeue trailer.
    """
    parsed = c.parse_codex_output("some random output without the marker")
    assert parsed.verdict == "UNKNOWN"
    assert "no codex final-verdict block" in parsed.prose.lower()


def test_parse_falls_back_to_single_stderr_marker():
    """Task #85: occasionally the Codex CLI routes the final verdict
    marker to stderr. If stdout has no marker and stderr has exactly one
    column-0 marker, parse stderr as a narrow fallback."""
    parsed = c.parse_codex_output(
        stdout="exec progress without final marker",
        stderr="progress chatter\ncodex\nAll clear from stderr.\n",
    )
    assert parsed.verdict == "PASS"
    assert "All clear from stderr" in parsed.prose


def test_parse_prefers_stdout_marker_over_stderr_noise():
    """Round-6 contamination guard: stderr can contain noisy P-tags; once
    stdout has a real verdict marker, stderr must not influence parsing."""
    parsed = c.parse_codex_output(
        stdout="codex\nAll clear.\n",
        stderr="progress\ncodex\n- [P2] stderr noise — not a finding\n",
    )
    assert parsed.verdict == "PASS"
    assert parsed.p2_count == 0
    assert "stderr noise" not in parsed.prose


def test_parse_multiple_stderr_markers_remains_unknown():
    """If stdout has no marker and stderr has multiple possible markers,
    fail closed rather than risk anchoring on progress chatter."""
    parsed = c.parse_codex_output(
        stdout="exec progress without final marker",
        stderr="codex\nquoted command\ncodex\nmaybe verdict\n",
    )
    assert parsed.verdict == "UNKNOWN"
    assert "multiple possible codex markers" in parsed.prose


def test_audit_pr_unknown_verdict_requeues_with_stale_trailer(monkeypatch):
    """Codex round 3 of #234 — P2 follow-on: when parser returns UNKNOWN,
    audit_pr must format a requeue comment (STALE_TRAILER) and report
    verdict="UNKNOWN", NOT silently post a done comment."""
    pr_meta = _fake_pr(head_sha="aaaa11112222")
    posted: List[Dict[str, Any]] = []
    _install_audit_mocks(
        monkeypatch,
        pr_meta=pr_meta,
        # Output without the `codex` final marker — parser → UNKNOWN
        codex_stdout="some exec output but no final codex line",
        posted_records=posted,
    )
    config = c.AuditConfig(
        github_token="test",
        repo_paths={"jeffhuber/cube-two-view-debugger": Path("/tmp/fake-repo")},
    )
    Path("/tmp/fake-repo").mkdir(exist_ok=True)
    try:
        result = c.audit_pr(config, "jeffhuber/cube-two-view-debugger", 17)
    finally:
        Path("/tmp/fake-repo").rmdir()
    assert result.verdict == "UNKNOWN"
    assert result.trailer == c.STALE_TRAILER
    assert c.STALE_TRAILER in posted[0]["body"]
    assert c.DONE_TRAILER not in posted[0]["body"]
    assert "Could not parse a Codex verdict" in posted[0]["body"]


def test_parse_p0_alone_triggers_blocked():
    output = "codex\nReview comments:\n- [P0] critical bug — file.py:1"
    parsed = c.parse_codex_output(output)
    assert parsed.verdict == "BLOCKED"
    assert parsed.p0_count == 1


def test_parse_only_p3_is_pass():
    """P3 findings don't trip BLOCKED per the policy in
    `CodexVerdict.blocker_count`."""
    output = "codex\nReview comments:\n- [P3] nit — file.py:1"
    parsed = c.parse_codex_output(output)
    assert parsed.verdict == "PASS"
    assert parsed.p3_count == 1
    assert parsed.blocker_count == 0


def test_parse_mixed_severities():
    output = """codex
Findings:
- [P1] one — a.py:1
- [P2] two — b.py:2
- [P2] three — c.py:3
- [P3] four — d.py:4"""
    parsed = c.parse_codex_output(output)
    assert parsed.verdict == "BLOCKED"
    assert parsed.p1_count == 1
    assert parsed.p2_count == 2
    assert parsed.p3_count == 1
    assert parsed.blocker_count == 3  # P1 + 2*P2


def test_parse_takes_last_codex_marker():
    """If the stream contains multiple `codex` lines (e.g., the literal
    string appears in a quoted command), the parser uses the LAST one
    as the final-verdict anchor."""
    output = """early
codex
some prose
exec
codex
final prose
- [P2] real finding — file.py:1"""
    parsed = c.parse_codex_output(output)
    # The final block has the P2 tag as a bullet item
    assert parsed.p2_count == 1
    assert "final prose" in parsed.prose


def test_parse_does_not_count_quoted_p_tags_in_prose():
    """Codex round 4 of #234 — P2: the parser must only count P-tags at
    the START of finding bullets (`- [P2]` or `* [P2]`), not anywhere
    on a line. Otherwise prose mentioning a priority tag (e.g., a P3
    concern explaining 'the protocol mentions [P2] elsewhere') would
    false-trigger BLOCKED."""
    # A PASS-style review where the prose mentions [P2] in passing.
    output = """codex
The change is clean. Note that the protocol doc references [P2] as the
typical severity for correctness fixes, but no [P2] findings here.

- [P3] minor nit — file.py:1
"""
    parsed = c.parse_codex_output(output)
    # The quoted [P2] in prose should NOT count; only the [P3] bullet should
    assert parsed.p2_count == 0
    assert parsed.p3_count == 1
    assert parsed.verdict == "PASS"  # P3-only stays PASS


def test_parse_empty_verdict_block_returns_unknown():
    """Codex round 4 of #234 — P2: when the `codex` marker is present
    but no review text follows (truncated CLI output or format drift),
    the parser must return UNKNOWN, NOT default to PASS."""
    output = "exec\n  ran something\n  done\n\ncodex\n"
    parsed = c.parse_codex_output(output)
    assert parsed.verdict == "UNKNOWN"
    assert "no review prose" in parsed.prose.lower()


def test_parse_whitespace_only_verdict_block_returns_unknown():
    """Same as above but with whitespace after the marker — still UNKNOWN."""
    output = "codex\n   \n\t\n  \n"
    parsed = c.parse_codex_output(output)
    assert parsed.verdict == "UNKNOWN"


def test_parse_does_not_anchor_on_indented_codex_in_prose():
    """Codex round 7 of #234 — P2: the marker regex must match column-0
    `codex` lines only, not indented or fenced occurrences inside the
    final review prose (e.g., when the prose quotes the documented
    transcript format). Anchoring on a quoted occurrence would discard
    the real findings above it.
    """
    # Real marker at column 0, then findings, then a QUOTED `codex` line
    # (indented, mimicking how Codex might quote the format in its prose).
    output = """exec
some pre-codex output

codex
[Real prose summary]

- [P2] real blocker — file.py:10
  This is a real finding. The transcript format looks like:
      codex
      <prose follows>
  ...note the column-0 marker.

- [P3] minor nit — file.py:20
"""
    parsed = c.parse_codex_output(output)
    # The parser must anchor on the column-0 `codex` (the real marker),
    # NOT the indented `      codex` inside the finding details.
    # Both findings should be counted.
    assert parsed.p2_count == 1, (
        f"parser anchored on indented codex: got {parsed.p2_count} P2 "
        f"findings, expected 1 (real findings discarded)"
    )
    assert parsed.p3_count == 1
    assert parsed.verdict == "BLOCKED"


def test_main_returns_exit_code_2_for_unknown_verdict(monkeypatch):
    """Codex round 7 of #234 — P2: UNKNOWN verdicts emit the requeue
    trailer (`needs-codex-audit`) just like STALE. The CLI exit code
    must also be 2 (retry) for both, not 0 (success) for UNKNOWN.
    Otherwise automation using the exit code can't distinguish a
    successful audit from a format-drift requeue.
    """
    # Mock the audit_pr to return an UNKNOWN result
    fake_result = c.AuditResult(
        repo="r", pr_number=1,
        head_sha_start="aaa", head_sha_end="aaa",
        verdict="UNKNOWN", trailer=c.STALE_TRAILER,
        comment_body="...",
        codex_stdout="", codex_stderr="",
    )
    monkeypatch.setattr(c, "audit_pr", lambda *a, **k: fake_result)
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("CODEX_AUDIT_REPO_PATHS", "owner/repo:/tmp/x")
    rc = c.main(["--repo", "owner/repo", "--pr", "1", "--dry-run"])
    assert rc == 2, (
        f"UNKNOWN verdict exited with code {rc}, expected 2 (retry)"
    )


def test_main_returns_exit_code_2_for_stale_verdict(monkeypatch):
    """Symmetric guard for STALE — already-correct behavior, locked in."""
    fake_result = c.AuditResult(
        repo="r", pr_number=1,
        head_sha_start="aaa", head_sha_end="bbb",
        verdict="STALE", trailer=c.STALE_TRAILER,
        comment_body="...",
        codex_stdout="", codex_stderr="",
    )
    monkeypatch.setattr(c, "audit_pr", lambda *a, **k: fake_result)
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("CODEX_AUDIT_REPO_PATHS", "owner/repo:/tmp/x")
    rc = c.main(["--repo", "owner/repo", "--pr", "1", "--dry-run"])
    assert rc == 2


def test_main_returns_exit_code_0_for_pass_verdict(monkeypatch):
    """Symmetric guard for PASS — already-correct behavior, locked in."""
    fake_result = c.AuditResult(
        repo="r", pr_number=1,
        head_sha_start="aaa", head_sha_end="aaa",
        verdict="PASS", trailer=c.DONE_TRAILER,
        comment_body="...",
        codex_stdout="", codex_stderr="",
    )
    monkeypatch.setattr(c, "audit_pr", lambda *a, **k: fake_result)
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("CODEX_AUDIT_REPO_PATHS", "owner/repo:/tmp/x")
    rc = c.main(["--repo", "owner/repo", "--pr", "1", "--dry-run"])
    assert rc == 0


# ----- Comment formatter tests -----


def test_format_comment_pass():
    parsed = c.CodexVerdict(verdict="PASS", prose="All clear.", p3_count=0)
    body = c.format_comment(parsed, "abc1234567890")
    assert "Codex Audit: PASS" in body
    assert c.DONE_TRAILER in body
    assert c.BLOCKED_TRAILER not in body
    assert "abc1234567890" in body
    assert "P0=0, P1=0, P2=0, P3=0" in body


def test_format_comment_blocked_includes_finding_counts():
    parsed = c.CodexVerdict(
        verdict="BLOCKED", prose="Found 2 issues.",
        p2_count=1, p3_count=1,
        findings=["[P2] foo", "[P3] bar"],
    )
    body = c.format_comment(parsed, "def1234567890")
    assert "Codex Audit: BLOCKED" in body
    assert c.BLOCKED_TRAILER in body
    assert c.DONE_TRAILER not in body
    assert "P2=1" in body and "P3=1" in body


def test_format_comment_stale():
    parsed = c.CodexVerdict(verdict="PASS", prose="(stale)")
    body = c.format_comment(parsed, "oldsha1234", is_stale=True, stale_end_sha="newsha9876")
    assert c.STALE_TRAILER in body
    assert "oldsha12" in body and "newsha98" in body
    assert "requeuing" in body


# ----- Repo-path parser tests -----


def test_parse_repo_paths_single():
    result = c._parse_repo_paths("owner/repo:/path/to/checkout")
    assert result == {"owner/repo": Path("/path/to/checkout")}


def test_parse_repo_paths_multiple():
    spec = "jeffhuber/cube-snap:/Users/x/cs,jeffhuber/cube-two-view-debugger:/Users/x/ctvd"
    result = c._parse_repo_paths(spec)
    assert len(result) == 2
    assert result["jeffhuber/cube-snap"] == Path("/Users/x/cs")
    assert result["jeffhuber/cube-two-view-debugger"] == Path("/Users/x/ctvd")


def test_parse_repo_paths_handles_whitespace():
    result = c._parse_repo_paths("  owner/repo : /path  ,  o2/r2 : /p2  ")
    assert result == {"owner/repo": Path("/path"), "o2/r2": Path("/p2")}


def test_parse_repo_paths_rejects_bad_format():
    import pytest
    with pytest.raises(ValueError, match="bad CODEX_AUDIT_REPO_PATHS"):
        c._parse_repo_paths("just-a-name-no-colon")


# ----- Orchestration tests (network/subprocess mocked) -----


def _fake_pr(head_sha: str = "deadbeef00000000", number: int = 17) -> Dict[str, Any]:
    return {
        "number": number,
        "title": "Test PR",
        "head": {"sha": head_sha},
        "base": {"ref": "main", "repo": {"full_name": "jeffhuber/cube-two-view-debugger"}},
    }


def _install_audit_mocks(
    monkeypatch,
    *,
    pr_meta: Dict[str, Any],
    codex_stdout: str,
    codex_stderr: str = "",
    pr_meta_after: Dict[str, Any] = None,
    posted_records: List[Dict[str, Any]] = None,
) -> None:
    """Patch network + subprocess + worktree calls in codex_audit_pr."""
    posted_records = posted_records if posted_records is not None else []
    fetch_counter = {"n": 0}

    def fake_fetch_pr(repo, pr_number, *, token):
        fetch_counter["n"] += 1
        if pr_meta_after is not None and fetch_counter["n"] >= 2:
            return pr_meta_after
        return pr_meta

    monkeypatch.setattr(c, "fetch_pull_request", fake_fetch_pr)

    def fake_post(repo, pr_number, body, *, token):
        posted_records.append({"repo": repo, "pr": pr_number, "body": body})
        return {"html_url": f"https://example/{repo}/pull/{pr_number}#c-fake"}

    monkeypatch.setattr(c, "post_pr_comment", fake_post)
    monkeypatch.setattr(c, "_fetch_pr_head", lambda *a, **k: None)
    monkeypatch.setattr(c, "_fetch_base_ref", lambda *a, **k: None)
    monkeypatch.setattr(c, "_create_temp_worktree",
                        lambda *a, **k: Path("/tmp/fake-wt-test"))
    monkeypatch.setattr(c, "_remove_worktree", lambda *a, **k: None)
    monkeypatch.setattr(
        c, "run_codex_review", lambda *a, **k: (codex_stdout, codex_stderr)
    )


def test_audit_pr_pass_path(monkeypatch):
    pr_meta = _fake_pr(head_sha="aaaa11112222")
    posted: List[Dict[str, Any]] = []
    _install_audit_mocks(
        monkeypatch,
        pr_meta=pr_meta,
        codex_stdout=REAL_PASS_OUTPUT,
        posted_records=posted,
    )
    config = c.AuditConfig(
        github_token="test",
        repo_paths={"jeffhuber/cube-two-view-debugger": Path("/tmp/fake-repo")},
    )
    # Skip the .exists() check by creating /tmp/fake-repo briefly
    Path("/tmp/fake-repo").mkdir(exist_ok=True)
    try:
        result = c.audit_pr(config, "jeffhuber/cube-two-view-debugger", 17)
    finally:
        Path("/tmp/fake-repo").rmdir()
    assert result.verdict == "PASS"
    assert result.trailer == c.DONE_TRAILER
    assert result.head_sha_start == result.head_sha_end
    assert len(posted) == 1
    assert "Codex Audit: PASS" in posted[0]["body"]


def test_audit_pr_blocked_path(monkeypatch):
    pr_meta = _fake_pr(head_sha="bbbb22223333")
    posted: List[Dict[str, Any]] = []
    _install_audit_mocks(
        monkeypatch,
        pr_meta=pr_meta,
        codex_stdout=REAL_BLOCKED_OUTPUT,
        posted_records=posted,
    )
    config = c.AuditConfig(
        github_token="test",
        repo_paths={"jeffhuber/cube-two-view-debugger": Path("/tmp/fake-repo")},
    )
    Path("/tmp/fake-repo").mkdir(exist_ok=True)
    try:
        result = c.audit_pr(config, "jeffhuber/cube-two-view-debugger", 17)
    finally:
        Path("/tmp/fake-repo").rmdir()
    assert result.verdict == "BLOCKED"
    assert result.trailer == c.BLOCKED_TRAILER
    assert result.parsed is not None
    assert result.parsed.p2_count == 1
    assert result.parsed.p3_count == 1
    assert "Codex Audit: BLOCKED" in posted[0]["body"]


def test_audit_pr_parses_stderr_fallback(monkeypatch):
    """End-to-end guard for task #85: audit_pr must pass stderr into the
    parser so a single-marker stderr verdict can become PASS/BLOCKED
    instead of UNKNOWN/requeue."""
    pr_meta = _fake_pr(head_sha="dddd44445555")
    posted: List[Dict[str, Any]] = []
    _install_audit_mocks(
        monkeypatch,
        pr_meta=pr_meta,
        codex_stdout="exec progress without final marker",
        codex_stderr=REAL_PASS_OUTPUT,
        posted_records=posted,
    )
    config = c.AuditConfig(
        github_token="test",
        repo_paths={"jeffhuber/cube-two-view-debugger": Path("/tmp/fake-repo")},
    )
    Path("/tmp/fake-repo").mkdir(exist_ok=True)
    try:
        result = c.audit_pr(config, "jeffhuber/cube-two-view-debugger", 17)
    finally:
        Path("/tmp/fake-repo").rmdir()
    assert result.verdict == "PASS"
    assert result.trailer == c.DONE_TRAILER
    assert result.parsed is not None
    assert result.parsed.prose.startswith("The changes add")
    assert "Codex Audit: PASS" in posted[0]["body"]


def test_audit_pr_stale_head(monkeypatch):
    pr_meta_before = _fake_pr(head_sha="oldhead0000")
    pr_meta_after = _fake_pr(head_sha="newhead0000")
    posted: List[Dict[str, Any]] = []
    _install_audit_mocks(
        monkeypatch,
        pr_meta=pr_meta_before,
        codex_stdout=REAL_PASS_OUTPUT,
        pr_meta_after=pr_meta_after,
        posted_records=posted,
    )
    config = c.AuditConfig(
        github_token="test",
        repo_paths={"jeffhuber/cube-two-view-debugger": Path("/tmp/fake-repo")},
    )
    Path("/tmp/fake-repo").mkdir(exist_ok=True)
    try:
        result = c.audit_pr(config, "jeffhuber/cube-two-view-debugger", 17)
    finally:
        Path("/tmp/fake-repo").rmdir()
    assert result.verdict == "STALE"
    assert result.trailer == c.STALE_TRAILER
    assert result.head_sha_start == "oldhead0000"
    assert result.head_sha_end == "newhead0000"
    assert c.STALE_TRAILER in posted[0]["body"]


def test_audit_pr_dry_run_does_not_post(monkeypatch):
    pr_meta = _fake_pr(head_sha="cccc33334444")
    posted: List[Dict[str, Any]] = []
    _install_audit_mocks(
        monkeypatch,
        pr_meta=pr_meta,
        codex_stdout=REAL_PASS_OUTPUT,
        posted_records=posted,
    )
    config = c.AuditConfig(
        github_token="test",
        repo_paths={"jeffhuber/cube-two-view-debugger": Path("/tmp/fake-repo")},
        dry_run=True,
    )
    Path("/tmp/fake-repo").mkdir(exist_ok=True)
    try:
        result = c.audit_pr(config, "jeffhuber/cube-two-view-debugger", 17)
    finally:
        Path("/tmp/fake-repo").rmdir()
    assert result.verdict == "PASS"
    assert posted == []
    assert result.posted_comment_url is None


def test_audit_pr_raises_without_repo_path(monkeypatch):
    import pytest
    pr_meta = _fake_pr()
    config = c.AuditConfig(
        github_token="test",
        repo_paths={},  # no entry for this repo
    )
    with pytest.raises(ValueError, match="no local repo path configured"):
        c.audit_pr(config, "jeffhuber/cube-two-view-debugger", 17)


# ----- Regression tests for Codex meta-review on #234 -----


def test_run_codex_review_raises_on_nonzero_exit(monkeypatch):
    """Codex meta-review on #234 — P2: a non-zero exit from `codex review`
    (auth failure, base ref missing, CLI crash, etc.) MUST raise
    CalledProcessError rather than returning partial/empty output that
    the parser would treat as a PASS-shaped safe default. Without this
    fix, broken-Codex runs would silently flow into `codex-audit-done`.
    """
    import pytest

    class FakeResult:
        returncode = 2
        stdout = ""
        stderr = "error: not authenticated\n"

    monkeypatch.setattr(c.subprocess, "run", lambda *a, **k: FakeResult())
    monkeypatch.setattr(c.Path, "exists", lambda self: True)  # codex_cli_path exists
    config = c.AuditConfig(
        github_token="x", repo_paths={},
        codex_cli_path="/fake/codex",
    )
    with pytest.raises(c.subprocess.CalledProcessError) as exc_info:
        c.run_codex_review(config, Path("/tmp/fake-wt"))
    assert exc_info.value.returncode == 2
    assert "not authenticated" in (exc_info.value.stderr or "")


def test_run_codex_review_returns_stdout_and_stderr_separately(monkeypatch):
    """Codex round 6 of #234 — P2: stdout and stderr must be returned
    separately so the parser doesn't see stderr noise (which could
    contain `- [P2]` bullets from echoed test output, etc.)."""
    class FakeResult:
        returncode = 0
        stdout = "codex\nPASS prose\n"
        stderr = "progress chatter\n- [P2] this is from stderr — should not count\n"

    monkeypatch.setattr(c.subprocess, "run", lambda *a, **k: FakeResult())
    monkeypatch.setattr(c.Path, "exists", lambda self: True)
    config = c.AuditConfig(github_token="x", repo_paths={}, codex_cli_path="/fake/codex")
    stdout, stderr = c.run_codex_review(config, Path("/tmp/fake-wt"))
    assert "PASS prose" in stdout
    assert "[P2]" not in stdout  # stderr leak would break parsing
    assert "[P2]" in stderr  # stderr captured separately for debugging


def test_fetch_base_ref_calls_git_fetch_with_correct_remote_branch(monkeypatch):
    """Codex meta-review on #234 — P2: the original code only fetched the
    PR head, leaving `origin/main` stale on long-lived local checkouts.
    `_fetch_base_ref` must fetch the remote branch that the base ref
    references.
    """
    calls = []

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return FakeResult()

    monkeypatch.setattr(c.subprocess, "run", fake_run)
    c._fetch_base_ref(Path("/fake/repo"), "origin/main")

    assert len(calls) == 1
    assert calls[0] == ["git", "-C", "/fake/repo", "fetch", "origin", "main"]


def test_fetch_base_ref_handles_non_origin_prefixed_ref(monkeypatch):
    """Fallback for unusual base refs (e.g., a release branch named
    without the `origin/` prefix). Should still attempt a fetch and not
    crash."""
    calls = []

    class FakeResult:
        returncode = 0; stdout = ""; stderr = ""

    monkeypatch.setattr(c.subprocess, "run",
                        lambda *a, **k: calls.append(a[0]) or FakeResult())
    c._fetch_base_ref(Path("/fake/repo"), "release-2026-05")
    assert calls[0] == ["git", "-C", "/fake/repo", "fetch", "origin", "release-2026-05"]


def test_audit_pr_force_push_race_emits_stale_not_crash(monkeypatch):
    """Codex round 6 of #234 — P3: when the PR is force-pushed between
    reading head_sha_start and creating the worktree, the SHA may not
    be locally fetched anymore. The worktree create then fails. The
    audit must catch that, refetch the PR, detect the head moved, and
    emit STALE rather than crash."""
    pr_meta_before = _fake_pr(head_sha="ooooldddhead")
    pr_meta_after = _fake_pr(head_sha="nnnnewwwhead")
    posted: List[Dict[str, Any]] = []
    fetch_count = {"n": 0}

    def fake_fetch_pr(repo, pr_number, *, token):
        fetch_count["n"] += 1
        # First call returns the "before" state; later calls return "after"
        return pr_meta_before if fetch_count["n"] == 1 else pr_meta_after

    monkeypatch.setattr(c, "fetch_pull_request", fake_fetch_pr)
    monkeypatch.setattr(c, "post_pr_comment",
                        lambda r, n, b, *, token: posted.append({"body": b})
                                                  or {"html_url": "u"})
    monkeypatch.setattr(c, "_fetch_pr_head", lambda *a, **k: None)
    monkeypatch.setattr(c, "_fetch_base_ref", lambda *a, **k: None)

    # Worktree create raises CalledProcessError (SHA not locally available)
    def fake_worktree(*a, **k):
        raise subprocess.CalledProcessError(
            128, ["git", "worktree", "add"],
            stderr="fatal: invalid reference: ooooldddhead",
        )
    monkeypatch.setattr(c, "_create_temp_worktree", fake_worktree)
    monkeypatch.setattr(c, "_remove_worktree", lambda *a, **k: None)
    monkeypatch.setattr(c, "run_codex_review",
                        lambda *a, **k: (REAL_PASS_OUTPUT, ""))

    config = c.AuditConfig(
        github_token="test",
        repo_paths={"jeffhuber/cube-two-view-debugger": Path("/tmp/fake-repo")},
    )
    Path("/tmp/fake-repo").mkdir(exist_ok=True)
    try:
        result = c.audit_pr(config, "jeffhuber/cube-two-view-debugger", 17)
    finally:
        Path("/tmp/fake-repo").rmdir()
    # Force-push detected → STALE, not crash
    assert result.verdict == "STALE"
    assert result.trailer == c.STALE_TRAILER
    assert result.head_sha_start == "ooooldddhead"
    assert result.head_sha_end == "nnnnewwwhead"
    assert c.STALE_TRAILER in posted[0]["body"]


def test_audit_pr_fetches_both_pr_head_and_base_ref(monkeypatch):
    """Integration: `audit_pr` should call both `_fetch_pr_head` and
    `_fetch_base_ref` before creating the worktree (Codex meta-review P2)."""
    pr_meta = _fake_pr(head_sha="aaaa00001111")
    posted: List[Dict[str, Any]] = []
    fetch_calls: List[str] = []

    def track_pr_head(local_repo, pr_number):
        fetch_calls.append(f"pr_head({pr_number})")

    def track_base_ref(local_repo, base_ref):
        fetch_calls.append(f"base_ref({base_ref})")

    monkeypatch.setattr(c, "fetch_pull_request", lambda *a, **k: pr_meta)
    monkeypatch.setattr(c, "post_pr_comment",
                        lambda *a, **k: posted.append("p") or {"html_url": "u"})
    monkeypatch.setattr(c, "_fetch_pr_head", track_pr_head)
    monkeypatch.setattr(c, "_fetch_base_ref", track_base_ref)
    monkeypatch.setattr(c, "_create_temp_worktree",
                        lambda *a, **k: Path("/tmp/fake-wt-test"))
    monkeypatch.setattr(c, "_remove_worktree", lambda *a, **k: None)
    monkeypatch.setattr(c, "run_codex_review", lambda *a, **k: (REAL_PASS_OUTPUT, ""))

    config = c.AuditConfig(
        github_token="test",
        repo_paths={"jeffhuber/cube-two-view-debugger": Path("/tmp/fake-repo")},
        base_ref="origin/main",
    )
    Path("/tmp/fake-repo").mkdir(exist_ok=True)
    try:
        c.audit_pr(config, "jeffhuber/cube-two-view-debugger", 17)
    finally:
        Path("/tmp/fake-repo").rmdir()
    # Both fetches happened before the review.
    assert "pr_head(17)" in fetch_calls
    assert "base_ref(origin/main)" in fetch_calls


# ---------------- Task #97: subprocess venv injection ----------------


def test_build_subprocess_env_noop_when_venv_path_is_none(monkeypatch):
    """When `venv_path` is None, the subprocess env mirrors os.environ
    exactly — no PATH mutation, no VIRTUAL_ENV injection. Required so
    repos without a venv (e.g. cube-snap) behave identically to the
    pre-patch behavior."""
    monkeypatch.setenv("PATH", "/sentinel/path")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    env = c._build_subprocess_env(None)
    assert env["PATH"] == "/sentinel/path"
    assert "VIRTUAL_ENV" not in env


def test_build_subprocess_env_noop_when_venv_missing_bin_python(
    tmp_path, monkeypatch,
):
    """If the venv path doesn't have a working `bin/python`, leave PATH
    alone rather than set a broken VIRTUAL_ENV. Defensive: a
    misconfigured venv path shouldn't silently produce more confusing
    failures than the original no-venv state."""
    monkeypatch.setenv("PATH", "/sentinel/path")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    fake_venv = tmp_path / "not-a-venv"
    fake_venv.mkdir()  # exists but has no bin/python
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        env = c._build_subprocess_env(fake_venv)
    assert env["PATH"] == "/sentinel/path"
    assert "VIRTUAL_ENV" not in env


def test_build_subprocess_env_warns_when_venv_path_is_invalid(
    tmp_path, monkeypatch,
):
    """An explicit invalid venv path should not silently fall back to
    ambient Python, since that is the failure mode this feature fixes."""
    monkeypatch.setenv("PATH", "/sentinel/path")
    fake_venv = tmp_path / "not-a-venv"
    fake_venv.mkdir()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = c._build_subprocess_env(fake_venv)
    assert env["PATH"] == "/sentinel/path"
    assert "VIRTUAL_ENV" not in env
    assert len(caught) == 1
    assert issubclass(caught[0].category, RuntimeWarning)
    assert str(fake_venv.resolve()) in str(caught[0].message)
    assert "bin/python" in str(caught[0].message)


def test_build_subprocess_env_prepends_venv_bin_when_python_exists(
    tmp_path, monkeypatch,
):
    """The positive path: a real venv with bin/python gets its bin/
    prepended to PATH and exported as VIRTUAL_ENV."""
    monkeypatch.setenv("PATH", "/sentinel/path")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    fake_venv = tmp_path / "venv"
    (fake_venv / "bin").mkdir(parents=True)
    (fake_venv / "bin" / "python").touch()
    env = c._build_subprocess_env(fake_venv)
    assert env["VIRTUAL_ENV"] == str(fake_venv)
    expected_prefix = f"{fake_venv / 'bin'}{os.pathsep}"
    assert env["PATH"].startswith(expected_prefix), (
        f"venv bin/ must be prepended to PATH; got {env['PATH']!r}"
    )
    # The original PATH must still be reachable after the prepended bin/.
    assert "/sentinel/path" in env["PATH"]


def test_build_subprocess_env_omits_trailing_separator_when_path_absent(
    tmp_path, monkeypatch,
):
    """If PATH is absent, do not create a trailing path separator.

    On POSIX, a trailing separator in PATH implies the current working
    directory, which is not acceptable when auditing untrusted PR
    worktrees.
    """
    monkeypatch.delenv("PATH", raising=False)
    fake_venv = tmp_path / "venv"
    (fake_venv / "bin").mkdir(parents=True)
    (fake_venv / "bin" / "python").touch()
    env = c._build_subprocess_env(fake_venv)
    assert env["VIRTUAL_ENV"] == str(fake_venv)
    assert env["PATH"] == str(fake_venv / "bin")


def test_build_subprocess_env_clears_pythonhome_when_venv_set(
    tmp_path, monkeypatch,
):
    """A stale PYTHONHOME would override the venv's site-packages
    lookup. Defensive: scrub it when we set VIRTUAL_ENV."""
    monkeypatch.setenv("PYTHONHOME", "/some/stale/python")
    fake_venv = tmp_path / "venv"
    (fake_venv / "bin").mkdir(parents=True)
    (fake_venv / "bin" / "python").touch()
    env = c._build_subprocess_env(fake_venv)
    assert "PYTHONHOME" not in env


def test_build_subprocess_env_resolves_relative_venv_path_to_absolute(
    tmp_path, monkeypatch,
):
    """Round-4 P2 on PR #264: a relative venv path injected into PATH
    would be re-resolved from the subprocess's cwd (the temp worktree),
    not the caller's cwd. `_build_subprocess_env` must resolve() to
    absolute before injecting so the subprocess finds the right
    `bin/python` regardless of where it runs."""
    monkeypatch.setenv("PATH", "/sentinel/path")
    fake_venv = tmp_path / "venv"
    (fake_venv / "bin").mkdir(parents=True)
    (fake_venv / "bin" / "python").touch()
    # Pass a RELATIVE path — by changing cwd, simulate the caller
    # passing `--venv-path venv` while pwd is tmp_path.
    monkeypatch.chdir(tmp_path)
    env = c._build_subprocess_env(Path("venv"))
    abs_venv = (tmp_path / "venv").resolve()
    assert env["VIRTUAL_ENV"] == str(abs_venv), (
        f"VIRTUAL_ENV must be absolute; got {env['VIRTUAL_ENV']!r}"
    )
    expected_prefix = f"{abs_venv / 'bin'}{os.pathsep}"
    assert env["PATH"].startswith(expected_prefix), (
        f"PATH must start with the ABSOLUTE bin/, not the relative form; "
        f"got {env['PATH']!r}"
    )


def test_build_subprocess_env_does_not_inject_cwd_when_path_is_empty(
    tmp_path, monkeypatch,
):
    """Round-4 round-2 P2 on PR #264: `Path("")` and `Path(".")` both
    stringify as `"."` and resolve to cwd. If cwd happens to have
    `./bin/python` (e.g. audit machine started from a venv-shaped
    directory), naive resolution would silently inject cwd as
    VIRTUAL_ENV — exactly the silent-cwd-injection bug round-4 was
    meant to fix.

    `_build_subprocess_env` must refuse to inject for these
    empty-ish Path values REGARDLESS of cwd contents. CLI callers
    never reach this function with `Path("")` (they route through
    `disable_venv=True`), but library callers can — and the function
    must be defensive for those callers too. Codex audit on PR #264
    round 4 caught the earlier "by design" pin that allowed this
    surprise injection.
    """
    monkeypatch.setenv("PATH", "/sentinel/path")
    # Build the booby-trap: pwd has bin/python.
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python").touch()
    monkeypatch.chdir(tmp_path)
    for empty_path in (Path(""), Path("."), Path("./")):
        env = c._build_subprocess_env(empty_path)
        assert env["PATH"] == "/sentinel/path", (
            f"_build_subprocess_env({empty_path!r}) must NOT inject cwd "
            f"as PATH prefix; got PATH={env['PATH']!r}"
        )
        assert "VIRTUAL_ENV" not in env, (
            f"_build_subprocess_env({empty_path!r}) must NOT set "
            f"VIRTUAL_ENV; got {env.get('VIRTUAL_ENV')!r}"
        )


def test_discover_venv_finds_venv_when_bin_python_present(tmp_path):
    """`_discover_venv` returns the path when `.venv/bin/python` exists."""
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").touch()
    discovered = c._discover_venv(tmp_path)
    assert discovered == venv


def test_discover_venv_returns_none_when_no_venv(tmp_path):
    """`_discover_venv` is a no-op when the local repo has no
    `.venv/bin/python` (matches cube-snap's actual state)."""
    discovered = c._discover_venv(tmp_path)
    assert discovered is None


def test_discover_venv_returns_none_when_venv_present_but_no_python(tmp_path):
    """A bare `.venv/` directory without `bin/python` is treated as
    'not a venv' — better to return None than to point at a broken
    one and create a misleading VIRTUAL_ENV later."""
    venv = tmp_path / ".venv"
    venv.mkdir()
    discovered = c._discover_venv(tmp_path)
    assert discovered is None


def test_audit_pr_auto_discovers_repo_venv_when_unset(monkeypatch, tmp_path):
    """End-to-end: when `config.venv_path` is None and the local repo
    contains a `.venv/bin/python`, audit_pr discovers it and passes
    the right config into run_codex_review.

    Captures the config passed to run_codex_review via a stub and
    asserts venv_path was promoted to the discovered path. Without
    this end-to-end check, the auto-discovery logic could regress to
    a no-op and we wouldn't notice until live audits started using
    anaconda Python again (the Task #97 failure mode).
    """
    pr_meta = _fake_pr(head_sha="aaaa11112222")
    captured: Dict[str, Any] = {}
    repo_root = tmp_path / "fake-repo"
    venv = repo_root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").touch()

    def fake_run_codex_review(cfg, worktree_path):
        captured["venv_path"] = cfg.venv_path
        return REAL_PASS_OUTPUT, ""

    _install_audit_mocks(
        monkeypatch, pr_meta=pr_meta, codex_stdout=REAL_PASS_OUTPUT,
    )
    monkeypatch.setattr(c, "run_codex_review", fake_run_codex_review)
    config = c.AuditConfig(
        github_token="test",
        repo_paths={"jeffhuber/cube-two-view-debugger": repo_root},
    )
    c.audit_pr(config, "jeffhuber/cube-two-view-debugger", 17)
    assert captured["venv_path"] == venv, (
        f"audit_pr should have auto-discovered {venv}; "
        f"got {captured.get('venv_path')}"
    )


def test_audit_pr_respects_explicit_venv_path_override(
    monkeypatch, tmp_path,
):
    """If `config.venv_path` is set explicitly, auto-discovery must NOT
    overwrite it — caller's choice wins."""
    pr_meta = _fake_pr(head_sha="aaaa11112222")
    captured: Dict[str, Any] = {}
    repo_root = tmp_path / "fake-repo"
    # Repo has its own .venv (would normally be auto-discovered).
    real_venv = repo_root / ".venv"
    (real_venv / "bin").mkdir(parents=True)
    (real_venv / "bin" / "python").touch()
    # But the caller pinned a different venv explicitly.
    explicit_venv = tmp_path / "other-venv"
    (explicit_venv / "bin").mkdir(parents=True)
    (explicit_venv / "bin" / "python").touch()

    def fake_run_codex_review(cfg, worktree_path):
        captured["venv_path"] = cfg.venv_path
        return REAL_PASS_OUTPUT, ""

    _install_audit_mocks(
        monkeypatch, pr_meta=pr_meta, codex_stdout=REAL_PASS_OUTPUT,
    )
    monkeypatch.setattr(c, "run_codex_review", fake_run_codex_review)
    config = c.AuditConfig(
        github_token="test",
        repo_paths={"jeffhuber/cube-two-view-debugger": repo_root},
        venv_path=explicit_venv,
    )
    c.audit_pr(config, "jeffhuber/cube-two-view-debugger", 17)
    assert captured["venv_path"] == explicit_venv


def test_audit_pr_no_op_when_repo_has_no_venv(monkeypatch, tmp_path):
    """A repo without a `.venv/` (e.g. cube-snap) gets `venv_path=None`
    — same as before this patch. No change in behavior for venv-less
    repos."""
    pr_meta = _fake_pr(head_sha="aaaa11112222")
    captured: Dict[str, Any] = {}
    repo_root = tmp_path / "no-venv-repo"
    repo_root.mkdir()  # exists but has no .venv

    def fake_run_codex_review(cfg, worktree_path):
        captured["venv_path"] = cfg.venv_path
        return REAL_PASS_OUTPUT, ""

    _install_audit_mocks(
        monkeypatch, pr_meta=pr_meta, codex_stdout=REAL_PASS_OUTPUT,
    )
    monkeypatch.setattr(c, "run_codex_review", fake_run_codex_review)
    config = c.AuditConfig(
        github_token="test",
        repo_paths={"jeffhuber/cube-snap": repo_root},
    )
    c.audit_pr(config, "jeffhuber/cube-snap", 17)
    assert captured["venv_path"] is None


def test_audit_pr_disable_venv_blocks_auto_discovery(
    monkeypatch, tmp_path,
):
    """Round-4 P2 on PR #264: when `config.disable_venv=True` (set by
    CLI's `--venv-path ""`), audit_pr must NOT auto-discover even if
    `<local_repo>/.venv/` exists."""
    pr_meta = _fake_pr(head_sha="aaaa11112222")
    captured: Dict[str, Any] = {}
    repo_root = tmp_path / "repo-with-venv"
    venv = repo_root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").touch()

    def fake_run_codex_review(cfg, worktree_path):
        captured["venv_path"] = cfg.venv_path
        return REAL_PASS_OUTPUT, ""

    _install_audit_mocks(
        monkeypatch, pr_meta=pr_meta, codex_stdout=REAL_PASS_OUTPUT,
    )
    monkeypatch.setattr(c, "run_codex_review", fake_run_codex_review)
    config = c.AuditConfig(
        github_token="test",
        repo_paths={"jeffhuber/cube-two-view-debugger": repo_root},
        disable_venv=True,
    )
    c.audit_pr(config, "jeffhuber/cube-two-view-debugger", 17)
    assert captured["venv_path"] is None, (
        f"disable_venv=True must block auto-discovery; got "
        f"{captured.get('venv_path')}"
    )


def test_audit_pr_auto_discovery_returns_absolute_venv_path(
    monkeypatch, tmp_path,
):
    """Round-4 P2 on PR #264: `_discover_venv` must return an absolute
    path. Otherwise the subprocess (cwd=worktree_path) would resolve
    a relative `.venv/bin` against the worktree dir, not the audit
    machine's cwd."""
    pr_meta = _fake_pr(head_sha="aaaa11112222")
    captured: Dict[str, Any] = {}
    # Use a RELATIVE local_repo path by changing into tmp_path first.
    repo_dir = tmp_path / "fake-repo"
    (repo_dir / ".venv" / "bin").mkdir(parents=True)
    (repo_dir / ".venv" / "bin" / "python").touch()
    monkeypatch.chdir(tmp_path)

    def fake_run_codex_review(cfg, worktree_path):
        captured["venv_path"] = cfg.venv_path
        return REAL_PASS_OUTPUT, ""

    _install_audit_mocks(
        monkeypatch, pr_meta=pr_meta, codex_stdout=REAL_PASS_OUTPUT,
    )
    monkeypatch.setattr(c, "run_codex_review", fake_run_codex_review)
    config = c.AuditConfig(
        github_token="test",
        repo_paths={"jeffhuber/cube-two-view-debugger": Path("fake-repo")},
    )
    c.audit_pr(config, "jeffhuber/cube-two-view-debugger", 17)
    discovered = captured["venv_path"]
    assert discovered is not None
    assert discovered.is_absolute(), (
        f"discovered venv path must be absolute (PR #264 round-4 P2); "
        f"got {discovered!r}"
    )


# ----- Wrapper regression: silent-success-on-Python-failure bug -----


def test_wrapper_propagates_python_failure_exit_code_to_audit_log(tmp_path, monkeypatch):
    """Regression: `tools/run_codex_audit_pr.sh` previously logged
    `exitCode=0, status=completed` even when `codex_audit_pr.py` exited 1
    (e.g. missing --repo-paths). Root cause: bash trap captured `local`'s
    own exit code (always 0) instead of the failing Python script's.

    Caught on cube-snap#184 where three audit runs all logged as completed
    but never posted a GitHub comment because Python failed each time.

    Fix: capture `$?` in the SAME statement as `local rc=$?`. This test
    invokes the real wrapper with conditions guaranteed to make Python
    exit 1 (no --repo-paths, no env var), and asserts the audit log
    correctly records the failure.
    """
    import json
    repo_root = Path(__file__).resolve().parents[1]
    wrapper = repo_root / "tools" / "run_codex_audit_pr.sh"
    assert wrapper.exists(), f"wrapper not found at {wrapper}"

    # Redirect the audit log to a tmp dir so we don't touch the user's
    # real ~/.cache/cube-agent-audits/.
    monkeypatch.setenv("AUDIT_HANDOFF_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("CODEX_AUDIT_REPO_PATHS", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    # Force-fail Python: no --repo-paths arg + no env var + no token.
    # Use the repo's own .venv so the wrapper's controlled-Python check
    # passes (we want to test the wrapper's exit-code propagation, not
    # its venv-discovery logic).
    result = subprocess.run(
        [str(wrapper), "--repo", "jeffhuber/cube-snap", "--pr", "999999"],
        capture_output=True, text=True, env={
            **{k: v for k, v in os.environ.items()
               if k not in ("CODEX_AUDIT_REPO_PATHS", "GITHUB_TOKEN")},
            "AUDIT_HANDOFF_LOG_DIR": str(tmp_path),
        },
    )

    # The wrapper itself must exit non-zero — that's the signal callers
    # use to detect failure. Before the fix this was 0.
    assert result.returncode != 0, (
        f"wrapper exited 0 despite Python failure; stderr={result.stderr!r}"
    )

    # The audit log must record exitCode != 0 and status='failed'.
    # Before the fix this recorded exitCode=0, status=completed.
    events_path = tmp_path / "events.jsonl"
    assert events_path.exists(), f"no events.jsonl created at {events_path}"
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
    finished = [e for e in events if e.get("event") == "finished"]
    assert len(finished) >= 1, f"no finished event in log; events={events!r}"
    last_finished = finished[-1]
    assert last_finished.get("exitCode") != 0, (
        f"audit log recorded exitCode=0 despite Python failure — the "
        f"`local rc status; rc=$?` bash trap bug. Event: {last_finished!r}"
    )
    assert last_finished.get("status") == "failed", (
        f"audit log recorded status={last_finished.get('status')!r} "
        f"despite Python failure; expected 'failed'. Event: {last_finished!r}"
    )
