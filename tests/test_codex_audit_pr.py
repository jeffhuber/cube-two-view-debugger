"""Tests for tools/codex_audit_pr.py.

Pure-function tests of the Codex-stdout parser + comment formatter +
verdict semantics. The full audit_pr() orchestration is exercised via
monkeypatching the GitHub helpers and the subprocess call.
"""

from __future__ import annotations

import subprocess
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
