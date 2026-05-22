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


def test_parse_no_codex_marker_returns_safe_default():
    """If the stdout doesn't have a `codex` final-verdict marker (e.g.,
    the subprocess crashed before producing one), parser returns a PASS-
    shaped verdict with a flag in the prose so caller can detect."""
    parsed = c.parse_codex_output("some random output without the marker")
    assert parsed.verdict == "PASS"
    assert "no codex final-verdict block" in parsed.prose.lower()


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
final prose [P2] real finding"""
    parsed = c.parse_codex_output(output)
    # The final block has the P2 tag
    assert parsed.p2_count == 1
    assert "final prose" in parsed.prose


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
    monkeypatch.setattr(c, "_create_temp_worktree",
                        lambda *a, **k: Path("/tmp/fake-wt-test"))
    monkeypatch.setattr(c, "_remove_worktree", lambda *a, **k: None)
    monkeypatch.setattr(c, "run_codex_review", lambda *a, **k: codex_stdout)


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
