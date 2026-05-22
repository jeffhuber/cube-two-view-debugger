"""Tests for tools/qwen_audit_pr.py.

The audit_pr() orchestration is exercised via monkeypatching the GitHub
helpers (fetch_pull_request, fetch_pr_files, fetch_file_content,
post_pr_comment) and the Qwen call (call_qwen). This keeps the tests
network-free and deterministic.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from tools import qwen_audit_pr


# ----- Pure helper functions -----


def test_parse_per_file_response_plain_json():
    blockers, concerns = qwen_audit_pr.parse_per_file_response(
        '{"blockers": ["missing test"], "concerns": ["awkward name"]}'
    )
    assert blockers == ["missing test"]
    assert concerns == ["awkward name"]


def test_parse_per_file_response_with_markdown_fences():
    text = '```json\n{"blockers": [], "concerns": ["nit"]}\n```'
    blockers, concerns = qwen_audit_pr.parse_per_file_response(text)
    assert blockers == []
    assert concerns == ["nit"]


def test_parse_per_file_response_with_leading_prose():
    text = (
        "Here is my review:\n\n"
        '{"blockers": ["calls undefined function bar"], "concerns": []}'
    )
    blockers, concerns = qwen_audit_pr.parse_per_file_response(text)
    assert blockers == ["calls undefined function bar"]
    assert concerns == []


def test_parse_per_file_response_malformed_returns_empty():
    blockers, concerns = qwen_audit_pr.parse_per_file_response("not json at all")
    assert blockers == []
    assert concerns == []


def test_parse_per_file_response_partial_json_returns_empty():
    blockers, concerns = qwen_audit_pr.parse_per_file_response('{"blockers": ["a"')
    assert blockers == []
    assert concerns == []


def test_parse_per_file_response_drops_empty_strings():
    blockers, concerns = qwen_audit_pr.parse_per_file_response(
        '{"blockers": ["", "real"], "concerns": [null, "also"]}'
    )
    assert blockers == ["real"]
    assert concerns == ["also"]


def test_ensure_trailer_appends_done():
    text = "Qwen Audit: PASS\n\nLooks fine."
    result = qwen_audit_pr.ensure_trailer(text, "PASS")
    assert result.rstrip().endswith(qwen_audit_pr.DONE_TRAILER)
    assert qwen_audit_pr.BLOCKED_TRAILER not in result


def test_ensure_trailer_appends_blocked():
    text = "Qwen Audit: BLOCKED\n\nBug at line 14."
    result = qwen_audit_pr.ensure_trailer(text, "BLOCKED")
    assert result.rstrip().endswith(qwen_audit_pr.BLOCKED_TRAILER)
    assert qwen_audit_pr.DONE_TRAILER not in result


def test_ensure_trailer_replaces_wrong_trailer_from_model():
    # Model emitted the wrong trailer; we should override with canonical
    # based on the verdict.
    text = (
        "Qwen Audit: BLOCKED\n\nbug found\n"
        "<!-- QWEN_AUDIT_STATE: qwen-audit-done -->"
    )
    result = qwen_audit_pr.ensure_trailer(text, "BLOCKED")
    assert result.count("QWEN_AUDIT_STATE") == 1
    assert qwen_audit_pr.BLOCKED_TRAILER in result
    assert qwen_audit_pr.DONE_TRAILER not in result


def test_ensure_trailer_strips_stale_trailer_when_verdict_known():
    # If somehow the synthesis emitted the stale trailer but we have a real
    # verdict, ensure_trailer normalizes to the verdict's trailer.
    text = "Qwen Audit: PASS\n\nfine\n<!-- QWEN_AUDIT_STATE: needs-qwen-audit -->"
    result = qwen_audit_pr.ensure_trailer(text, "PASS")
    assert "needs-qwen-audit" not in result
    assert qwen_audit_pr.DONE_TRAILER in result


def test_is_likely_binary_filename_for_known_extensions():
    assert qwen_audit_pr._is_likely_binary_filename("photo.JPG")
    assert qwen_audit_pr._is_likely_binary_filename("model.onnx")
    assert qwen_audit_pr._is_likely_binary_filename("font.woff2")
    assert not qwen_audit_pr._is_likely_binary_filename("tools/foo.py")
    assert not qwen_audit_pr._is_likely_binary_filename("README.md")


def test_safe_decode_text():
    text, reason = qwen_audit_pr._safe_decode(b"hello\nworld")
    assert text == "hello\nworld"
    assert reason == "ok"


def test_safe_decode_binary_with_nul_byte():
    text, reason = qwen_audit_pr._safe_decode(b"PNG\x00\x01\x02data")
    assert text is None
    assert reason == "binary"


def test_safe_decode_invalid_utf8():
    text, reason = qwen_audit_pr._safe_decode(b"\xff\xfe\xfd")
    assert text is None
    assert reason == "binary"


def test_build_per_file_user_prompt_truncates_oversized_content():
    big_content = "a" * (qwen_audit_pr.MAX_FILE_BYTES + 100)
    file_meta = {"filename": "tools/foo.py", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b"}
    pr_meta = {"title": "T", "body": "B"}
    prompt = qwen_audit_pr.build_per_file_user_prompt(file_meta, big_content, pr_meta)
    assert "truncated" in prompt
    # The truncated content marker should reference the original size.
    assert f"{len(big_content)} bytes" in prompt


def test_build_per_file_user_prompt_handles_missing_content():
    file_meta = {"filename": "deleted.py", "status": "removed", "patch": "@@ -1 +0,0 @@\n-old"}
    pr_meta = {"title": "T", "body": ""}
    prompt = qwen_audit_pr.build_per_file_user_prompt(file_meta, None, pr_meta)
    assert "file content not available" in prompt
    assert "@@ -1 +0,0 @@" in prompt


def test_format_top_matter_counts_findings():
    findings = [
        qwen_audit_pr.FileFinding(path="a.py", status="modified", blockers=["x"]),
        qwen_audit_pr.FileFinding(path="b.py", status="modified", concerns=["nit"]),
        qwen_audit_pr.FileFinding(path="c.py", status="modified"),
        qwen_audit_pr.FileFinding(path="d.png", status="skipped-binary"),
    ]
    out = qwen_audit_pr._format_top_matter("jeffhuber/cube-snap", 7, "abc12345", findings)
    assert "Files reviewed: 4" in out
    assert "blocker findings: 1" in out
    assert "concern-only: 1" in out
    # Clean = total - blocker - concern-only; binary counts as clean here.
    assert "clean: 2" in out
    assert "`abc12345`" in out


def test_format_stale_comment_includes_both_shas():
    out = qwen_audit_pr._format_stale_comment(
        "jeffhuber/cube-snap", 7, "abcd1234567890", "wxyz9876543210"
    )
    assert "abcd1234" in out
    assert "wxyz9876" in out
    assert qwen_audit_pr.STALE_TRAILER in out


# ----- audit_pr orchestration (network mocked) -----


def _fake_pr(head_sha: str = "deadbeef00", number: int = 17) -> Dict[str, Any]:
    return {
        "number": number,
        "title": "Test PR",
        "body": "This is the PR body. Claims to add a new tool.",
        "user": {"login": "claude"},
        "head": {"sha": head_sha},
        "base": {"ref": "main", "repo": {"full_name": "jeffhuber/cube-snap"}},
        "changed_files": 1,
        "additions": 3,
        "deletions": 0,
    }


def _install_mocks(
    monkeypatch,
    *,
    pr_meta: Dict[str, Any],
    files: List[Dict[str, Any]],
    file_contents: Dict[str, bytes],
    per_file_response: str,
    synthesis_response: str,
    pr_meta_after: Dict[str, Any] = None,
    posted: List[Dict[str, Any]] = None,
) -> None:
    """Patch the network-touching helpers in qwen_audit_pr."""
    posted = posted if posted is not None else []
    fetch_calls = {"count": 0}

    def fake_fetch_pr(repo, pr_number, *, token):
        fetch_calls["count"] += 1
        if pr_meta_after is not None and fetch_calls["count"] >= 2:
            return pr_meta_after
        return pr_meta

    monkeypatch.setattr(qwen_audit_pr, "fetch_pull_request", fake_fetch_pr)
    monkeypatch.setattr(qwen_audit_pr, "fetch_pr_files", lambda r, p, *, token: files)
    monkeypatch.setattr(
        qwen_audit_pr,
        "fetch_file_content",
        lambda repo, path, ref, *, token: file_contents.get(path),
    )

    def fake_post(repo, pr_number, body, *, token):
        record = {"repo": repo, "pr": pr_number, "body": body}
        posted.append(record)
        return {"html_url": f"https://github.com/{repo}/pull/{pr_number}#comment-1"}

    monkeypatch.setattr(qwen_audit_pr, "post_pr_comment", fake_post)

    qwen_calls: List[str] = []

    def fake_qwen(config, system, user, *, max_tokens=2048):
        qwen_calls.append(system[:40])
        # System prompts begin with distinct sentences; use that to route.
        if system.startswith("You are an automated code reviewer. Your job is to find BLOCKERS"):
            return per_file_response
        return synthesis_response

    monkeypatch.setattr(qwen_audit_pr, "call_qwen", fake_qwen)
    qwen_audit_pr._test_qwen_calls = qwen_calls  # exposed for inspection


def test_audit_pr_pass_path(monkeypatch):
    pr_meta = _fake_pr(head_sha="aaaa1111")
    files = [{"filename": "tools/hello.py", "status": "added", "patch": "+def hi(): pass"}]
    _install_mocks(
        monkeypatch,
        pr_meta=pr_meta,
        files=files,
        file_contents={"tools/hello.py": b"def hi():\n    pass\n"},
        per_file_response='{"blockers": [], "concerns": []}',
        synthesis_response="Qwen Audit: PASS\n\nLooks clean.\n",
    )
    config = qwen_audit_pr.AuditConfig(github_token="test")
    posted_records: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        qwen_audit_pr,
        "post_pr_comment",
        lambda r, n, b, *, token: posted_records.append({"body": b}) or {"html_url": "u"},
    )

    result = qwen_audit_pr.audit_pr(config, "jeffhuber/cube-snap", 17)

    assert result.verdict == "PASS"
    assert result.trailer == qwen_audit_pr.DONE_TRAILER
    assert result.head_sha_start == "aaaa1111"
    assert result.head_sha_end == "aaaa1111"
    assert len(result.file_findings) == 1
    assert result.file_findings[0].blockers == []
    assert len(posted_records) == 1
    assert qwen_audit_pr.DONE_TRAILER in posted_records[0]["body"]


def test_audit_pr_blocked_when_per_file_finds_blocker(monkeypatch):
    pr_meta = _fake_pr(head_sha="bbbb2222")
    files = [{"filename": "tools/api.py", "status": "modified", "patch": "+import os.system"}]
    # Per-file pass returns a blocker; synthesis (even if model says PASS)
    # must NOT override that blocker into a PASS verdict.
    _install_mocks(
        monkeypatch,
        pr_meta=pr_meta,
        files=files,
        file_contents={"tools/api.py": b"def f(): pass"},
        per_file_response='{"blockers": ["calls os.system on user input — security"], "concerns": []}',
        synthesis_response="Qwen Audit: PASS\n\nNothing wrong here.\n",
    )
    posted_records: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        qwen_audit_pr,
        "post_pr_comment",
        lambda r, n, b, *, token: posted_records.append({"body": b}) or {"html_url": "u"},
    )
    config = qwen_audit_pr.AuditConfig(github_token="test")

    result = qwen_audit_pr.audit_pr(config, "jeffhuber/cube-snap", 18)

    # Per-file pass found a blocker — verdict must be BLOCKED regardless of
    # what synthesis self-reported. Codex's "don't approve by vibes" rule.
    assert result.verdict == "BLOCKED"
    assert qwen_audit_pr.BLOCKED_TRAILER in posted_records[0]["body"]
    assert qwen_audit_pr.DONE_TRAILER not in posted_records[0]["body"]


def test_audit_pr_blocked_when_synthesis_says_blocked(monkeypatch):
    # Per-file pass found nothing, but synthesis pass found a cross-cutting
    # issue (e.g., doc-index inconsistency) and reported BLOCKED.
    pr_meta = _fake_pr(head_sha="cccc3333")
    files = [{"filename": "tools/foo.py", "status": "modified", "patch": "+pass"}]
    _install_mocks(
        monkeypatch,
        pr_meta=pr_meta,
        files=files,
        file_contents={"tools/foo.py": b"pass\n"},
        per_file_response='{"blockers": [], "concerns": []}',
        synthesis_response=(
            "Qwen Audit: BLOCKED\n\nNew tool not registered in README.\n"
        ),
    )
    posted_records: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        qwen_audit_pr,
        "post_pr_comment",
        lambda r, n, b, *, token: posted_records.append({"body": b}) or {"html_url": "u"},
    )
    config = qwen_audit_pr.AuditConfig(github_token="test")

    result = qwen_audit_pr.audit_pr(config, "jeffhuber/cube-snap", 19)
    assert result.verdict == "BLOCKED"
    assert qwen_audit_pr.BLOCKED_TRAILER in posted_records[0]["body"]


def test_audit_pr_stale_head_during_review(monkeypatch):
    pr_meta_before = _fake_pr(head_sha="oldhead00")
    pr_meta_after = _fake_pr(head_sha="newhead00")
    files = [{"filename": "tools/x.py", "status": "modified", "patch": "+x"}]
    _install_mocks(
        monkeypatch,
        pr_meta=pr_meta_before,
        files=files,
        file_contents={"tools/x.py": b"x = 1\n"},
        per_file_response='{"blockers": [], "concerns": []}',
        synthesis_response="(should not be reached)",
        pr_meta_after=pr_meta_after,
    )
    posted_records: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        qwen_audit_pr,
        "post_pr_comment",
        lambda r, n, b, *, token: posted_records.append({"body": b}) or {"html_url": "u"},
    )
    config = qwen_audit_pr.AuditConfig(github_token="test")

    result = qwen_audit_pr.audit_pr(config, "jeffhuber/cube-snap", 20)
    assert result.verdict == "STALE"
    assert result.trailer == qwen_audit_pr.STALE_TRAILER
    assert result.head_sha_start == "oldhead00"
    assert result.head_sha_end == "newhead00"
    assert qwen_audit_pr.STALE_TRAILER in posted_records[0]["body"]
    # Synthesis pass should NOT have been called for stale review.
    assert all("Your job is two things" not in c for c in qwen_audit_pr._test_qwen_calls)


def test_audit_pr_dry_run_does_not_post(monkeypatch):
    pr_meta = _fake_pr(head_sha="dddd4444")
    files = [{"filename": "x.py", "status": "modified", "patch": "+x"}]
    _install_mocks(
        monkeypatch,
        pr_meta=pr_meta,
        files=files,
        file_contents={"x.py": b"x=1"},
        per_file_response='{"blockers": [], "concerns": []}',
        synthesis_response="Qwen Audit: PASS\n\nFine.\n",
    )
    posted_records: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        qwen_audit_pr,
        "post_pr_comment",
        lambda r, n, b, *, token: posted_records.append({"body": b}) or {"html_url": "u"},
    )
    config = qwen_audit_pr.AuditConfig(github_token="test", dry_run=True)
    result = qwen_audit_pr.audit_pr(config, "jeffhuber/cube-snap", 21)
    assert result.verdict == "PASS"
    assert posted_records == []  # dry-run skips posting
    assert result.posted_comment_url is None


def test_audit_pr_skips_binary_files_by_extension(monkeypatch):
    pr_meta = _fake_pr(head_sha="eeee5555")
    files = [
        {"filename": "fixtures/cube.png", "status": "added", "patch": ""},
        {"filename": "tools/foo.py", "status": "modified", "patch": "+x"},
    ]
    _install_mocks(
        monkeypatch,
        pr_meta=pr_meta,
        files=files,
        file_contents={"tools/foo.py": b"x = 1"},
        per_file_response='{"blockers": [], "concerns": []}',
        synthesis_response="Qwen Audit: PASS\n\nFine.\n",
    )
    monkeypatch.setattr(qwen_audit_pr, "post_pr_comment", lambda *a, **k: {"html_url": "u"})
    config = qwen_audit_pr.AuditConfig(github_token="test")
    result = qwen_audit_pr.audit_pr(config, "jeffhuber/cube-snap", 22)
    assert len(result.file_findings) == 2
    binary_finding = next(f for f in result.file_findings if f.path.endswith(".png"))
    assert binary_finding.status == "skipped-binary"
    assert binary_finding.blockers == []


def test_audit_pr_per_file_qwen_error_becomes_blocker(monkeypatch):
    pr_meta = _fake_pr(head_sha="ffff6666")
    files = [{"filename": "x.py", "status": "modified", "patch": "+x"}]
    monkeypatch.setattr(qwen_audit_pr, "fetch_pull_request", lambda *a, **k: pr_meta)
    monkeypatch.setattr(qwen_audit_pr, "fetch_pr_files", lambda *a, **k: files)
    monkeypatch.setattr(qwen_audit_pr, "fetch_file_content", lambda *a, **k: b"x = 1")
    monkeypatch.setattr(
        qwen_audit_pr,
        "post_pr_comment",
        lambda r, n, b, *, token: {"html_url": "u"},
    )

    call_count = {"n": 0}

    def boom(config, system, user, *, max_tokens=2048):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Per-file pass fails — should be caught and converted to a
            # synthesized "could not review" blocker, not propagated.
            import urllib.error
            raise urllib.error.URLError("connection refused")
        return "Qwen Audit: BLOCKED\n\nCould not review.\n"

    monkeypatch.setattr(qwen_audit_pr, "call_qwen", boom)
    config = qwen_audit_pr.AuditConfig(github_token="test")
    result = qwen_audit_pr.audit_pr(config, "jeffhuber/cube-snap", 23)
    assert result.verdict == "BLOCKED"
    assert any("Per-file Qwen call failed" in b for b in result.file_findings[0].blockers)
