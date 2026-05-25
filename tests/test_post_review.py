"""Verdict-gated label-removal behavior in tools/post_review.sh.

Codex P2 on cube-snap#173 / ctvd#311: the prior version of
post_review.sh removed the routing label unconditionally even for
blocker reviews, which would have caused the queue-visibility
failure the whole review-log-events PR is meant to prevent. This
test pins the fixed behavior: --verdict pass calls
`gh pr edit ... --remove-label`; any other verdict skips that call
and keeps the label so the queue sweep still sees the PR as needing
follow-up.

Mocks `gh` and `safe_gh_comment.py` via PATH-shadowing — we don't
want the test to touch real GitHub. Each fake records its argv to a
log file in the test's tmp_path; the test then asserts on that log.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
POST_REVIEW = REPO_ROOT / "tools" / "post_review.sh"


def _install_fakes(tmp_path: Path) -> tuple[Path, Path]:
    """Create fake `gh` and `safe_gh_comment.py` in tmp_path/bin.

    Each fake just appends its argv (newline-separated) to a per-fake
    log file. The test reads those logs to assert what the script
    actually called.

    Returns (fake_bin_dir, gh_log).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh_log = tmp_path / "gh.log"
    sgc_log = tmp_path / "safe_gh_comment.log"

    fake_gh = bin_dir / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\0" "$@" >> "{gh_log}"\n'
        'printf "\\n" >> "' + str(gh_log) + '"\n'
        "exit 0\n"
    )
    fake_gh.chmod(0o755)

    # Shadow safe_gh_comment.py too: we shadow at the
    # tools/safe_gh_comment.py path that post_review.sh resolves
    # relative to ${script_dir}. The script invokes it via:
    #   "${python_bin}" "${script_dir}/safe_gh_comment.py" ...
    # so we can't easily shadow via PATH. Instead the test fixture
    # uses a real python_bin (the repo's .venv) and the real
    # safe_gh_comment.py path is overwritten via a workspace copy
    # of tools/post_review.sh (see _stage_post_review).
    return bin_dir, gh_log


def _stage_post_review(tmp_path: Path, sgc_log: Path) -> Path:
    """Copy post_review.sh into tmp_path and rewrite the
    safe_gh_comment.py invocation to a no-op recorder so the test
    doesn't actually post to GitHub.
    """
    src = POST_REVIEW.read_text(encoding="utf-8")
    # Replace the safe_gh_comment.py invocation with a bash here-doc
    # that just records its args. This is the simplest way to mock
    # without faking the full Python interpreter resolution.
    staged_src = src.replace(
        '"${python_bin}" "${script_dir}/safe_gh_comment.py" \\\n'
        '  --repo "${REPO}" --pr "${PR}" --body-file "${BODY_FILE}"',
        f'printf "POST %s %s %s\\n" "${{REPO}}" "${{PR}}" "${{BODY_FILE}}" >> "{sgc_log}"',
    )
    # Also short-circuit the record_event call so we don't touch
    # the real audit-handoff-log. Replace with a stub that just
    # records the lane + verdict to a file.
    staged_src = staged_src.replace(
        '"${python_bin}" "${script_dir}/audit_handoff_log.py" record \\\n'
        '  --lane "${LANE}" \\\n'
        '  --repo "${REPO}" \\\n'
        '  --pr "${PR}" \\\n'
        '  --head "${HEAD}" \\\n'
        '  --event finished \\\n'
        '  --verdict "${VERDICT}" \\\n'
        '  --actor "${ACTOR}"',
        f'printf "LOG %s %s\\n" "${{LANE}}" "${{VERDICT}}" >> "{sgc_log}"',
    )
    staged = tmp_path / "post_review.sh"
    staged.write_text(staged_src)
    staged.chmod(0o755)
    return staged


def _run(staged: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(staged), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _parse_gh_log(gh_log: Path) -> list[list[str]]:
    """Return list of argv arrays, one per gh invocation."""
    if not gh_log.exists():
        return []
    raw = gh_log.read_text(encoding="utf-8")
    # Each gh invocation writes NUL-separated argv terminated by \n.
    calls = [c for c in raw.split("\n") if c]
    return [c.split("\x00")[:-1] for c in calls]


def test_pass_verdict_removes_label(tmp_path):
    bin_dir, gh_log = _install_fakes(tmp_path)
    body_file = tmp_path / "body.md"
    body_file.write_text("PASS body")
    sgc_log = tmp_path / "sgc.log"
    staged = _stage_post_review(tmp_path, sgc_log)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "CODEX_AUDIT_PYTHON": sys.executable,
    }

    result = _run(
        staged, env,
        "--lane", "claude-review",
        "--repo", "owner/repo",
        "--pr", "100",
        "--head", "abc1234",
        "--verdict", "pass",
        "--label", "needs-claude-review",
        "--body-file", str(body_file),
    )
    assert result.returncode == 0, result.stderr

    calls = _parse_gh_log(gh_log)
    # Exactly one gh call: pr edit --remove-label
    assert len(calls) == 1, f"expected 1 gh call, got {len(calls)}: {calls}"
    assert calls[0][0] == "pr"
    assert calls[0][1] == "edit"
    assert "--remove-label" in calls[0]
    assert "needs-claude-review" in calls[0]


def test_blocked_verdict_keeps_label(tmp_path):
    bin_dir, gh_log = _install_fakes(tmp_path)
    body_file = tmp_path / "body.md"
    body_file.write_text("BLOCKED body — needs follow-up")
    sgc_log = tmp_path / "sgc.log"
    staged = _stage_post_review(tmp_path, sgc_log)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "CODEX_AUDIT_PYTHON": sys.executable,
    }

    result = _run(
        staged, env,
        "--lane", "claude-review",
        "--repo", "owner/repo",
        "--pr", "200",
        "--head", "def5678",
        "--verdict", "blocked",
        "--label", "needs-claude-review",
        "--body-file", str(body_file),
    )
    assert result.returncode == 0, result.stderr

    # Zero gh calls: label removal must be skipped for blocker verdicts
    # so the queue sweep / standing instructions still see the PR as
    # needing follow-up. This is the central regression test for the
    # Codex P2 finding on cube-snap#173 / ctvd#311.
    calls = _parse_gh_log(gh_log)
    assert calls == [], (
        f"expected 0 gh calls for blocked verdict (label must be "
        f"kept for follow-up), got {len(calls)}: {calls}"
    )
    # And the script should have emitted the keeping-label notice
    # to stderr so the operator sees what happened.
    assert "keeping label" in result.stderr.lower()
    assert "needs-claude-review" in result.stderr


def test_arbitrary_non_pass_verdict_keeps_label(tmp_path):
    """Defense-in-depth: any verdict string that isn't exactly "pass"
    should keep the label. Guards against future verdict values like
    "concerns", "deferred", typos, etc."""
    bin_dir, gh_log = _install_fakes(tmp_path)
    body_file = tmp_path / "body.md"
    body_file.write_text("concerns-level review")
    sgc_log = tmp_path / "sgc.log"
    staged = _stage_post_review(tmp_path, sgc_log)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
        "CODEX_AUDIT_PYTHON": sys.executable,
    }

    for verdict in ("concerns", "deferred", "PASS"):  # uppercase != "pass" — strict match
        gh_log.unlink(missing_ok=True)
        result = _run(
            staged, env,
            "--lane", "claude-review",
            "--repo", "owner/repo",
            "--pr", "300",
            "--head", "feedface",
            "--verdict", verdict,
            "--label", "needs-claude-review",
            "--body-file", str(body_file),
        )
        assert result.returncode == 0, f"verdict={verdict}: {result.stderr}"
        calls = _parse_gh_log(gh_log)
        assert calls == [], (
            f"verdict={verdict}: expected label kept (0 gh calls), "
            f"got {calls}"
        )
