#!/usr/bin/env python3
"""Codex audit CLI — review a single PR via the OpenAI Codex CLI.

Standalone command + Python module. UNLIKE the Devin bridge (cloud webhook)
and the Qwen bridge (local OpenAI-compatible endpoint), the Codex bridge
invokes the local `codex review` CLI as a subprocess. The CLI runs against
a checked-out worktree of the PR branch and produces a code-traced review
that has empirically caught real bugs the other reviewers miss on the same
PR — see the multi-reviewer calibration in PR #233 (Devin PASS + Qwen 7
false positives + Codex 6 real findings).

Pipeline per audit:

  1. Resolve the LOCAL repo path for `owner/repo` from
     `CODEX_AUDIT_REPO_PATHS` (the CLI requires a real git checkout to
     run `codex review --base origin/main` against).
  2. Fetch the PR head SHA via the GitHub API.
  3. `git fetch origin pull/<N>/head` + create a temporary worktree at
     the head SHA (detached, so we don't pollute branch namespace).
  4. Run `codex review --base origin/main` from that worktree, capturing
     stdout/stderr.
  5. Parse the output: extract the final verdict block, count
     `[P0]/[P1]/[P2]/[P3]` severity tags, classify PASS vs BLOCKED.
     Policy: P0/P1/P2 = blocker; P3 = concern (non-blocking). This
     matches how Codex's #233 findings broke down empirically.
  6. Stale-head check: refetch PR head; if it changed mid-review, emit
     the `needs-codex-audit` trailer instead of PASS/BLOCKED.
  7. Post the parsed review as a PR comment ending with the authoritative
     trailer line.
  8. Clean up the worktree.

CLI usage:

    CODEX_AUDIT_REPO_PATHS=\\
      jeffhuber/cube-snap:/Users/jhuber/cube-snap,\\
      jeffhuber/cube-two-view-debugger:/Users/jhuber/cube-two-view-debugger \\
    GITHUB_TOKEN=... \\
    python3 tools/codex_audit_pr.py --repo jeffhuber/cube-two-view-debugger --pr 233

Module usage (called by `codex_audit_bridge.py` if/when we add a polling
daemon analogous to qwen_audit_bridge.py):

    from tools.codex_audit_pr import AuditConfig, audit_pr
    result = audit_pr(config, "jeffhuber/cube-two-view-debugger", 233)

Exit codes (CLI mode):
    0  comment posted (or dry-run printed)
    1  generic error (config, network, Codex CLI failure)
    2  stale head SHA detected mid-review (caller may requeue)

Trailer protocol (mirrors Devin/Qwen):
    <!-- CODEX_AUDIT_STATE: codex-audit-done -->     (verdict was PASS)
    <!-- CODEX_AUDIT_STATE: codex-audit-blocked --> (verdict was BLOCKED)
    <!-- CODEX_AUDIT_STATE: needs-codex-audit -->    (stale head; requeue)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----- Configuration / defaults -----

DEFAULT_CODEX_CLI_PATH = "/Applications/Codex.app/Contents/Resources/codex"
DEFAULT_CODEX_TIMEOUT = 900  # 15 min; Codex review can take 5-10 min on a large diff
DEFAULT_BASE_REF = "origin/main"
DEFAULT_REPOS = "jeffhuber/cube-snap,jeffhuber/cube-two-view-debugger"


# ----- Data classes -----


@dataclass
class AuditConfig:
    github_token: str
    repo_paths: Dict[str, Path]  # "owner/repo" → local checkout path
    codex_cli_path: str = DEFAULT_CODEX_CLI_PATH
    base_ref: str = DEFAULT_BASE_REF
    timeout: int = DEFAULT_CODEX_TIMEOUT
    dry_run: bool = False


@dataclass
class CodexVerdict:
    """Parsed verdict from `codex review` stdout."""
    verdict: str  # "PASS" | "BLOCKED"
    prose: str    # the final review block (clean, ready to render as a comment)
    p0_count: int = 0
    p1_count: int = 0
    p2_count: int = 0
    p3_count: int = 0
    findings: List[str] = field(default_factory=list)  # raw [P*] finding lines

    @property
    def blocker_count(self) -> int:
        # Policy: P0/P1/P2 = blocker; P3 = concern (non-blocking).
        # Matches the empirical severity breakdown on #233 where every P2
        # finding was a real bug worth fixing.
        return self.p0_count + self.p1_count + self.p2_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "p0_count": self.p0_count,
            "p1_count": self.p1_count,
            "p2_count": self.p2_count,
            "p3_count": self.p3_count,
            "blocker_count": self.blocker_count,
            "findings": self.findings,
        }


@dataclass
class AuditResult:
    repo: str
    pr_number: int
    head_sha_start: str
    head_sha_end: str
    verdict: str        # "PASS" | "BLOCKED" | "STALE"
    trailer: str
    comment_body: str
    codex_stdout: str   # stdout from the codex CLI (the review proper)
    codex_stderr: str = ""  # stderr captured separately for debugging
    parsed: Optional[CodexVerdict] = None
    posted_comment_url: Optional[str] = None

    def head_changed_during_review(self) -> bool:
        return self.head_sha_start != self.head_sha_end


# ----- Trailers -----


STALE_TRAILER = "<!-- CODEX_AUDIT_STATE: needs-codex-audit -->"
DONE_TRAILER = "<!-- CODEX_AUDIT_STATE: codex-audit-done -->"
BLOCKED_TRAILER = "<!-- CODEX_AUDIT_STATE: codex-audit-blocked -->"


# ----- GitHub helpers -----


def _gh_request(
    method: str,
    path: str,
    *,
    token: str,
    body: Optional[Dict[str, Any]] = None,
    accept: str = "application/vnd.github+json",
    timeout: int = 30,
) -> Any:
    """Single GitHub REST call. Returns parsed JSON."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=data,
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        text = response.read().decode("utf-8")
        return json.loads(text) if text else None


def fetch_pull_request(repo: str, pr_number: int, *, token: str) -> Dict[str, Any]:
    return _gh_request("GET", f"/repos/{repo}/pulls/{pr_number}", token=token)


def post_pr_comment(repo: str, pr_number: int, body: str, *, token: str) -> Dict[str, Any]:
    return _gh_request(
        "POST",
        f"/repos/{repo}/issues/{pr_number}/comments",
        token=token,
        body={"body": body},
    )


# ----- Codex output parsing -----


# Match `[P0]`, `[P1]`, `[P2]`, `[P3]` finding header tags. Codex emits
# these as the leading token of finding bullets — e.g. `- [P2] title`
# or `* [P2] title`. Codex round 4 of #234 — P2: anchor the regex to the
# bullet-marker prefix so quotes of priority tags inside finding details
# or summary prose don't get counted as real findings (which would turn
# a P3-only or PASS review into BLOCKED).
_P_TAG_RE = re.compile(r"^\s*[-*]\s*\[P([0-3])\]")


def parse_codex_output(stdout: str) -> CodexVerdict:
    """Extract the verdict block from `codex review` stdout.

    Codex's CLI emits a streaming log: interspersed `exec` blocks (commands
    it ran) and intermediate prose, ending with a final review section
    introduced by a single `codex` line on its own. We extract that final
    section, classify the verdict, and count severity tags.

    Output shape (BLOCKED case):

        ... interleaved exec blocks ...

        codex
        <one-sentence summary>

        Full review comments:

        - [P2] <title> — <file:line>
          <details>
        - [P3] <title> — <file:line>
          <details>

    Output shape (PASS case):

        ... interleaved exec blocks ...

        codex
        <one-sentence summary including phrases like "without introducing a
        clear correctness, runtime, or schema issue" or similar>

    The trailing prose can appear once or duplicated by the CLI; we keep
    the first occurrence.
    """
    # Split by lines that are exactly "codex" (the final-verdict marker).
    # The verdict block is everything from the LAST such line onward.
    lines = stdout.splitlines()
    last_codex_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == "codex":
            last_codex_idx = i
    if last_codex_idx == -1:
        # No final `codex` marker found. Codex round 3 of #234 — P2: the
        # original code returned a PASS-shaped verdict here, which would
        # silently mark unaudited PRs as `codex-audit-done` on format
        # drift or empty/partial successful output. The safe behavior is
        # to surface this as UNKNOWN so the orchestrator emits the
        # `needs-codex-audit` (requeue) trailer instead of auto-PASS.
        return CodexVerdict(
            verdict="UNKNOWN",
            prose="(no codex final-verdict block found in output — "
                  "the codex CLI may have produced no review or its "
                  "output format may have drifted; requeue and retry)",
        )

    verdict_block = "\n".join(lines[last_codex_idx + 1 :]).strip()

    # Codex round 4 of #234 — P2: an empty verdict block (the marker is
    # present but no review prose follows — truncated output, CLI format
    # drift, etc.) MUST flow through the UNKNOWN path, not default to
    # PASS. The auto-PASS regression would silently mark an unaudited PR
    # as `codex-audit-done`.
    if not verdict_block:
        return CodexVerdict(
            verdict="UNKNOWN",
            prose="(codex final-verdict marker present but no review prose "
                  "followed — the CLI output may have been truncated or its "
                  "format drifted; requeue and retry)",
        )

    # De-duplicate: Codex sometimes streams the same summary twice. If the
    # block is ≥2 copies of the same prose, keep just the first.
    if len(verdict_block) > 200:
        half = len(verdict_block) // 2
        if verdict_block[:half].strip() == verdict_block[half:].strip():
            verdict_block = verdict_block[:half].strip()

    # Count severity tags
    p_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    findings: List[str] = []
    for line in verdict_block.splitlines():
        m = _P_TAG_RE.search(line)
        if m:
            p_counts[int(m.group(1))] += 1
            findings.append(line.strip())

    blocker_count = p_counts[0] + p_counts[1] + p_counts[2]
    verdict = "BLOCKED" if blocker_count > 0 else "PASS"

    return CodexVerdict(
        verdict=verdict,
        prose=verdict_block,
        p0_count=p_counts[0],
        p1_count=p_counts[1],
        p2_count=p_counts[2],
        p3_count=p_counts[3],
        findings=findings,
    )


# ----- Worktree management -----


def _create_temp_worktree(local_repo: Path, head_sha: str) -> Path:
    """Create a detached worktree at `head_sha` under a tempdir. Returns
    the worktree path."""
    # First make sure the SHA is present locally. `git fetch origin
    # pull/<N>/head` would also work but we don't know the PR number here;
    # use a direct fetch of the commit.
    # Actually for PR commits not on main, we need to ensure the SHA is
    # locally available. The caller (audit_pr) handles this by passing
    # the head SHA after fetching pull/<n>/head.
    tmp_root = Path(tempfile.mkdtemp(prefix="codex-audit-"))
    worktree_path = tmp_root / "wt"
    subprocess.run(
        ["git", "-C", str(local_repo), "worktree", "add", "--detach",
         str(worktree_path), head_sha],
        check=True, capture_output=True, text=True,
    )
    return worktree_path


def _remove_worktree(local_repo: Path, worktree_path: Path) -> None:
    """Remove a worktree, ignoring errors (best-effort cleanup)."""
    try:
        subprocess.run(
            ["git", "-C", str(local_repo), "worktree", "remove", "--force",
             str(worktree_path)],
            check=False, capture_output=True, text=True,
        )
    except Exception:  # noqa: BLE001
        pass
    # Also rmdir the tempdir parent
    try:
        worktree_path.parent.rmdir()
    except Exception:  # noqa: BLE001
        pass


def _fetch_pr_head(local_repo: Path, pr_number: int) -> None:
    """Ensure the PR's head commit is locally available."""
    subprocess.run(
        ["git", "-C", str(local_repo), "fetch", "origin",
         f"pull/{pr_number}/head"],
        check=True, capture_output=True, text=True,
    )


def _fetch_base_ref(local_repo: Path, base_ref: str) -> None:
    """Refresh the base ref (typically `origin/main`) so the codex review
    diff is against current upstream, not a stale local snapshot. Codex
    round 1 of #234 — P2: without this, a long-lived local checkout
    whose `origin/main` is stale would have Codex diff the PR against
    an older base, including already-merged changes and/or missing
    current-base context.

    `base_ref` is in the form passed to `codex review --base` — typically
    `origin/main`. We `git fetch origin main` (the remote branch part)
    to update the corresponding tracking ref.
    """
    # Parse `origin/<branch>` or fall back to the full ref string.
    # Codex meta-review round 2 — P2: avoid str.removeprefix() (Python 3.9+
    # only). The repo's local `python3` is 3.7, and we don't want this
    # tool to require .venv just to extract a branch name.
    if "/" in base_ref and base_ref.startswith("origin/"):
        remote_branch = base_ref[len("origin/"):]
        subprocess.run(
            ["git", "-C", str(local_repo), "fetch", "origin", remote_branch],
            check=True, capture_output=True, text=True,
        )
    else:
        # Generic fetch — let git figure out the remote.
        subprocess.run(
            ["git", "-C", str(local_repo), "fetch", "origin", base_ref],
            check=True, capture_output=True, text=True,
        )


# ----- Codex CLI invocation -----


def run_codex_review(
    config: AuditConfig,
    worktree_path: Path,
) -> Tuple[str, str]:
    """Run `codex review --base <ref>` from the worktree. Returns
    `(stdout, stderr)` — keep them separate so the parser only sees the
    review output proper.

    Codex round 6 of #234 — P2: previously this returned the
    concatenation of stdout + stderr, so progress chatter or warning
    text on stderr (e.g., echoed markdown that contained a `- [P2]`
    bullet, or a stray `codex` line in a quoted command) would
    contaminate `parse_codex_output()` and could mislabel a PASS as
    BLOCKED or trigger a false UNKNOWN.

    Raises `subprocess.CalledProcessError` on non-zero exit (Codex
    round 1 of #234 — P2: without this check, broken-Codex runs would
    silently flow into `codex-audit-done`).
    """
    if not Path(config.codex_cli_path).exists():
        raise FileNotFoundError(
            f"Codex CLI not found at {config.codex_cli_path}. "
            f"Install Codex (https://codex.dev) or override "
            f"CODEX_CLI_PATH."
        )
    result = subprocess.run(
        [config.codex_cli_path, "review", "--base", config.base_ref],
        cwd=str(worktree_path),
        capture_output=True,
        text=True,
        timeout=config.timeout,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            [config.codex_cli_path, "review", "--base", config.base_ref],
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout, result.stderr


# ----- Comment formatting -----


def format_comment(
    parsed: CodexVerdict,
    head_sha: str,
    is_stale: bool = False,
    stale_end_sha: Optional[str] = None,
    is_unknown: bool = False,
) -> str:
    """Build the GitHub comment body with header, prose, and trailer."""
    header = "## Codex audit (calibration phase — informational only)\n\n"
    header += f"Head SHA: `{head_sha}`\n"
    if is_stale:
        return (
            header
            + f"\nHead SHA changed during review (`{head_sha[:8]}` → "
            + f"`{(stale_end_sha or '?')[:8]}`). Skipping this verdict and "
            + f"requeuing for re-review of the new head.\n\n"
            + STALE_TRAILER
            + "\n"
        )
    if is_unknown:
        return (
            header
            + "\nCould not parse a Codex verdict from the CLI output "
            + "(no `codex` final-verdict marker found). The CLI may have "
            + "produced no review, or its output format may have drifted. "
            + "Requeuing for re-review.\n\n"
            + STALE_TRAILER
            + "\n"
        )

    header += (
        f"Findings: P0={parsed.p0_count}, P1={parsed.p1_count}, "
        f"P2={parsed.p2_count}, P3={parsed.p3_count} "
        f"(blocker policy: any P0/P1/P2 → BLOCKED)\n\n"
    )

    verdict_line = (
        "Codex Audit: BLOCKED" if parsed.verdict == "BLOCKED"
        else "Codex Audit: PASS"
    )
    trailer = BLOCKED_TRAILER if parsed.verdict == "BLOCKED" else DONE_TRAILER

    body_lines = [
        header.rstrip(),
        "",
        verdict_line,
        "",
        parsed.prose.rstrip(),
        "",
        trailer,
    ]
    return "\n".join(body_lines) + "\n"


# ----- Orchestration -----


def audit_pr(config: AuditConfig, repo: str, pr_number: int) -> AuditResult:
    """End-to-end audit of one PR. Creates a temporary worktree at the PR
    head, runs `codex review`, parses output, formats + posts a comment."""
    local_repo = config.repo_paths.get(repo)
    if local_repo is None:
        raise ValueError(
            f"no local repo path configured for {repo}. Set "
            f"CODEX_AUDIT_REPO_PATHS=owner/repo:/path[,owner/repo:/path,...]"
        )
    if not local_repo.exists():
        raise FileNotFoundError(
            f"configured local repo path does not exist: {local_repo}"
        )

    pr_meta = fetch_pull_request(repo, pr_number, token=config.github_token)
    head_sha_start = pr_meta["head"]["sha"]

    print(f"audit {repo}#{pr_number} head={head_sha_start[:8]} "
          f"(local: {local_repo})", file=sys.stderr, flush=True)

    # Ensure both the PR head AND the base ref are locally fetched
    # before running the review. Stale base = wrong diff = wrong review.
    _fetch_pr_head(local_repo, pr_number)
    _fetch_base_ref(local_repo, config.base_ref)

    # Codex round 6 of #234 — P3: handle the force-push race where
    # `head_sha_start` was read before the user force-pushed and our
    # `pull/<N>/head` fetch now points at the new SHA. If the old SHA
    # isn't locally available, treat it as a stale-head condition
    # rather than crashing on the worktree create.
    try:
        worktree_path = _create_temp_worktree(local_repo, head_sha_start)
    except subprocess.CalledProcessError as exc:
        # Worktree create failed — likely because the SHA isn't in the
        # local repo anymore (force-push race). Refetch and compare;
        # if the head has moved, emit the stale-requeue comment.
        pr_meta_after = fetch_pull_request(repo, pr_number, token=config.github_token)
        head_sha_after = pr_meta_after["head"]["sha"]
        if head_sha_after != head_sha_start:
            print(f"  force-push race: head moved from {head_sha_start[:8]} "
                  f"to {head_sha_after[:8]} during fetch; emitting STALE",
                  file=sys.stderr, flush=True)
            comment_body = format_comment(
                CodexVerdict(verdict="UNKNOWN",
                             prose="(force-push detected before worktree created)"),
                head_sha_start, is_stale=True, stale_end_sha=head_sha_after,
            )
            result = AuditResult(
                repo=repo, pr_number=pr_number,
                head_sha_start=head_sha_start, head_sha_end=head_sha_after,
                verdict="STALE", trailer=STALE_TRAILER,
                comment_body=comment_body, codex_stdout="", codex_stderr="",
            )
            if not config.dry_run:
                posted = post_pr_comment(repo, pr_number, comment_body,
                                          token=config.github_token)
                result.posted_comment_url = posted.get("html_url")
            return result
        # Same head; the worktree failure is for some other reason.
        # Re-raise so the outer error path runs.
        raise

    try:
        t0 = time.time()
        codex_stdout, codex_stderr = run_codex_review(config, worktree_path)
        dt = time.time() - t0
        print(f"  codex review completed in {dt:.0f}s", file=sys.stderr, flush=True)
    finally:
        _remove_worktree(local_repo, worktree_path)

    # Parse Codex's verdict.
    parsed = parse_codex_output(codex_stdout)

    # Stale-head check: refetch PR head and compare. If it changed mid-review,
    # the verdict applies to a no-longer-current SHA — requeue.
    pr_meta_after = fetch_pull_request(repo, pr_number, token=config.github_token)
    head_sha_end = pr_meta_after["head"]["sha"]
    is_stale = head_sha_start != head_sha_end

    if is_stale:
        comment_body = format_comment(parsed, head_sha_start, is_stale=True,
                                       stale_end_sha=head_sha_end)
        result_verdict = "STALE"
        trailer = STALE_TRAILER
    elif parsed.verdict == "UNKNOWN":
        # Codex round 3 of #234 — P2: when the parser can't find the
        # `codex` final-verdict marker, requeue rather than auto-PASS.
        # The trailer is the same as STALE because the downstream
        # outcome (re-run, don't trust this comment) is identical.
        comment_body = format_comment(parsed, head_sha_start, is_unknown=True)
        result_verdict = "UNKNOWN"
        trailer = STALE_TRAILER
    else:
        comment_body = format_comment(parsed, head_sha_start)
        result_verdict = parsed.verdict
        trailer = BLOCKED_TRAILER if parsed.verdict == "BLOCKED" else DONE_TRAILER

    result = AuditResult(
        repo=repo,
        pr_number=pr_number,
        head_sha_start=head_sha_start,
        head_sha_end=head_sha_end,
        verdict=result_verdict,
        trailer=trailer,
        comment_body=comment_body,
        codex_stdout=codex_stdout,
        codex_stderr=codex_stderr,
        parsed=parsed,
    )

    if not config.dry_run:
        posted = post_pr_comment(repo, pr_number, comment_body,
                                  token=config.github_token)
        result.posted_comment_url = posted.get("html_url")
        print(f"posted {repo}#{pr_number} verdict={result_verdict} "
              f"url={result.posted_comment_url}", file=sys.stderr)

    return result


# ----- CLI entry point -----


def _parse_repo_paths(spec: str) -> Dict[str, Path]:
    """Parse `owner/repo:/path,owner/repo:/path,...` into a dict."""
    out: Dict[str, Path] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                f"bad CODEX_AUDIT_REPO_PATHS entry: {entry!r} (expected owner/repo:/path)"
            )
        repo, path = entry.split(":", 1)
        out[repo.strip()] = Path(path.strip())
    return out


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Codex audit CLI — review a single pull request.",
    )
    ap.add_argument("--repo", required=True, help="owner/repo, e.g. jeffhuber/cube-snap")
    ap.add_argument("--pr", type=int, required=True, help="PR number")
    ap.add_argument("--codex-cli-path",
                    default=os.environ.get("CODEX_CLI_PATH", DEFAULT_CODEX_CLI_PATH))
    ap.add_argument("--base-ref", default=os.environ.get("CODEX_BASE_REF", DEFAULT_BASE_REF),
                    help="base ref to diff against (default: origin/main)")
    ap.add_argument("--repo-paths",
                    default=os.environ.get("CODEX_AUDIT_REPO_PATHS", ""),
                    help="comma-separated owner/repo:/path entries. Required.")
    ap.add_argument("--timeout", type=int,
                    default=int(os.environ.get("CODEX_AUDIT_TIMEOUT", DEFAULT_CODEX_TIMEOUT)))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=bool(os.environ.get("CODEX_AUDIT_DRY_RUN")),
        help="Print the audit comment to stdout instead of posting it.",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("error: GITHUB_TOKEN env var is required", file=sys.stderr)
        return 1
    if not args.repo_paths:
        print("error: --repo-paths or CODEX_AUDIT_REPO_PATHS is required",
              file=sys.stderr)
        return 1

    try:
        repo_paths = _parse_repo_paths(args.repo_paths)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    config = AuditConfig(
        github_token=token,
        repo_paths=repo_paths,
        codex_cli_path=args.codex_cli_path,
        base_ref=args.base_ref,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )

    try:
        result = audit_pr(config, args.repo, args.pr)
    except urllib.error.HTTPError as exc:
        print(f"error: GitHub API HTTP {exc.code} — {exc.reason}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"error: network — {exc}", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print(f"error: codex review timed out after {args.timeout}s",
              file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"error: subprocess failed — {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(result.comment_body)

    if result.verdict == "STALE":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
