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
    tools/run_codex_audit_pr.sh --repo jeffhuber/cube-two-view-debugger --pr 233

Prefer the wrapper over direct `python3 tools/codex_audit_pr.py`: it
selects a controlled Python interpreter for this script and refuses to
fall back silently to ambient system Python.

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
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
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
    # Optional override for the Python venv whose `bin/` is prepended to
    # PATH (and exported as VIRTUAL_ENV) when invoking `codex review`.
    # If None AND `disable_venv` is False, `audit_pr()` auto-discovers
    # `<local_repo>/.venv/`.
    #
    # Without this, `codex review`'s subprocesses (pytest, build_oracle_*,
    # etc.) inherit the audit machine's system PATH — on a typical
    # macOS dev box that resolves to anaconda Python 3.7.6 with stale
    # numpy/Pillow, instead of the canonical .venv (Python 3.12.13,
    # numpy 2.3.5, Pillow 12.2.0 per corpus_manifest.json). The drift
    # produced bogus per-pixel diffs on PR #262 round 3.
    venv_path: Optional[Path] = None
    # Explicit kill switch for venv injection: when True, audit_pr()
    # does NOT auto-discover even if `<local_repo>/.venv/` exists.
    # The CLI sets this via `--venv-path ""` / empty
    # CODEX_AUDIT_VENV_PATH. Separate field rather than overloading
    # venv_path with a `Path("")` sentinel — round-4 P2 on PR #264:
    # `str(Path(""))` returns `"."`, not `""`, so the sentinel could
    # accidentally inject `./bin` if cwd had one.
    disable_venv: bool = False


@dataclass
class CodexVerdict:
    """Parsed verdict from `codex review` output (stdout, or stderr fallback)."""
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


def _looks_like_codex_verdict_block(block: str) -> bool:
    """Validate a candidate verdict-block extracted from stderr-fallback.

    Returns True if the block contains at least one P-tag finding
    line (BLOCKED-shape, or PASS-shape with only P3 findings).
    Returns False otherwise — in which case the caller should treat
    the parse as UNKNOWN and requeue.

    Why P-tag presence specifically (and NOT a phrase / prose
    heuristic): Codex P1 audit on cube-snap#198 f52d089 caught that
    *any* substring-based phrase heuristic on free-form text is
    fundamentally too leaky. The codex review subprocess regularly
    cats files, runs tests, and prints diffs whose content can
    contain arbitrary user-provided strings — including phrases
    like "without introducing" or "no actionable" embedded in
    commit messages, fixtures, or quoted source. A column-0 `codex`
    line that lands by coincidence inside such output, followed by
    truncation, would satisfy any phrase check and produce
    auto-PASS. The P-tag pattern (`- [P[0-3]] <title> — <file:line>`)
    is enough of a structural signal that incidental chatter is
    extremely unlikely to mimic it.
    The cost: real Codex PASS verdicts that arrive *only via the
    stderr fallback path* AND have zero findings (no P-tags at
    all) will fall to UNKNOWN. That case is rare (stderr-routed
    verdicts are a CLI flake; the next attempt usually emits to
    stdout normally) and the UNKNOWN trailer auto-requeues, so the
    visible cost is one extra audit attempt. Acceptable in
    exchange for closing the auto-PASS-from-chatter regression.
    Stdout-anchored blocks bypass this check: the column-0 `codex`
    line in stdout is the canonical final-verdict anchor and has
    not been observed as a false positive in practice.
    """
    for line in block.splitlines():
        if _P_TAG_RE.match(line):
            return True
    return False


def _codex_marker_indices(lines: List[str]) -> List[int]:
    """Return indices of column-0 `codex` verdict markers."""
    return [i for i, line in enumerate(lines) if line.rstrip() == "codex"]


def parse_codex_output(stdout: str, stderr: str = "") -> CodexVerdict:
    """Extract the verdict block from `codex review` output.

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
    # The verdict block is everything from the LAST stdout marker onward.
    #
    # Codex round 7 of #234 — P2: the original code used `line.strip() ==
    # "codex"` which would match indented `  codex  ` lines INSIDE the
    # final review prose (e.g., when the review quotes the documented
    # transcript shape). The last-marker-wins logic would then anchor
    # on the quoted occurrence and discard the real findings above it.
    # Codex's actual transcript markers are at column 0 with no
    # indentation. Match exactly, allowing only trailing whitespace.
    lines = stdout.splitlines()
    marker_indices = _codex_marker_indices(lines)

    if marker_indices:
        last_codex_idx = marker_indices[-1]
    else:
        # Task #85 / PR #255: the Codex CLI can occasionally route the
        # final verdict marker to stderr even though the normal review
        # channel is stdout. Fall back to the LAST column-0 `codex`
        # marker in stderr, applying the same last-marker-wins rule
        # that already governs the stdout path.
        #
        # Updated by cube-snap#198 after the CLI-failure capture from
        # #196 produced empirical evidence: ctvd#358's three repeated
        # UNKNOWN verdicts (captured at
        # ~/.cache/cube-agent-audits/cli-failures/) were ALL the same
        # shape — stdout had the review prose with no `codex` marker,
        # stderr had multiple column-0 `codex` markers, the last of
        # which preceded the real final-verdict block. The previous
        # "exactly one stderr marker" guard refused these as ambiguous
        # progress chatter; the captured dumps showed they were
        # genuine final verdicts.
        #
        # The column-0 exact-match filter (in _codex_marker_indices)
        # is what makes the last-marker-wins rule safe for both
        # streams: Codex's `exec` log lines and quoted-in-prose
        # mentions of "codex" are not column-0 bare lines, so they
        # don't create false markers.
        stderr_lines = stderr.splitlines()
        stderr_marker_indices = _codex_marker_indices(stderr_lines)
        if stderr_marker_indices:
            # Validate the candidate block before treating it as
            # authoritative (cube-snap P2 audit on #198 → #199):
            # Codex CLI output can emit incidental column-0 `codex`
            # lines from log/test chatter BEFORE the real final
            # verdict block, then get truncated or never reach the
            # real verdict. The last-marker-wins rule alone would
            # anchor on the incidental marker, find no P-tag in the
            # following chatter, and auto-PASS — silently marking
            # an unaudited PR done. The validator requires either
            # a P-tag finding or canonical Codex-verdict prose
            # phrase in the candidate block before accepting it.
            candidate_idx = stderr_marker_indices[-1]
            candidate_block = "\n".join(
                stderr_lines[candidate_idx + 1:]
            ).strip()
            if _looks_like_codex_verdict_block(candidate_block):
                lines = stderr_lines
                last_codex_idx = candidate_idx
            else:
                return CodexVerdict(
                    verdict="UNKNOWN",
                    prose="(stderr fallback found a column-0 `codex` marker "
                          "but the following text has no P-tag and no "
                          "canonical Codex-verdict prose phrase — likely "
                          "an incidental log line ahead of a truncated or "
                          "format-drifted real verdict; requeue and retry)",
                )
        else:
            # No final `codex` marker found in stdout OR stderr.
            # Codex round 3 of #234 — P2: the original code returned a
            # PASS-shaped verdict here, which would silently mark
            # unaudited PRs as `codex-audit-done` on format drift or
            # empty/partial successful output. The safe behavior is to
            # surface this as UNKNOWN so the orchestrator emits the
            # `needs-codex-audit` (requeue) trailer instead of auto-PASS.
            return CodexVerdict(
                verdict="UNKNOWN",
                prose="(no codex final-verdict block found in stdout or "
                      "stderr — the codex CLI may have produced no review "
                      "or its output format may have drifted; requeue "
                      "and retry)",
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


# ----- CLI parse-failure capture -----


DEFAULT_CLI_FAILURE_DIR = (
    Path.home() / ".cache" / "cube-agent-audits" / "cli-failures"
)


def dump_cli_failure(
    repo: str,
    pr_number: int,
    head_sha: str,
    stdout: str,
    stderr: str,
    reason: str,
    base_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Persist raw Codex CLI output to disk when verdict-parsing failed.

    The orchestrator auto-requeues on UNKNOWN, but the raw bytes that
    confused the parser are not otherwise preserved — the worktree gets
    torn down and the next audit's output overwrites the in-memory
    capture. Without this dump, diagnosing "is this a real parser bug
    or a one-shot Codex CLI flake?" requires reproducing the original
    audit, which often is not possible (head SHA moves on).

    Output: `<base_dir>/<owner>_<name>_pr<N>_<sha[:12]>_<UTC>.log`.
    The file is self-describing — header includes repo/pr/sha/reason
    plus stdout/stderr byte counts so it stays useful when attached
    to an issue without the surrounding context.

    File permissions: 0o600 (owner read/write only). Directory: 0o700.
    The raw output can contain source snippets and credentials
    accidentally printed by PR-controlled subprocesses, so on a
    shared audit host the default umask (often 0o644 file / 0o755
    dir) would leak audit data to other users on the box. Created
    via `os.open(..., mode=0o600)` rather than `write_text()` +
    `chmod()` so there's no TOCTOU window where the file briefly
    exists with the umask perms before being restricted. (Codex P2
    audit on cube-snap#196 5e97011.)

    Symlink / pre-existing-path safety: open uses `O_EXCL | O_NOFOLLOW`
    in addition to `O_CREAT`, so a PR-controlled subprocess that
    pre-creates the dump path (predictable filename) as either a
    symlink-to-sensitive-target or a world-readable regular file
    cannot trick this helper into truncating an arbitrary file or
    leaving our 0o600 mode unenforced (`mode` is ignored for
    existing targets). A collision fails the open; the OSError
    handler returns None and the dump is just skipped — preferable
    to writing potentially-credential-laden content into an attacker-
    controlled path. The timestamp includes microseconds to make
    natural collisions vanishingly rare. (Codex P2 audit on
    cube-snap#196 17b4156.)

    Returns the dump path on success, or None if the dump failed
    (best-effort; we never want this helper to interrupt an audit).

    `base_dir` is parameterized for tests; defaults to
    `DEFAULT_CLI_FAILURE_DIR` (`~/.cache/cube-agent-audits/cli-failures/`).
    """
    try:
        if base_dir is None:
            base_dir = DEFAULT_CLI_FAILURE_DIR
        # `mkdir(mode=0o700)` only takes effect on dirs we create; if the
        # dir already exists with looser perms, mkdir(exist_ok=True) is
        # a no-op. Chmod afterward to make it idempotent. Wrap in its
        # own try so a chmod failure (we don't own a pre-existing dir)
        # doesn't drop the dump — the file's own 0o600 still protects
        # the content.
        base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(base_dir, 0o700)
        except OSError:
            pass

        owner_name = repo.replace("/", "_")
        short_sha = (head_sha or "unknown")[:12]
        # Microsecond resolution — same-second collisions across audits
        # would have caused O_EXCL (below) to refuse the open.
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        path = base_dir / f"{owner_name}_pr{pr_number}_{short_sha}_{ts}.log"

        # Codex P3 audit on cube-snap#196 / ctvd#358 0eff691/9cd9213:
        # `len(str)` counts Python characters, not UTF-8 bytes. The
        # CLI output frequently contains non-ASCII (em-dashes, smart
        # quotes, etc.), so a char-count header on a multi-byte body
        # would under-report and mislead anyone investigating
        # truncation. Encode then measure to get the real byte count
        # that landed on disk via `fdopen(..., encoding="utf-8")`.
        stdout_bytes = len(stdout.encode("utf-8"))
        stderr_bytes = len(stderr.encode("utf-8"))
        header = (
            "# Codex CLI raw output — verdict UNKNOWN\n"
            f"# repo: {repo}\n"
            f"# pr: {pr_number}\n"
            f"# head_sha: {head_sha}\n"
            f"# captured_at: {ts}\n"
            f"# parser_reason: {reason}\n"
            f"# stdout_bytes: {stdout_bytes}\n"
            f"# stderr_bytes: {stderr_bytes}\n"
        )
        body = (
            header
            + "\n===== STDOUT =====\n"
            + (stdout if stdout else "<empty>\n")
            + "\n===== STDERR =====\n"
            + (stderr if stderr else "<empty>\n")
        )
        # Atomic create-with-mode + symlink/pre-existing refusal:
        # - O_CREAT|O_EXCL: refuses if path already exists; together
        #   they guarantee our 0o600 mode is enforced (the mode arg
        #   is ignored by the kernel for existing targets).
        # - O_NOFOLLOW: refuses if the final path component is a
        #   symlink, so a pre-planted symlink-to-sensitive-file
        #   cannot redirect our write.
        # On collision/symlink: open raises OSError → caught by the
        # outer try → dump skipped (None returned). Better to lose a
        # diagnostic dump than write credential-laden bytes to an
        # attacker-controlled path.
        fd = os.open(
            str(path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        return path
    except OSError as exc:
        # Never propagate: a logging failure must not break the audit.
        print(
            f"  warning: failed to dump CLI output: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None


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


def _discover_venv(local_repo: Path) -> Optional[Path]:
    """Look for a venv at `<local_repo>/.venv/` and return its
    ABSOLUTE path if it contains a working `bin/python`. Returns None
    otherwise.

    Used by `audit_pr()` when `config.venv_path` is unset and
    `config.disable_venv` is False, so most repos get the right
    Python without any explicit configuration. Repos without a venv
    (e.g. cube-snap, which uses npm/vitest for tests) get a no-op.

    The returned path is `resolve()`d so the subprocess (which runs
    with `cwd=worktree_path`) can still find `bin/python` when the
    caller's `local_repo` was a relative path. Round-4 P2 on PR #264.
    """
    candidate = (local_repo / ".venv").resolve()
    if (candidate / "bin" / "python").exists():
        return candidate
    return None


def _build_subprocess_env(venv_path: Optional[Path]) -> Dict[str, str]:
    """Return an env dict for the `codex review` subprocess.

    When `venv_path` is a real path with `bin/python`, prepend
    `<venv>/bin` to PATH and export VIRTUAL_ENV so any Python
    invocations from `codex review` (pytest, diagnostic tools, etc.)
    pick up the right interpreter. Also clear PYTHONHOME if set —
    leaving it pointing at a different Python install breaks the
    venv's site-packages resolution.

    `venv_path` is `resolve()`d to an absolute path before injection.
    The subprocess runs with `cwd=worktree_path`, so any RELATIVE
    PATH entry would resolve from the temp worktree, not the
    caller's cwd. Round-4 P2 on PR #264: previously a relative
    `--venv-path .venv` got injected literally and the subprocess
    then searched `<worktree>/.venv/bin` (which doesn't exist) and
    silently fell back to the ambient interpreter.

    No-op (returns os.environ unchanged) when `venv_path` is None
    or its resolved form doesn't point at a working `bin/python`.
    Codex P3 fix for PR #262 (issue tracked as Task #97): without
    this, audits inherit the system PATH which on a typical macOS
    dev box resolves to anaconda Python 3.7.6 with numpy 1.18 /
    Pillow 9.5 — producing per-pixel drift relative to the
    canonical .venv (3.12.13 / 2.3.5 / 12.2.0).
    """
    env = dict(os.environ)
    # The `codex review` subprocess executes scripts/tooling from the
    # PR worktree under audit — that is untrusted code by definition.
    # GITHUB_TOKEN / GH_TOKEN in this script's parent env are used for
    # GitHub API calls from THIS process (PR fetch, comment post). They
    # must not leak into the codex subprocess where the PR code runs.
    # Caught by Codex P1 audit on cube-snap#194 / cube-two-view-debugger#354:
    # `gh auth token` fallback in the wrapper made every audit caller a
    # potential exposure, but the same issue applied to manually-exported
    # tokens before that. This sanitization closes both paths.
    for sensitive in ("GITHUB_TOKEN", "GH_TOKEN"):
        env.pop(sensitive, None)
    if venv_path is None:
        return env
    # Path("") and Path(".") both stringify as "." and resolve to cwd.
    # If cwd happens to have ./bin/python (e.g. the audit machine
    # was started from a directory shaped like a venv) the resolution
    # below would naively inject cwd as VIRTUAL_ENV — exactly the
    # silent-cwd-injection bug round-4 was meant to fix. Round-4
    # round-2 P2 on PR #264: refuse to inject for empty-ish paths
    # regardless of cwd contents. CLI callers never reach here with
    # Path("") because main() routes `--venv-path ""` through
    # `disable_venv=True`, but library callers can.
    if str(venv_path) in ("", "."):
        return env
    abs_venv = venv_path.resolve()
    if not (abs_venv / "bin" / "python").exists():
        warnings.warn(
            f"venv path {abs_venv} has no bin/python; "
            "leaving subprocess environment unchanged",
            RuntimeWarning,
            stacklevel=2,
        )
        return env
    venv_bin = str(abs_venv / "bin")
    env["VIRTUAL_ENV"] = str(abs_venv)
    existing_path = env.get("PATH", "")
    env["PATH"] = (
        f"{venv_bin}{os.pathsep}{existing_path}"
        if existing_path else venv_bin
    )
    env.pop("PYTHONHOME", None)
    return env


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
    env = _build_subprocess_env(config.venv_path)
    result = subprocess.run(
        [config.codex_cli_path, "review", "--base", config.base_ref],
        cwd=str(worktree_path),
        capture_output=True,
        text=True,
        timeout=config.timeout,
        env=env,
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

    # Auto-discover a venv under the local repo and use it for the
    # `codex review` subprocess unless the caller explicitly set one
    # or explicitly disabled discovery via `--venv-path ""`. Without
    # this, audits inherit the system PATH which on a typical macOS
    # dev box resolves to anaconda Python 3.7.6 instead of the
    # canonical .venv (Task #97 / Codex P3 on PR #262). Mutating the
    # config here is local to `audit_pr` (the caller's config dict
    # isn't shared across audits via mutable state).
    #
    # Round-4 P2 on PR #264: the previous version used `Path("")` as
    # the disable sentinel and checked `str(venv_path) == ""` in
    # `_build_subprocess_env`. But `str(Path(""))` returns `"."`,
    # not `""`, so the sentinel never tripped — and if cwd happened
    # to have `./bin/python` the "disable" path silently injected
    # that bogus Python. Replaced with an explicit `disable_venv`
    # bool field; sentinel is gone.
    if config.venv_path is None and not config.disable_venv:
        discovered_venv = _discover_venv(local_repo)
        if discovered_venv is not None:
            print(
                f"  using venv {discovered_venv} for subprocess Python "
                "(set --venv-path to override or empty string to disable)",
                file=sys.stderr, flush=True,
            )
            config = replace(config, venv_path=discovered_venv)

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
                # Loud-failure contract: post_pr_comment() raises on any
                # HTTP error or network failure. DO NOT catch here. The
                # wrapper (tools/run_codex_audit_pr.sh) relies on this
                # script exiting non-zero when posting fails so its
                # finish_lock trap can log status="failed" instead of
                # "completed". A silent catch would re-create the
                # cube-snap#184 silent-failure mode where three audit
                # runs reported exitCode=0 but never posted a GitHub
                # comment. See run_codex_audit_pr.sh's finish_lock
                # comment for the wrapper-side history and the memory
                # file feedback_silent_success_silent_failure.md for the
                # general lesson.
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
    parsed = parse_codex_output(codex_stdout, codex_stderr)

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
        #
        # Persist the raw CLI bytes that confused the parser. The
        # post-comment trailer auto-requeues, but the next attempt
        # overwrites the in-memory capture and the worktree is gone,
        # so without this dump there's no way to diagnose "real
        # parser bug or one-shot Codex CLI flake?" after the fact.
        # Best-effort: a logging failure must not interrupt the audit.
        dump_path = dump_cli_failure(
            repo=repo,
            pr_number=pr_number,
            head_sha=head_sha_start,
            stdout=codex_stdout,
            stderr=codex_stderr,
            reason=parsed.prose,
        )
        if dump_path is not None:
            print(
                f"  captured CLI output for diagnosis: {dump_path}",
                file=sys.stderr,
                flush=True,
            )
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
        # Loud-failure contract: see the matching comment ~70 lines above
        # at the STALE-handler call site. tl;dr: post_pr_comment() raising
        # is the only signal the wrapper's finish_lock has that posting
        # failed. Do not catch the exception here.
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
    ap.add_argument(
        "--venv-path",
        default=os.environ.get("CODEX_AUDIT_VENV_PATH"),
        help=(
            "Python venv whose bin/ is prepended to PATH (and exported "
            "as VIRTUAL_ENV) when running `codex review`. Default: "
            "auto-discover `<local_repo>/.venv/`. Pass empty string to "
            "disable. Without this, audits inherit the system PATH "
            "which typically resolves to a stale system Python instead "
            "of the canonical .venv (see corpus_manifest.json for the "
            "pinned environment)."
        ),
    )
    ap.add_argument(
        "--read-token-from-stdin",
        action="store_true",
        help=(
            "Read GITHUB_TOKEN from the first line of stdin instead of "
            "os.environ. The wrapper passes this so the token never "
            "appears in this Python process's initial environment, "
            "which on Linux remains visible via /proc/<pid>/environ "
            "to PR-controlled subprocesses even after "
            "os.environ.pop(). When set, GITHUB_TOKEN / GH_TOKEN env "
            "vars are also cleared as defense-in-depth."
        ),
    )
    return ap.parse_args(argv)


def _extract_and_clear_github_token() -> Optional[str]:
    """Legacy: read GITHUB_TOKEN from os.environ, then remove it (and
    GH_TOKEN) from os.environ.

    `os.environ.pop` clears the var from Python's view, but on Linux
    the kernel-exposed /proc/<pid>/environ shows the INITIAL exec
    environment block, not Python's live runtime env. So if the
    parent process exec'd Python with GITHUB_TOKEN in env, that
    initial-env exposure persists for the lifetime of the Python
    process regardless of pop(). The wrapper avoids this by piping
    the token via stdin and passing --read-token-from-stdin instead;
    this function is kept for callers that invoke `codex_audit_pr.py`
    directly without the wrapper, where the /proc exposure is
    accepted as part of the caller's threat model.

    Returns the token value (kept only in a local variable, NOT
    re-exported via os.environ), or None if no token was present.
    """
    token = os.environ.pop("GITHUB_TOKEN", None)
    # GH_TOKEN is the gh CLI's alternate env name; strip it too even if
    # unused here, since the same exposure applies.
    os.environ.pop("GH_TOKEN", None)
    return token


def _resolve_github_token(read_from_stdin: bool) -> Optional[str]:
    """Return the GITHUB_TOKEN, either piped via stdin (wrapper path —
    avoids /proc/<pid>/environ exposure of the initial Python env on
    Linux) or read from os.environ (legacy path for direct callers).

    The wrapper passes --read-token-from-stdin and pipes the token as
    the first line of stdin. When that flag is set, this function
    reads stdin and ALSO clears GITHUB_TOKEN / GH_TOKEN from
    os.environ as defense-in-depth (in case the caller accidentally
    also exported the env vars — the wrapper unsets them before exec
    but a misuse path could leave them).

    Returns None if no token can be resolved by either mechanism.
    """
    if read_from_stdin:
        line = sys.stdin.readline()
        # Belt-and-suspenders: even though the wrapper unsets these
        # before exec, clear them if they somehow made it through.
        os.environ.pop("GITHUB_TOKEN", None)
        os.environ.pop("GH_TOKEN", None)
        if not line:
            return None
        return line.rstrip("\r\n")
    return _extract_and_clear_github_token()


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    token = _resolve_github_token(args.read_token_from_stdin)
    if not token:
        if args.read_token_from_stdin:
            print(
                "error: --read-token-from-stdin was passed but stdin "
                "did not contain a token",
                file=sys.stderr,
            )
        else:
            print("error: GITHUB_TOKEN env var is required (or pass "
                  "--read-token-from-stdin and pipe the token in)",
                  file=sys.stderr)
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

    # --venv-path semantics:
    #   not set         (None)         -> auto-discover <local_repo>/.venv/
    #   non-empty str                  -> use that path (resolved to absolute)
    #   empty str       ("")           -> explicit disable (disable_venv=True)
    #
    # Round-4 P2 on PR #264:
    # - Empty string is handled via the explicit `disable_venv` field,
    #   not a `Path("")` sentinel. `str(Path(""))` returns `"."` and
    #   could accidentally inject `./bin` if cwd had one.
    # - Non-empty paths are resolved to absolute BEFORE storing on the
    #   config, so a relative `--venv-path .venv` still finds its
    #   bin/python when the subprocess runs with `cwd=worktree_path`.
    explicit_venv: Optional[Path]
    disable_venv = False
    if args.venv_path is None:
        explicit_venv = None
    elif args.venv_path == "":
        explicit_venv = None
        disable_venv = True
    else:
        explicit_venv = Path(args.venv_path).resolve()
    config = AuditConfig(
        github_token=token,
        repo_paths=repo_paths,
        codex_cli_path=args.codex_cli_path,
        base_ref=args.base_ref,
        timeout=args.timeout,
        dry_run=args.dry_run,
        venv_path=explicit_venv,
        disable_venv=disable_venv,
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

    # Codex round 7 of #234 — P2: both STALE and UNKNOWN result in a
    # `needs-codex-audit` requeue comment. Automation using the exit
    # code to decide whether to retry must see exit 2 for BOTH cases;
    # otherwise UNKNOWN looks like a successful audit even though the
    # comment requests re-review.
    if result.verdict in ("STALE", "UNKNOWN"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
