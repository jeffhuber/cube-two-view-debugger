#!/usr/bin/env python3
"""Qwen audit CLI — review a single PR.

Standalone command + Python module. Reviews ONE PR end-to-end:

  1. Fetch PR metadata + per-file diffs from GitHub API.
  2. For each changed file (under size budget): fetch full file content at
     head SHA. This is the "local checkout" context Codex flagged as
     necessary — diff-only review is too weak.
  3. Per-file Qwen pass: prompt sees full file content + diff hunks + PR
     title/body. Asks Qwen to classify findings as BLOCKER / CONCERN /
     NONE for that file.
  4. Synthesis Qwen pass: gathers per-file findings + cross-cutting
     concerns (test coverage, doc-index, lane discipline) → final verdict.
  5. Refetch head SHA at end. If it changed mid-review, emit the
     `needs-qwen-audit` trailer instead of PASS/BLOCKED.
  6. Post a structured comment ending with the authoritative trailer.

CLI usage:

    GITHUB_TOKEN=... python3 tools/qwen_audit_pr.py \
        --repo jeffhuber/cube-snap --pr 142

Module usage (called by `qwen_audit_bridge.py`):

    from tools.qwen_audit_pr import AuditConfig, audit_pr
    result = audit_pr(config, "jeffhuber/cube-snap", 142)
    # result.posted_comment_url, result.verdict, result.trailer

Exit codes (CLI mode):
    0  comment posted (or dry-run printed)
    1  generic error (config, network, API)
    2  stale head SHA detected mid-review (caller may requeue)

Severity contract (REQUIRED in the prompt):
    BLOCKER  — must fix before merge (correctness, test gap, secrets, etc.)
    CONCERN  — non-blocking observation
    NONE     — no issues

Final verdict:
    BLOCKED  — any BLOCKER findings
    PASS     — no BLOCKERs (CONCERNs allowed and surfaced)

Codex's "don't approve by vibes" safety net is enforced in the prompt:
if Qwen cannot verify a PR description's claim due to missing context
(file not provided, dependency not visible, etc.), it must classify
that as a BLOCKER ("cannot verify claim X"), NOT silently PASS.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ----- Configuration / defaults -----

DEFAULT_API_BASE = "http://localhost:1234/v1"  # LM Studio
DEFAULT_MODEL = "qwen3-coder-next"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_HTTP_TIMEOUT = 600

# File-size budgets. Files over MAX_FILE_BYTES are referenced by name only;
# Qwen sees the diff hunks but not the full content. Files under that limit
# are included in full.
MAX_FILE_BYTES = 60_000          # ~1500 lines of typical code
MAX_FILES_PER_REVIEW = 25        # synthesis pass struggles past ~25 per-file findings
MAX_PER_FILE_PROMPT_TOKENS = 6000  # rough budget; we truncate content if hit


# ----- Data classes -----


@dataclass
class AuditConfig:
    github_token: str
    api_base: str = DEFAULT_API_BASE
    model: str = DEFAULT_MODEL
    api_key: str = DEFAULT_API_KEY
    http_timeout: int = DEFAULT_HTTP_TIMEOUT
    dry_run: bool = False
    max_file_bytes: int = MAX_FILE_BYTES
    max_files: int = MAX_FILES_PER_REVIEW


@dataclass
class FileFinding:
    """Per-file finding emitted by the per-file Qwen pass."""
    path: str
    status: str  # "added" | "modified" | "removed" | "renamed" | "skipped-binary" | "skipped-toobig"
    blockers: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    raw_response: str = ""  # for debugging / dry-run inspection

    def has_blocker(self) -> bool:
        return bool(self.blockers)


@dataclass
class AuditResult:
    repo: str
    pr_number: int
    head_sha_start: str
    head_sha_end: str
    file_findings: List[FileFinding]
    synthesis_response: str
    verdict: str        # "PASS" | "BLOCKED" | "STALE"
    trailer: str        # the full HTML-comment trailer line
    comment_body: str
    posted_comment_url: Optional[str] = None

    def head_changed_during_review(self) -> bool:
        return self.head_sha_start != self.head_sha_end


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
    """Single GitHub REST call. Returns parsed JSON, or text for diff Accept."""
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
        body_bytes = response.read()
        if accept.endswith("diff"):
            return body_bytes.decode("utf-8", errors="replace")
        text = body_bytes.decode("utf-8")
        return json.loads(text) if text else None


def fetch_pull_request(repo: str, pr_number: int, *, token: str) -> Dict[str, Any]:
    return _gh_request("GET", f"/repos/{repo}/pulls/{pr_number}", token=token)


def fetch_pr_files(repo: str, pr_number: int, *, token: str) -> List[Dict[str, Any]]:
    """Return list of per-file diff entries (status, filename, patch, etc.).
    GitHub paginates at 100; we collect up to 5 pages = 500 files."""
    all_files: List[Dict[str, Any]] = []
    for page in range(1, 6):
        chunk = _gh_request(
            "GET",
            f"/repos/{repo}/pulls/{pr_number}/files?per_page=100&page={page}",
            token=token,
        )
        if not chunk:
            break
        all_files.extend(chunk)
        if len(chunk) < 100:
            break
    return all_files


def fetch_file_content(repo: str, path: str, ref: str, *, token: str) -> Optional[bytes]:
    """Fetch raw file content at a given ref. Returns None if not found (e.g.
    file deleted in the PR), or if the path is a directory / symlink / submodule."""
    try:
        meta = _gh_request(
            "GET",
            f"/repos/{repo}/contents/{path}?ref={ref}",
            token=token,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if not isinstance(meta, dict):
        # Directory listing — we asked for a path that's actually a directory.
        return None
    if meta.get("type") != "file":
        return None
    encoding = meta.get("encoding", "")
    content = meta.get("content", "")
    if encoding == "base64":
        try:
            return base64.b64decode(content)
        except (ValueError, TypeError):
            return None
    # Large files: GitHub returns a download_url instead of inline content.
    download_url = meta.get("download_url")
    if download_url:
        try:
            with urllib.request.urlopen(download_url, timeout=30) as response:
                return response.read()
        except urllib.error.URLError:
            return None
    return None


def post_pr_comment(repo: str, pr_number: int, body: str, *, token: str) -> Dict[str, Any]:
    return _gh_request(
        "POST",
        f"/repos/{repo}/issues/{pr_number}/comments",
        token=token,
        body={"body": body},
    )


# ----- Qwen call -----


def call_qwen(config: AuditConfig, system: str, user: str, *, max_tokens: int = 2048) -> str:
    """Single Qwen chat-completion. Returns raw text."""
    req_body = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    data = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(
        f"{config.api_base.rstrip('/')}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config.http_timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


# ----- Prompts -----


PER_FILE_SYSTEM_PROMPT = """\
You are an automated code reviewer. Your job is to find BLOCKERS — issues
that must be fixed before merge — and surface non-blocking CONCERNS.

You will see ONE file at a time: its full current contents (at PR head)
plus the unified diff of what changed. You also see the PR's title and
body for context.

You MUST classify every observation into one of three severities:

    BLOCKER  — must fix before merge. Examples:
                 - correctness bug introduced by the change
                 - missing test for new behavior (when the PR claims tests)
                 - schema / contract break
                 - committed secret or credential
                 - claim in PR body cannot be verified from the visible code
                   (treat unverifiable claims as BLOCKERs — DO NOT silently
                   approve by vibes)
    CONCERN  — non-blocking observation (style, naming, refactor opportunity)
    NONE     — no issues to report

Output STRICTLY in this JSON format and NOTHING else (no prose around it,
no markdown fences):

    {"blockers": ["...", "..."], "concerns": ["...", "..."]}

Empty arrays are fine. Each entry is one specific finding, ideally with
file location ("line ~123: ...") and what the fix would be.
"""


SYNTHESIS_SYSTEM_PROMPT = """\
You are an automated code reviewer producing the FINAL verdict for a
pull request. The per-file pass already produced findings for each
changed file. Your job is two things:

  1. Surface CROSS-FILE / PR-LEVEL issues the per-file pass cannot see:
       - test coverage: does the changed behavior have tests that
         actually exercise the changed code paths?
       - doc-index consistency: did the PR add tools/fixtures that
         should be registered in README / index docs?
       - lane discipline: does the PR touch paths that are owned by
         a different lane (e.g., Codex-owned production code in a
         Claude PR) without coordination?
       - secrets / config: any `.env`, credentials, or tokens?
       - schema / regression-gate: if production geometry / recognizer
         behavior changed, does the PR include the row-level baseline
         diff (not just aggregate A/B)?

  2. Produce the FINAL human-readable comment.

Severity rules (same as per-file pass):
    BLOCKER → must fix before merge
    CONCERN → non-blocking
    NONE    → no issues

Final verdict logic:
    Any BLOCKER (from per-file findings OR your cross-cutting pass) → BLOCKED
    Else                                                            → PASS

Output STRICTLY in this format (no JSON, no markdown fences around the
whole output — but you may use markdown WITHIN the human-readable body):

Qwen Audit: <PASS or BLOCKED>

<one or two sentence headline>

<detailed body — markdown OK. List BLOCKERs first if any, then CONCERNs,
then "Cross-cutting notes" with anything from your PR-level pass.>

<!-- QWEN_AUDIT_STATE: qwen-audit-done -->
or
<!-- QWEN_AUDIT_STATE: qwen-audit-blocked -->

Pick exactly ONE trailer matching the verdict. The trailer MUST be the
final line of your output.

Important:
- If you cannot verify a claim made in the PR body, treat it as a
  BLOCKER ("cannot verify: ..."). Do NOT silently approve.
- If you only have CONCERNs (no BLOCKERs), the verdict is PASS, and
  the CONCERNs are listed in the body for the author to consider.
"""


def _safe_decode(content_bytes: bytes) -> Tuple[Optional[str], str]:
    """Return (decoded_text, reason) where reason is 'ok', 'binary', or 'too-big'."""
    if b"\x00" in content_bytes[:8192]:
        return None, "binary"
    try:
        return content_bytes.decode("utf-8"), "ok"
    except UnicodeDecodeError:
        return None, "binary"


def build_per_file_user_prompt(
    file_meta: Dict[str, Any],
    file_content: Optional[str],
    pr_meta: Dict[str, Any],
) -> str:
    """Compose the per-file Qwen prompt."""
    patch = file_meta.get("patch", "") or "(no patch — likely a binary or rename-only)"
    status = file_meta.get("status", "modified")
    path = file_meta.get("filename", "?")

    if file_content is None:
        content_block = "(file content not available — file may have been deleted,\nbe binary, exceed size limit, or be a directory/submodule)"
    elif len(file_content) > MAX_FILE_BYTES:
        content_block = (
            file_content[:MAX_FILE_BYTES]
            + f"\n\n... (truncated: file is {len(file_content)} bytes, showing first {MAX_FILE_BYTES})\n"
        )
    else:
        content_block = file_content

    pr_title = pr_meta.get("title", "")
    pr_body = (pr_meta.get("body") or "").strip()
    pr_body_short = pr_body[:2000] + ("\n... (truncated)\n" if len(pr_body) > 2000 else "")

    return textwrap.dedent(
        """\
        PR title: {title}
        PR body (truncated to 2000 chars):
        ---
        {body}
        ---

        File path: {path}
        Status: {status}

        ## Full file content at PR head SHA

        ```
        {content}
        ```

        ## Diff (unified)

        ```diff
        {patch}
        ```

        Review this single file. Output the JSON object only.
        """
    ).format(
        title=pr_title,
        body=pr_body_short or "(empty)",
        path=path,
        status=status,
        content=content_block,
        patch=patch,
    )


def build_synthesis_user_prompt(
    pr_meta: Dict[str, Any],
    findings: List[FileFinding],
) -> str:
    """Compose the synthesis Qwen prompt."""
    pr_title = pr_meta.get("title", "")
    pr_body = (pr_meta.get("body") or "").strip()
    pr_body_short = pr_body[:3000] + ("\n... (truncated)\n" if len(pr_body) > 3000 else "")
    repo = pr_meta.get("base", {}).get("repo", {}).get("full_name", "?")
    head_sha = pr_meta.get("head", {}).get("sha", "?")

    file_list_lines = []
    findings_lines = []
    for f in findings:
        file_list_lines.append(f"- `{f.path}` ({f.status})")
        if f.blockers or f.concerns:
            findings_lines.append(f"### {f.path}")
            for b in f.blockers:
                findings_lines.append(f"- BLOCKER: {b}")
            for c in f.concerns:
                findings_lines.append(f"- CONCERN: {c}")
            findings_lines.append("")
        elif f.status.startswith("skipped"):
            findings_lines.append(f"### {f.path}")
            findings_lines.append(f"- {f.status} (no review)")
            findings_lines.append("")

    files_block = "\n".join(file_list_lines) or "(none)"
    findings_block = "\n".join(findings_lines) or "(per-file pass found no issues)"

    return textwrap.dedent(
        """\
        Repository: {repo}
        PR #{pr_number}: {title}
        Head SHA: {head_sha}
        Author: {author}
        Files changed ({n_files}):
        {files}

        ## PR body (truncated to 3000 chars)

        {body}

        ## Per-file findings

        {findings}

        Now produce the FINAL audit comment per your system prompt. Remember:
        - Surface cross-file / PR-level issues the per-file pass cannot see.
        - If you cannot verify a claim in the PR body, that's a BLOCKER, not a PASS.
        - Final line MUST be the trailer:
            <!-- QWEN_AUDIT_STATE: qwen-audit-done -->   (PASS)
            <!-- QWEN_AUDIT_STATE: qwen-audit-blocked --> (BLOCKED)
        """
    ).format(
        repo=repo,
        pr_number=pr_meta.get("number", "?"),
        title=pr_title,
        head_sha=head_sha,
        author=pr_meta.get("user", {}).get("login", "?"),
        n_files=len(findings),
        files=files_block,
        body=pr_body_short or "(empty)",
        findings=findings_block,
    )


# ----- Response parsing -----


def parse_per_file_response(text: str) -> Tuple[List[str], List[str]]:
    """Parse the JSON {blockers: [...], concerns: [...]} from the per-file pass.
    Tolerant of extra prose / markdown fences."""
    text = text.strip()
    # Strip code fences if present.
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    # Find the outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return [], []
    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return [], []
    blockers = [str(x) for x in obj.get("blockers", []) if x]
    concerns = [str(x) for x in obj.get("concerns", []) if x]
    return blockers, concerns


STALE_TRAILER = "<!-- QWEN_AUDIT_STATE: needs-qwen-audit -->"
DONE_TRAILER = "<!-- QWEN_AUDIT_STATE: qwen-audit-done -->"
BLOCKED_TRAILER = "<!-- QWEN_AUDIT_STATE: qwen-audit-blocked -->"


def ensure_trailer(synthesis_text: str, verdict: str) -> str:
    """Ensure the synthesis response ends with the correct authoritative trailer.
    Strips any other QWEN_AUDIT_STATE trailers Qwen may have emitted and
    appends the canonical one for the verdict."""
    canonical = DONE_TRAILER if verdict == "PASS" else BLOCKED_TRAILER
    cleaned_lines = [
        line for line in synthesis_text.splitlines()
        if "QWEN_AUDIT_STATE" not in line
    ]
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    cleaned_lines.append("")
    cleaned_lines.append(canonical)
    return "\n".join(cleaned_lines) + "\n"


# ----- Orchestration -----


def _is_likely_binary_filename(path: str) -> bool:
    binary_exts = {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".ico",
        ".pdf", ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
        ".woff", ".woff2", ".ttf", ".otf", ".eot",
        ".mp3", ".mp4", ".mov", ".avi", ".wav", ".flac",
        ".pyc", ".so", ".dylib", ".dll", ".exe",
        ".onnx", ".pt", ".pth", ".bin", ".npy", ".npz",
    }
    lower = path.lower()
    return any(lower.endswith(ext) for ext in binary_exts)


def review_one_file(
    config: AuditConfig,
    repo: str,
    head_sha: str,
    file_meta: Dict[str, Any],
    pr_meta: Dict[str, Any],
    *,
    log: bool = True,
) -> FileFinding:
    path = file_meta.get("filename", "?")
    status = file_meta.get("status", "modified")

    if _is_likely_binary_filename(path):
        if log:
            print(f"  skip-binary {path}", file=sys.stderr, flush=True)
        return FileFinding(path=path, status="skipped-binary")

    # Removed files have no content at head; only patch.
    file_content: Optional[str] = None
    if status != "removed":
        raw = fetch_file_content(repo, path, head_sha, token=config.github_token)
        if raw is None:
            file_content = None
        else:
            text, reason = _safe_decode(raw)
            if reason == "binary":
                if log:
                    print(f"  skip-binary-content {path}", file=sys.stderr, flush=True)
                return FileFinding(path=path, status="skipped-binary")
            if len(raw) > config.max_file_bytes:
                if log:
                    print(f"  truncate-toobig {path} ({len(raw)}B)", file=sys.stderr, flush=True)
            file_content = text

    user_prompt = build_per_file_user_prompt(file_meta, file_content, pr_meta)

    if log:
        size_hint = len(file_content) if file_content else 0
        print(f"  review {path} (content={size_hint}ch)", file=sys.stderr, flush=True)

    try:
        response = call_qwen(config, PER_FILE_SYSTEM_PROMPT, user_prompt, max_tokens=1024)
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        if log:
            print(f"  qwen-error {path}: {exc}", file=sys.stderr, flush=True)
        return FileFinding(
            path=path,
            status=status,
            blockers=[f"Per-file Qwen call failed: {exc}. Cannot review this file."],
            raw_response="",
        )

    blockers, concerns = parse_per_file_response(response)
    return FileFinding(
        path=path,
        status=status,
        blockers=blockers,
        concerns=concerns,
        raw_response=response,
    )


def audit_pr(config: AuditConfig, repo: str, pr_number: int) -> AuditResult:
    """End-to-end audit of one PR. Returns AuditResult (does NOT post by itself
    unless caller wants — this function is pure orchestration plus optional
    posting controlled by config.dry_run).
    """
    pr_meta = fetch_pull_request(repo, pr_number, token=config.github_token)
    head_sha_start = pr_meta["head"]["sha"]

    files = fetch_pr_files(repo, pr_number, token=config.github_token)
    if len(files) > config.max_files:
        # Note the truncation but proceed with the first N — synthesis prompt
        # will flag this if relevant.
        files = files[: config.max_files]

    print(
        f"audit {repo}#{pr_number} head={head_sha_start[:8]} files={len(files)}",
        file=sys.stderr, flush=True,
    )

    findings: List[FileFinding] = []
    for file_meta in files:
        finding = review_one_file(config, repo, head_sha_start, file_meta, pr_meta)
        findings.append(finding)

    # Stale HEAD check before synthesis. If HEAD changed during the per-file
    # passes, requeue rather than produce a misleading verdict.
    pr_meta_after = fetch_pull_request(repo, pr_number, token=config.github_token)
    head_sha_end = pr_meta_after["head"]["sha"]

    if head_sha_start != head_sha_end:
        comment_body = _format_stale_comment(repo, pr_number, head_sha_start, head_sha_end)
        result = AuditResult(
            repo=repo,
            pr_number=pr_number,
            head_sha_start=head_sha_start,
            head_sha_end=head_sha_end,
            file_findings=findings,
            synthesis_response="",
            verdict="STALE",
            trailer=STALE_TRAILER,
            comment_body=comment_body,
        )
        if not config.dry_run:
            posted = post_pr_comment(repo, pr_number, comment_body, token=config.github_token)
            result.posted_comment_url = posted.get("html_url")
        return result

    # Synthesis pass.
    synth_user = build_synthesis_user_prompt(pr_meta, findings)
    try:
        synth_response = call_qwen(config, SYNTHESIS_SYSTEM_PROMPT, synth_user, max_tokens=3000)
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        synth_response = (
            f"Qwen Audit: BLOCKED\n\n"
            f"Synthesis pass failed: {exc}. The per-file pass produced the "
            f"findings above but the final synthesis could not complete. "
            f"Treating as BLOCKED out of caution.\n"
        )

    # Determine verdict from per-file findings + Qwen's synthesis self-claim.
    has_blocker = any(f.has_blocker() for f in findings)
    qwen_says_blocked = "Qwen Audit: BLOCKED" in synth_response
    verdict = "BLOCKED" if (has_blocker or qwen_says_blocked) else "PASS"

    full_body = ensure_trailer(synth_response, verdict)
    comment_body = _format_top_matter(repo, pr_number, head_sha_start, findings) + full_body

    result = AuditResult(
        repo=repo,
        pr_number=pr_number,
        head_sha_start=head_sha_start,
        head_sha_end=head_sha_end,
        file_findings=findings,
        synthesis_response=synth_response,
        verdict=verdict,
        trailer=DONE_TRAILER if verdict == "PASS" else BLOCKED_TRAILER,
        comment_body=comment_body,
    )

    if not config.dry_run:
        posted = post_pr_comment(repo, pr_number, comment_body, token=config.github_token)
        result.posted_comment_url = posted.get("html_url")

    return result


def _format_top_matter(repo: str, pr_number: int, head_sha: str, findings: List[FileFinding]) -> str:
    n_files = len(findings)
    n_blocker_files = sum(1 for f in findings if f.has_blocker())
    n_concern_files = sum(1 for f in findings if f.concerns and not f.has_blocker())
    n_clean = n_files - n_blocker_files - n_concern_files
    return (
        f"## Qwen audit (calibration phase — informational only)\n\n"
        f"Head SHA: `{head_sha}`\n"
        f"Files reviewed: {n_files} "
        f"(blocker findings: {n_blocker_files}, concern-only: {n_concern_files}, clean: {n_clean})\n\n"
    )


def _format_stale_comment(repo: str, pr_number: int, start_sha: str, end_sha: str) -> str:
    return (
        f"## Qwen audit (calibration phase — informational only)\n\n"
        f"Head SHA changed during review (`{start_sha[:8]}` → `{end_sha[:8]}`). "
        f"Skipping this verdict and requeuing for re-review of the new head.\n\n"
        f"{STALE_TRAILER}\n"
    )


# ----- CLI entry point -----


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Qwen audit CLI — review a single pull request.",
    )
    ap.add_argument("--repo", required=True, help="owner/repo, e.g. jeffhuber/cube-snap")
    ap.add_argument("--pr", type=int, required=True, help="PR number")
    ap.add_argument("--api-base", default=os.environ.get("QWEN_API_BASE", DEFAULT_API_BASE))
    ap.add_argument("--model", default=os.environ.get("QWEN_API_MODEL", DEFAULT_MODEL))
    ap.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY", DEFAULT_API_KEY))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=bool(os.environ.get("QWEN_AUDIT_DRY_RUN")),
        help="Print the audit comment to stdout instead of posting it.",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("error: GITHUB_TOKEN env var is required", file=sys.stderr)
        return 1

    config = AuditConfig(
        github_token=token,
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
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

    if args.dry_run:
        print(result.comment_body)
    else:
        print(
            f"posted {args.repo}#{args.pr} verdict={result.verdict} "
            f"url={result.posted_comment_url}",
            file=sys.stderr,
        )

    if result.verdict == "STALE":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
