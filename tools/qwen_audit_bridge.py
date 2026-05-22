#!/usr/bin/env python3
"""Qwen audit bridge — local polling daemon.

UNLIKE the Devin bridge (which runs as a GitHub Action and calls Devin's
cloud webhook), the Qwen bridge runs as a local daemon on the user's
machine because the Qwen3-Coder-Next model is served locally. The daemon:

  1. Polls GitHub every POLL_INTERVAL seconds for PRs labeled
     `needs-qwen-audit` (across one or more repos).
  2. Dedupes by `(repo, pr_number, head_sha)` so the same head SHA
     never re-triggers an audit. State persists in QWEN_AUDIT_STATE_PATH
     across daemon restarts.
  3. Fetches the PR diff + PR metadata.
  4. Calls the local Qwen serving endpoint (OpenAI-compatible by
     default) with the audit prompt.
  5. Posts the Qwen response as a PR comment that ends with the
     authoritative trailer line:
        <!-- QWEN_AUDIT_STATE: qwen-audit-done -->
        <!-- QWEN_AUDIT_STATE: qwen-audit-blocked -->
  6. The `qwen-audit-labeler.yml` workflow fires on the new comment,
     parses the trailer, and applies the appropriate label.

Configuration (environment variables):

    GITHUB_TOKEN          — PAT with `repo` scope for the bot account.
                            Comments are posted under this account's
                            identity, so it must match QWEN_BOT_AUTHORS
                            in qwen_audit_labeler.py.
    QWEN_AUDIT_REPOS      — comma-separated `owner/repo` list. Default:
                            jeffhuber/cube-snap,jeffhuber/cube-two-view-debugger
    QWEN_API_BASE         — OpenAI-compatible base URL. Default:
                            http://localhost:8000/v1
    QWEN_API_MODEL        — model name. Default: qwen3-coder-next
    QWEN_API_KEY          — bearer for the local server (most local
                            servers accept "EMPTY" or any string).
                            Default: EMPTY
    QWEN_POLL_INTERVAL    — seconds between polls. Default: 60.
    QWEN_AUDIT_STATE_PATH — local JSON file with seen head SHAs per PR.
                            Default: ~/.config/qwen-audit-bridge/state.json
    QWEN_AUDIT_DRY_RUN    — if set, log what WOULD be posted but don't
                            actually call the GitHub API.

Run manually for a single pass (no loop) with `--once`. Useful for
debugging and for one-shot manual triggers.

Run as a daemon with systemd-style supervision or just inside tmux
on the user's primary machine.

Calibration phase: this bridge is INFORMATIONAL ONLY. It runs in
parallel with Devin. Claude's standing in-thread merge delegation
authorizes merge on `devin-audit-done` + CLEAN, NOT on
`qwen-audit-done`. After ~10-20 PRs of overlap data, the user can
promote Qwen to a merge-authority label if it has demonstrated
parity with Devin's catches. See `tools/QWEN_AUDIT_PROTOCOL.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


DEFAULT_REPOS = "jeffhuber/cube-snap,jeffhuber/cube-two-view-debugger"
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_MODEL = "qwen3-coder-next"
DEFAULT_POLL_INTERVAL = 60
DEFAULT_STATE_PATH = Path.home() / ".config" / "qwen-audit-bridge" / "state.json"
DEFAULT_TIMEOUT = 600  # seconds for Qwen API call (large diffs take a while)


@dataclass
class BridgeConfig:
    github_token: str
    repos: List[str]
    api_base: str
    model: str
    api_key: str
    state_path: Path
    poll_interval: int = DEFAULT_POLL_INTERVAL
    dry_run: bool = False
    max_diff_chars: int = 200_000  # truncate beyond this; logs a warning


@dataclass
class State:
    """Per-(repo, pr_number) head SHA that was most-recently audited.
    Skip if (repo, pr_number, current_head) matches stored entry."""
    seen: Dict[str, str] = field(default_factory=dict)  # "{repo}#{pr}" -> head_sha

    @classmethod
    def load(cls, path: Path) -> "State":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(seen=data.get("seen", {}))
        except (json.JSONDecodeError, OSError):
            return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"seen": self.seen}, indent=2))

    def already_reviewed(self, repo: str, pr_number: int, head_sha: str) -> bool:
        return self.seen.get(f"{repo}#{pr_number}") == head_sha

    def record(self, repo: str, pr_number: int, head_sha: str) -> None:
        self.seen[f"{repo}#{pr_number}"] = head_sha


# ----- GitHub helpers -----

def gh_request(
    method: str,
    path: str,
    *,
    token: str,
    body: Optional[Dict[str, Any]] = None,
    accept: str = "application/vnd.github+json",
) -> Any:
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
    with urllib.request.urlopen(req, timeout=30) as response:
        body_bytes = response.read()
        # When the Accept header asks for diff, content-type is text/plain
        if accept.endswith("diff"):
            return body_bytes.decode("utf-8", errors="replace")
        body_text = body_bytes.decode("utf-8")
        return json.loads(body_text) if body_text else None


def list_open_prs_with_label(repo: str, label: str, *, token: str) -> List[Dict[str, Any]]:
    return gh_request(
        "GET",
        f"/repos/{repo}/issues?state=open&labels={label}&per_page=50",
        token=token,
    )


def fetch_pull_request(repo: str, pr_number: int, *, token: str) -> Dict[str, Any]:
    return gh_request("GET", f"/repos/{repo}/pulls/{pr_number}", token=token)


def fetch_pr_diff(repo: str, pr_number: int, *, token: str) -> str:
    return gh_request(
        "GET",
        f"/repos/{repo}/pulls/{pr_number}",
        token=token,
        accept="application/vnd.github.v3.diff",
    )


def post_comment(repo: str, pr_number: int, body: str, *, token: str) -> Dict[str, Any]:
    return gh_request(
        "POST",
        f"/repos/{repo}/issues/{pr_number}/comments",
        token=token,
        body={"body": body},
    )


# ----- Qwen call -----

def call_qwen(
    config: BridgeConfig,
    diff_text: str,
    pr_meta: Dict[str, Any],
) -> str:
    """Call the local Qwen serving endpoint. Returns the raw model text."""
    prompt = build_audit_prompt(diff_text, pr_meta)
    req_body = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": "You are an automated code reviewer."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.1,
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
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


# ----- Audit prompt -----

AUDIT_PROMPT_TEMPLATE = """\
You are an automated code reviewer for the {repo} repository, running in
parallel to Devin during a calibration period. Your verdict is currently
INFORMATIONAL — it does NOT yet authorize a merge.

Repository: {repo}
PR #{pr_number}: {title}
Head SHA: {head_sha}
Author: {author}
Base: {base_ref}
Files changed: {files_changed}
Additions: +{additions}  Deletions: -{deletions}

## Review criteria

1. **Correctness** — do the changes do what the PR description claims?
2. **Schema / contract preservation** — JSON fixtures match documented
   schema? Tool interfaces stay backward-compatible?
3. **Test coverage** — new behavior has tests; tests would actually
   exercise the changed code paths.
4. **Geometry regression-gate** (if PR touches `tools/global_cube_model.py`
   or production geometry behavior): PR body must include the row-level
   diff from `tools/baseline_post_218.py --diff`. Aggregate-only A/B is
   insufficient.
5. **Doc-index consistency** — new tools/fixtures/reports registered in
   `tools/README.md`, `tools/BENCHMARK_INDEX.md`, `tools/STATE_OF_THE_WORLD.md`
   when applicable.
6. **Lane discipline** — Claude-owned PRs should not touch `rubik_recognizer/*`
   or other Codex-owned production paths without coordination.
7. **No accidental secrets** — no `.env`, credentials, or tokens.

## Output format (REQUIRED)

Lead with a clear PASS or BLOCKED verdict on its own line:
    Qwen Audit: PASS
    or
    Qwen Audit: BLOCKED

Then provide your detailed review. If BLOCKED, list specific issues
with file paths and line numbers where possible.

End your response with EXACTLY ONE of these trailer lines on its own
final line (the labeler treats these as authoritative):
    <!-- QWEN_AUDIT_STATE: qwen-audit-done -->     (verdict was PASS)
    <!-- QWEN_AUDIT_STATE: qwen-audit-blocked -->  (verdict was BLOCKED)
    <!-- QWEN_AUDIT_STATE: needs-qwen-audit -->    (only if head SHA
                                                    changed during review)

## PR diff (truncated to {diff_chars} chars if larger):

```diff
{diff}
```
"""


def build_audit_prompt(diff_text: str, pr_meta: Dict[str, Any]) -> str:
    return AUDIT_PROMPT_TEMPLATE.format(
        repo=pr_meta.get("base", {}).get("repo", {}).get("full_name", "?"),
        pr_number=pr_meta.get("number"),
        title=pr_meta.get("title", ""),
        head_sha=pr_meta.get("head", {}).get("sha", ""),
        author=pr_meta.get("user", {}).get("login", "?"),
        base_ref=pr_meta.get("base", {}).get("ref", "?"),
        files_changed=pr_meta.get("changed_files", "?"),
        additions=pr_meta.get("additions", "?"),
        deletions=pr_meta.get("deletions", "?"),
        diff_chars=len(diff_text),
        diff=diff_text,
    )


def format_comment(qwen_response: str, pr_meta: Dict[str, Any]) -> str:
    head_sha = pr_meta.get("head", {}).get("sha", "")
    return (
        f"## Qwen audit (calibration phase — informational only)\n\n"
        f"Head SHA: `{head_sha}`\n\n"
        f"{qwen_response}\n"
    )


# ----- Main loop -----

def process_pr(config: BridgeConfig, state: State, repo: str, pr_summary: Dict[str, Any]) -> str:
    pr_number = pr_summary["number"]
    pr = fetch_pull_request(repo, pr_number, token=config.github_token)
    head_sha = pr["head"]["sha"]

    if state.already_reviewed(repo, pr_number, head_sha):
        return f"skip {repo}#{pr_number}: head {head_sha[:8]} already audited"

    diff = fetch_pr_diff(repo, pr_number, token=config.github_token)
    truncated = False
    if len(diff) > config.max_diff_chars:
        diff = diff[: config.max_diff_chars] + "\n... (truncated)\n"
        truncated = True

    if config.dry_run:
        print(f"[dry-run] would audit {repo}#{pr_number} head={head_sha[:8]} "
              f"diff={len(diff)}ch truncated={truncated}", file=sys.stderr)
        state.record(repo, pr_number, head_sha)
        state.save(config.state_path)
        return f"dry-run {repo}#{pr_number}"

    print(f"audit {repo}#{pr_number} head={head_sha[:8]} diff={len(diff)}ch", file=sys.stderr, flush=True)
    try:
        qwen_response = call_qwen(config, diff, pr)
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        print(f"  qwen call failed: {exc}", file=sys.stderr)
        return f"error {repo}#{pr_number}: {exc}"

    comment = format_comment(qwen_response, pr)
    post_comment(repo, pr_number, comment, token=config.github_token)
    state.record(repo, pr_number, head_sha)
    state.save(config.state_path)
    return f"posted {repo}#{pr_number}"


def poll_once(config: BridgeConfig, state: State) -> List[str]:
    results = []
    for repo in config.repos:
        try:
            prs = list_open_prs_with_label(repo, "needs-qwen-audit", token=config.github_token)
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            results.append(f"list-error {repo}: {exc}")
            continue
        for pr in prs:
            if "pull_request" not in pr:  # issue not a PR
                continue
            try:
                results.append(process_pr(config, state, repo, pr))
            except (urllib.error.URLError, RuntimeError) as exc:
                results.append(f"error {repo}#{pr.get('number')}: {exc}")
    return results


def daemon_loop(config: BridgeConfig) -> None:
    state = State.load(config.state_path)
    print(f"qwen audit bridge starting: repos={config.repos}, "
          f"interval={config.poll_interval}s, state={config.state_path}",
          file=sys.stderr, flush=True)
    while True:
        results = poll_once(config, state)
        for r in results:
            print(f"  {r}", file=sys.stderr, flush=True)
        if not results:
            print(f"  (no PRs needing audit)", file=sys.stderr, flush=True)
        time.sleep(config.poll_interval)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true",
                    help="Run a single poll iteration and exit (no daemon loop). "
                         "Useful for testing and cron-driven invocation.")
    ap.add_argument("--repos", default=os.environ.get("QWEN_AUDIT_REPOS", DEFAULT_REPOS),
                    help=f"Comma-separated owner/repo list. Default: {DEFAULT_REPOS}")
    ap.add_argument("--api-base", default=os.environ.get("QWEN_API_BASE", DEFAULT_API_BASE))
    ap.add_argument("--model", default=os.environ.get("QWEN_API_MODEL", DEFAULT_MODEL))
    ap.add_argument("--api-key", default=os.environ.get("QWEN_API_KEY", "EMPTY"))
    ap.add_argument("--poll-interval", type=int,
                    default=int(os.environ.get("QWEN_POLL_INTERVAL", DEFAULT_POLL_INTERVAL)))
    ap.add_argument("--state-path", default=os.environ.get("QWEN_AUDIT_STATE_PATH", str(DEFAULT_STATE_PATH)))
    ap.add_argument("--dry-run", action="store_true",
                    default=bool(os.environ.get("QWEN_AUDIT_DRY_RUN")))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("error: GITHUB_TOKEN env var is required", file=sys.stderr)
        return 1

    config = BridgeConfig(
        github_token=token,
        repos=[r.strip() for r in args.repos.split(",") if r.strip()],
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
        state_path=Path(args.state_path).expanduser(),
        poll_interval=args.poll_interval,
        dry_run=args.dry_run,
    )

    if args.once:
        state = State.load(config.state_path)
        for r in poll_once(config, state):
            print(r)
        return 0

    daemon_loop(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
