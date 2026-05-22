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
  3. For each new PR, delegates the actual review work to the CLI
     module `tools/qwen_audit_pr.py` via `audit_pr()`. The CLI handles
     per-file chunking, full-file-content context, synthesis, and the
     stale-HEAD check — see that module for details.
  4. The CLI posts the comment with the authoritative trailer line:
        <!-- QWEN_AUDIT_STATE: qwen-audit-done -->
        <!-- QWEN_AUDIT_STATE: qwen-audit-blocked -->
        <!-- QWEN_AUDIT_STATE: needs-qwen-audit -->     (stale head)
  5. The `qwen-audit-labeler.yml` workflow fires on the new comment,
     parses the trailer, and applies the appropriate label.

Configuration (environment variables):

    GITHUB_TOKEN          — PAT with `repo` scope for the bot account.
                            Comments are posted under this account's
                            identity, so it must match QWEN_BOT_AUTHORS
                            in qwen_audit_labeler.py.
    QWEN_AUDIT_REPOS      — comma-separated `owner/repo` list. Default:
                            jeffhuber/cube-snap,jeffhuber/cube-two-view-debugger
    QWEN_API_BASE         — OpenAI-compatible base URL. Default:
                            http://localhost:1234/v1 (LM Studio).
                            For ollama use http://localhost:11434/v1.
    QWEN_API_MODEL        — model name. Default: qwen3-coder-next.
                            For ollama use the exact local tag, e.g.
                            "hf.co/prism-ml/Bonsai-8B-gguf:latest".
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

To review a single PR on demand (bypass the daemon entirely):
    GITHUB_TOKEN=... python3 tools/qwen_audit_pr.py \
        --repo jeffhuber/cube-snap --pr 142

The bridge and the CLI share the same review logic — the bridge is just
the polling / label-state-machine scheduler on top.

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
from typing import Any, Dict, List, Optional

# The review work lives in qwen_audit_pr; this file is just the daemon.
# Import the CLI module so the bridge inherits any improvements made
# there without code duplication.
try:
    from tools import qwen_audit_pr as _audit_cli  # when imported as package
except ImportError:  # pragma: no cover — top-level invocation fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import qwen_audit_pr as _audit_cli


DEFAULT_REPOS = "jeffhuber/cube-snap,jeffhuber/cube-two-view-debugger"
DEFAULT_API_BASE = _audit_cli.DEFAULT_API_BASE
DEFAULT_MODEL = _audit_cli.DEFAULT_MODEL
DEFAULT_POLL_INTERVAL = 60
DEFAULT_STATE_PATH = Path.home() / ".config" / "qwen-audit-bridge" / "state.json"


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

    def to_audit_config(self) -> _audit_cli.AuditConfig:
        return _audit_cli.AuditConfig(
            github_token=self.github_token,
            api_base=self.api_base,
            model=self.model,
            api_key=self.api_key,
            dry_run=self.dry_run,
        )


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


# ----- GitHub helpers (label query only — heavy lifting is in qwen_audit_pr) -----


def list_open_prs_with_label(repo: str, label: str, *, token: str) -> List[Dict[str, Any]]:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues?state=open&labels={label}&per_page=50",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


# ----- Main loop -----


def process_pr(
    config: BridgeConfig,
    state: State,
    repo: str,
    pr_summary: Dict[str, Any],
) -> str:
    pr_number = pr_summary["number"]
    # Quick HEAD fetch via the CLI's helper, just to check dedupe.
    pr = _audit_cli.fetch_pull_request(repo, pr_number, token=config.github_token)
    head_sha = pr["head"]["sha"]

    if state.already_reviewed(repo, pr_number, head_sha):
        return f"skip {repo}#{pr_number}: head {head_sha[:8]} already audited"

    if config.dry_run:
        print(f"[dry-run] would audit {repo}#{pr_number} head={head_sha[:8]}",
              file=sys.stderr, flush=True)
        state.record(repo, pr_number, head_sha)
        state.save(config.state_path)
        return f"dry-run {repo}#{pr_number}"

    # Delegate to the CLI module for the actual review.
    try:
        result = _audit_cli.audit_pr(config.to_audit_config(), repo, pr_number)
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"  audit_pr failed for {repo}#{pr_number}: {exc}",
              file=sys.stderr, flush=True)
        return f"error {repo}#{pr_number}: {exc}"

    # Only record the head SHA if the review completed (not STALE — caller
    # will retry once the new HEAD picks up the trailer & gets re-labeled).
    if result.verdict != "STALE":
        state.record(repo, pr_number, head_sha)
        state.save(config.state_path)

    return (
        f"posted {repo}#{pr_number} verdict={result.verdict} "
        f"head={head_sha[:8]} url={result.posted_comment_url}"
    )


def poll_once(config: BridgeConfig, state: State) -> List[str]:
    results = []
    for repo in config.repos:
        try:
            prs = list_open_prs_with_label(repo, "needs-qwen-audit", token=config.github_token)
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            results.append(f"list-error {repo}: {exc}")
            continue
        for pr in prs:
            if "pull_request" not in pr:  # issue, not a PR
                continue
            try:
                results.append(process_pr(config, state, repo, pr))
            except (urllib.error.URLError, RuntimeError) as exc:
                results.append(f"error {repo}#{pr.get('number')}: {exc}")
    return results


def daemon_loop(config: BridgeConfig) -> None:
    state = State.load(config.state_path)
    print(
        f"qwen audit bridge starting: repos={config.repos}, "
        f"interval={config.poll_interval}s, state={config.state_path}, "
        f"api={config.api_base} model={config.model}",
        file=sys.stderr, flush=True,
    )
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
