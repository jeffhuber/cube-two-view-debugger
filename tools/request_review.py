#!/usr/bin/env python3
"""Request a peer review without shell-interpreting Markdown.

This helper bundles the three review-request steps:

1. Post a generated PR comment through ``safe_gh_comment.py``'s JSON-backed
   GitHub API path.
2. Add the routing label, e.g. ``needs-claude-review``.
3. Append a ``review_requested`` event to the shared local audit log.

The comment body is generated from structured arguments inside Python. Do not
hand-roll Markdown bodies in shell heredocs for review requests.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from typing import Callable, List, Optional, Sequence

try:
    from tools import audit_handoff_log, safe_gh_comment
except ImportError:  # pragma: no cover - direct script execution fallback
    import audit_handoff_log  # type: ignore
    import safe_gh_comment  # type: ignore


RunFn = Callable[..., subprocess.CompletedProcess[str]]
PostCommentFn = Callable[[str, int, str], None]


def _actor_default() -> str:
    return os.environ.get("AUDIT_ACTOR") or os.environ.get("USER") or "unknown"


def _default_reviewer(lane: str) -> str:
    if lane == "claude-review":
        return "Claude"
    if lane == "codex-review":
        return "Codex"
    return "Peer"


def build_body(
    *,
    reviewer: str,
    head: str,
    scope: str,
    validations: Sequence[str],
    notes: Sequence[str],
) -> str:
    lines: List[str] = [
        f"{reviewer} review requested for current head `{head}` at {audit_handoff_log.now_record()['pt']}.",
        "",
        f"Scope: {scope}",
        "",
        "Expected output: PASS or blockers on the current head.",
    ]
    if validations:
        lines.extend(["", "Validation run:"])
        lines.extend(["", *[f"- `{item}`" for item in validations]])
    if notes:
        lines.extend(["", "Notes:"])
        lines.extend(["", *[f"- {item}" for item in notes]])
    return "\n".join(lines) + "\n"


def request_review(
    *,
    lane: str,
    repo: str,
    pr: int,
    head: str,
    label: str,
    reviewer: str,
    scope: str,
    validations: Sequence[str],
    notes: Sequence[str],
    actor: str,
    run: RunFn = subprocess.run,
    post_comment: PostCommentFn = safe_gh_comment.post_comment,
) -> None:
    body = build_body(
        reviewer=reviewer,
        head=head,
        scope=scope,
        validations=validations,
        notes=notes,
    )
    post_comment(repo, pr, body)
    run(
        ["gh", "issue", "edit", str(pr), "--repo", repo, "--add-label", label],
        check=True,
        text=True,
        capture_output=True,
    )
    audit_handoff_log.append_event(
        {
            "event": "review_requested",
            "lane": lane,
            "repo": repo,
            "pr": pr,
            "head": head,
            "actor": actor,
            "notes": f"{reviewer} review requested: {scope}",
        }
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, help="Review lane, e.g. claude-review or codex-review.")
    parser.add_argument("--repo", required=True, help="GitHub owner/repo.")
    parser.add_argument("--pr", required=True, type=int, help="Pull request number.")
    parser.add_argument("--head", required=True, help="Current PR head SHA.")
    parser.add_argument("--label", required=True, help="Routing label to add.")
    parser.add_argument("--reviewer", help="Reviewer display name. Defaults from --lane.")
    parser.add_argument("--scope", required=True, help="Short review scope.")
    parser.add_argument(
        "--validation",
        action="append",
        default=[],
        help="Validation command/result to include. Repeatable.",
    )
    parser.add_argument("--note", action="append", default=[], help="Extra note bullet. Repeatable.")
    parser.add_argument("--actor", default=_actor_default(), help="Actor value for the shared audit log.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        request_review(
            lane=args.lane,
            repo=args.repo,
            pr=args.pr,
            head=args.head,
            label=args.label,
            reviewer=args.reviewer or _default_reviewer(args.lane),
            scope=args.scope,
            validations=args.validation,
            notes=args.note,
            actor=args.actor,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
