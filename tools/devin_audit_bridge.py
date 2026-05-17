#!/usr/bin/env python3
"""Dispatch Devin PR audit requests from GitHub Actions event payloads.

This script is intentionally metadata-only. The workflow that invokes it should
check out the default branch copy of this script, not pull request code.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


NEEDS_LABEL = "needs-devin-audit"
DONE_LABEL = "devin-audit-done"
BLOCKED_LABEL = "devin-audit-blocked"
TRIGGER_PHRASE = "@devin audit"
FORCE_TRIGGER_PHRASE = "@devin audit force"
IGNORED_ACTORS = {"devin-ai-integration[bot]", "vercel[bot]"}
TRUSTED_COMMENTER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
PULL_REQUEST_ACTIONS = {"opened", "synchronize", "reopened", "labeled"}
DEVIN_COMMENT_AUTHORS = {"devin-ai-integration", "devin-ai-integration[bot]"}

DEVIN_INSTRUCTIONS = textwrap.dedent(
    f"""\
    Review the GitHub PR from this payload.

    Scope:
    - Only work in the affected repo.
    - Active repos are jeffhuber/cube-two-view-debugger and jeffhuber/cube-snap.
    - Inspect the PR description, diff, current head SHA, labels, and existing comments.
    - Run focused checks appropriate for the files changed.
    - Do not merge, close, force-push, or modify code.
    - Ignore comments/events authored by devin-ai-integration[bot] or vercel[bot].

    Required label tools:
    - Use Devin's built-in GitHub PR label tools for label changes, not shell gh,
      curl, or direct calls to api.github.com.
    - Use git_add_labels to add {DONE_LABEL} or {BLOCKED_LABEL}.
    - Use git_remove_labels to remove {NEEDS_LABEL}.
    - Use git_view_pr to re-read and verify final labels.
    - If those built-in label tools are unavailable or fail, include
      LABEL_UPDATE_FAILED: <reason> in the final PR comment.

    SHA handling:
    - Record the PR head SHA before reviewing.
    - Include the reviewed SHA in the audit result, e.g. Head SHA: <sha>.
    - Before changing labels, re-read the PR head SHA.
    - If the head SHA changed during review, do not mark done/blocked. Comment
      HEAD_CHANGED_DURING_REVIEW: reviewed <old>, current <new> and leave/re-add
      {NEEDS_LABEL}.

    Pass path:
    - Re-read the PR head SHA and confirm it still matches the reviewed SHA.
    - Add label {DONE_LABEL} using git_add_labels.
    - Remove label {NEEDS_LABEL} using git_remove_labels.
    - Re-read the PR using git_view_pr and verify final labels include
      {DONE_LABEL} and do not include {NEEDS_LABEL}.
    - Post one final audit-pass PR comment with Head SHA: <sha>, checks run, and
      Label state: {DONE_LABEL}.
    - If label updates or verification fail, post the final audit comment with
      LABEL_UPDATE_FAILED: <reason> and the observed labels.

    Blocked path:
    - Re-read the PR head SHA and confirm it still matches the reviewed SHA.
    - Add label {BLOCKED_LABEL} using git_add_labels.
    - Remove label {NEEDS_LABEL} using git_remove_labels.
    - Re-read the PR using git_view_pr and verify final labels include
      {BLOCKED_LABEL} and do not include {NEEDS_LABEL}.
    - Post one final blocker/incomplete PR comment with Head SHA: <sha>,
      blocker summary or incomplete reason, and Label state: {BLOCKED_LABEL}.
    - If label updates or verification fail, post the final blocker/incomplete
      comment with LABEL_UPDATE_FAILED: <reason> and the observed labels.
    """
).strip()


@dataclass(frozen=True)
class AuditRequest:
    pull_request: Dict[str, Any]
    trigger: Dict[str, str]
    force: bool = False


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pr_labels(pr: Dict[str, Any]) -> set[str]:
    return {label.get("name", "") for label in pr.get("labels", [])}


def has_trigger_phrase(text: str) -> bool:
    return TRIGGER_PHRASE in text.lower()


def has_force_trigger_phrase(text: str) -> bool:
    return FORCE_TRIGGER_PHRASE in text.lower()


def resolve_audit_request(
    *,
    event_name: str,
    action: str,
    actor: str,
    event: Dict[str, Any],
    fetch_pull_request: Callable[[int], Dict[str, Any]],
) -> Tuple[Optional[AuditRequest], str]:
    if actor in IGNORED_ACTORS:
        return None, f"ignored actor: {actor}"

    if event_name == "pull_request_target":
        if action not in PULL_REQUEST_ACTIONS:
            return None, f"unsupported pull_request_target action: {action}"
        pr = event.get("pull_request") or {}
        if pr.get("state") != "open":
            return None, "pull request is not open"
        if action == "labeled":
            label_name = (event.get("label") or {}).get("name")
            if label_name != NEEDS_LABEL:
                return None, f"ignored label: {label_name}"
        elif NEEDS_LABEL not in pr_labels(pr):
            return None, f"{NEEDS_LABEL} label is absent"
        return AuditRequest(
            pull_request=pr,
            trigger={"event": event_name, "action": action, "actor": actor},
        ), "dispatch"

    if event_name == "issue_comment":
        if action != "created":
            return None, f"unsupported issue_comment action: {action}"
        issue = event.get("issue") or {}
        comment = event.get("comment") or {}
        if not issue.get("pull_request"):
            return None, "comment is not on a pull request"
        if not has_trigger_phrase(comment.get("body") or ""):
            return None, f"missing trigger phrase: {TRIGGER_PHRASE}"
        association = comment.get("author_association")
        if association not in TRUSTED_COMMENTER_ASSOCIATIONS:
            return None, f"untrusted commenter association: {association}"
        pr = fetch_pull_request(int(issue["number"]))
        if pr.get("state") != "open":
            return None, "pull request is not open"
        force = has_force_trigger_phrase(comment.get("body") or "")
        return AuditRequest(
            pull_request=pr,
            trigger={"event": event_name, "action": action, "actor": actor},
            force=force,
        ), "dispatch"

    return None, f"unsupported event: {event_name}"


def build_payload(repository: str, request: AuditRequest) -> Dict[str, Any]:
    pr = request.pull_request
    head = pr.get("head") or {}
    base = pr.get("base") or {}
    head_sha = head["sha"]
    number = int(pr["number"])
    labels = sorted(pr_labels(pr))
    return {
        "source": "github-actions-devin-audit-bridge",
        "dedupe_key": f"{repository}#{number}@{head_sha}",
        "trigger": request.trigger,
        "repository": repository,
        "pull_request": {
            "number": number,
            "url": pr.get("html_url"),
            "title": pr.get("title"),
            "state": pr.get("state"),
            "head_sha": head_sha,
            "head_ref": head.get("ref"),
            "head_repo": (head.get("repo") or {}).get("full_name"),
            "base_ref": base.get("ref"),
            "labels": labels,
        },
        "instructions": DEVIN_INSTRUCTIONS,
    }


def comment_author_login(comment: Dict[str, Any]) -> str:
    user = comment.get("user") or comment.get("author") or {}
    login = user.get("login")
    return login if isinstance(login, str) else ""


def comment_body(comment: Dict[str, Any]) -> str:
    body = comment.get("body")
    return body if isinstance(body, str) else ""


def is_devin_comment(comment: Dict[str, Any]) -> bool:
    return comment_author_login(comment) in DEVIN_COMMENT_AUTHORS


def devin_already_reviewed_sha(head_sha: str, comments: Iterable[Dict[str, Any]]) -> bool:
    return any(is_devin_comment(comment) and head_sha in comment_body(comment) for comment in comments)


def github_api_json(path: str, token: str) -> Dict[str, Any]:
    request = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def github_api_paginated(path: str, token: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    page = 1
    while True:
        separator = "&" if "?" in path else "?"
        page_path = f"{path}{separator}per_page=100&page={page}"
        batch = github_api_json(page_path, token)
        if not isinstance(batch, list):
            raise RuntimeError(f"expected list response from GitHub API path {page_path}")
        items.extend(batch)
        if len(batch) < 100:
            return items
        page += 1


def post_webhook(url: str, secret: str, payload: Dict[str, Any]) -> Tuple[int, str]:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "github-actions-devin-audit-bridge",
            "x-webhook-secret": secret,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status, body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not configured")
    return value


def main() -> int:
    return run()


def run() -> int:
    event_path = Path(require_env("GITHUB_EVENT_PATH"))
    event_name = require_env("GITHUB_EVENT_NAME")
    action = os.environ.get("GITHUB_EVENT_ACTION", "")
    repository = require_env("GITHUB_REPOSITORY")
    actor = require_env("GITHUB_ACTOR")
    event = load_json(event_path)

    token: Optional[str] = None

    def github_token() -> str:
        nonlocal token
        if token is None:
            token = require_env("GITHUB_TOKEN")
        return token

    def fetch_pull_request(number: int) -> Dict[str, Any]:
        return github_api_json(f"/repos/{repository}/pulls/{number}", github_token())

    audit_request, reason = resolve_audit_request(
        event_name=event_name,
        action=action,
        actor=actor,
        event=event,
        fetch_pull_request=fetch_pull_request,
    )
    if audit_request is None:
        print(f"skip: {reason}")
        return 0

    webhook_url = require_env("DEVIN_WEBHOOK_URL")
    webhook_secret = require_env("DEVIN_WEBHOOK_SECRET")
    payload = build_payload(repository, audit_request)
    if os.environ.get("DRY_RUN") == "1":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    pr_number = payload["pull_request"]["number"]
    head_sha = payload["pull_request"]["head_sha"]
    if not audit_request.force:
        comments = github_api_paginated(f"/repos/{repository}/issues/{pr_number}/comments", github_token())
        reviews = github_api_paginated(f"/repos/{repository}/pulls/{pr_number}/reviews", github_token())
        if devin_already_reviewed_sha(head_sha, [*comments, *reviews]):
            print(f"skip: Devin already reviewed head SHA {head_sha}")
            return 0
    status, body = post_webhook(webhook_url, webhook_secret, payload)
    print(f"Devin webhook response: HTTP {status}")
    if body:
        print("--- response body ---")
        print(body)
        print("--- end response body ---")
    if not 200 <= status < 300:
        print(f"error: Devin webhook returned non-2xx status {status}", file=sys.stderr)
        return 1
    print(f"dispatched: {payload['dedupe_key']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
