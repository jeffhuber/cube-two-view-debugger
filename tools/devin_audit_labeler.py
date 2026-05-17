#!/usr/bin/env python3
"""Apply Devin audit result labels from Devin-authored PR comments."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote


NEEDS_LABEL = "needs-devin-audit"
DONE_LABEL = "devin-audit-done"
BLOCKED_LABEL = "devin-audit-blocked"
DEVIN_COMMENT_AUTHORS = {"devin-ai-integration", "devin-ai-integration[bot]"}


@dataclass(frozen=True)
class LabelDecision:
    issue_number: int
    add_label: str
    remove_labels: tuple[str, ...]
    reviewed_sha: Optional[str]
    reason: str


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_devin_comment_author(login: str) -> bool:
    return login in DEVIN_COMMENT_AUTHORS


def extract_reviewed_sha(body: str) -> Optional[str]:
    patterns = [
        r"Head SHA:\*\*\s*`?([0-9a-fA-F]{7,40})`?",
        r"Head SHA:\s*`?([0-9a-fA-F]{7,40})`?",
    ]
    for pattern in patterns:
        match = re.search(pattern, body, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


TRAILER_PATTERN = re.compile(
    r"<!--\s*DEVIN_AUDIT_STATE:\s*"
    r"(devin-audit-done|devin-audit-blocked|needs-devin-audit)"
    r"\s*-->",
    flags=re.IGNORECASE,
)


def classify_audit_comment(body: str) -> Optional[str]:
    # HEAD_CHANGED is conservative — wins over any positive trailer to avoid
    # marking a PR done/blocked when Devin observed the head moving mid-review.
    if "HEAD_CHANGED_DURING_REVIEW" in body:
        return "needs"

    # Authoritative machine-readable trailer. Prose phrasing below is kept as
    # a fallback for any Devin sessions that haven't yet adopted the trailer.
    trailer = TRAILER_PATTERN.search(body)
    if trailer:
        label = trailer.group(1).lower()
        if label == "devin-audit-done":
            return "done"
        if label == "devin-audit-blocked":
            return "blocked"
        return "needs"

    if re.search(
        r"Devin Audit(?:\s+Result)?\s*[—–:-]\s*PASS\b",
        body,
        flags=re.IGNORECASE,
    ):
        return "done"
    if re.search(
        r"\**(?:Intended\s+|Expected\s+)?Label state:\**\s*`?\bdevin-audit-done\b",
        body,
        flags=re.IGNORECASE,
    ):
        return "done"
    if re.search(
        r"(?:Intended|Expected)\s+labels?:\s*(?:add\s+)?`?devin-audit-done\b",
        body,
        flags=re.IGNORECASE,
    ):
        return "done"

    if re.search(
        (
            r"Devin Audit(?:\s+Result)?\s*[—–:-]\s*"
            r"(BLOCKED|BLOCKER|INCOMPLETE)\b"
        ),
        body,
        flags=re.IGNORECASE,
    ):
        return "blocked"
    if re.search(
        r"\**(?:Intended\s+|Expected\s+)?Label state:\**\s*`?\bdevin-audit-blocked\b",
        body,
        flags=re.IGNORECASE,
    ):
        return "blocked"
    if re.search(
        r"(?:Intended|Expected)\s+labels?:\s*(?:add\s+)?`?devin-audit-blocked\b",
        body,
        flags=re.IGNORECASE,
    ):
        return "blocked"

    return None


def sha_matches_reviewed_head(reviewed_sha: str, current_head_sha: str) -> bool:
    reviewed = reviewed_sha.lower()
    current = current_head_sha.lower()
    return reviewed == current or current.startswith(reviewed)


def resolve_label_decision(
    event: Dict[str, Any],
    *,
    current_head_sha: Optional[str],
) -> tuple[Optional[LabelDecision], str]:
    if event.get("action") != "created":
        return None, f"unsupported action: {event.get('action')}"

    issue = event.get("issue") or {}
    if "pull_request" not in issue:
        return None, "comment is not on a pull request"

    comment = event.get("comment") or {}
    author = (comment.get("user") or {}).get("login", "")
    if not is_devin_comment_author(author):
        return None, f"ignored comment author: {author}"

    body = comment.get("body") or ""
    status = classify_audit_comment(body)
    if status is None:
        return None, "comment is not a final Devin audit result"

    issue_number = int(issue["number"])
    reviewed_sha = extract_reviewed_sha(body)
    if status != "needs" and not reviewed_sha:
        return None, "audit result is missing Head SHA"
    if reviewed_sha and current_head_sha and not sha_matches_reviewed_head(reviewed_sha, current_head_sha):
        return LabelDecision(
            issue_number=issue_number,
            add_label=NEEDS_LABEL,
            remove_labels=(DONE_LABEL, BLOCKED_LABEL),
            reviewed_sha=reviewed_sha,
            reason=f"reviewed SHA {reviewed_sha} does not match current head {current_head_sha}",
        ), "label needs audit"

    if status == "needs":
        return LabelDecision(
            issue_number=issue_number,
            add_label=NEEDS_LABEL,
            remove_labels=(DONE_LABEL, BLOCKED_LABEL),
            reviewed_sha=reviewed_sha,
            reason="audit reported head changed during review",
        ), "label needs audit"

    if status == "done":
        return LabelDecision(
            issue_number=issue_number,
            add_label=DONE_LABEL,
            remove_labels=(NEEDS_LABEL, BLOCKED_LABEL),
            reviewed_sha=reviewed_sha,
            reason="audit passed",
        ), "label done"

    return LabelDecision(
        issue_number=issue_number,
        add_label=BLOCKED_LABEL,
        remove_labels=(NEEDS_LABEL, DONE_LABEL),
        reviewed_sha=reviewed_sha,
        reason="audit blocked or incomplete",
    ), "label blocked"


def github_request(
    method: str,
    path: str,
    *,
    token: str,
    body: Optional[Dict[str, Any]] = None,
    allow_missing: bool = False,
) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        f"https://api.github.com{path}",
        data=data,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body) if response_body else None
    except urllib.error.HTTPError as exc:
        if allow_missing and exc.code == 404:
            return None
        response_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {method} {path} failed: HTTP {exc.code}\n{response_body}") from exc


def fetch_pull_request(repo: str, issue_number: int, *, token: str) -> Dict[str, Any]:
    return github_request("GET", f"/repos/{repo}/pulls/{issue_number}", token=token)


def apply_label_decision(repo: str, decision: LabelDecision, *, token: str) -> None:
    github_request(
        "POST",
        f"/repos/{repo}/issues/{decision.issue_number}/labels",
        token=token,
        body={"labels": [decision.add_label]},
    )
    for label in decision.remove_labels:
        github_request(
            "DELETE",
            f"/repos/{repo}/issues/{decision.issue_number}/labels/{quote(label, safe='')}",
            token=token,
            allow_missing=True,
        )


def main() -> int:
    event_path = Path(os.environ["GITHUB_EVENT_PATH"])
    repo = os.environ["GITHUB_REPOSITORY"]
    event = load_json(event_path)

    current_head_sha = os.environ.get("DRY_RUN_HEAD_SHA")
    token = os.environ.get("GITHUB_TOKEN")
    if not os.environ.get("DRY_RUN"):
        if not token:
            print("error: GITHUB_TOKEN is required", file=sys.stderr)
            return 1
        issue_number = int((event.get("issue") or {}).get("number", 0))
        if issue_number:
            current_head_sha = fetch_pull_request(repo, issue_number, token=token)["head"]["sha"]

    decision, reason = resolve_label_decision(event, current_head_sha=current_head_sha)
    if decision is None:
        print(f"skip: {reason}")
        return 0

    if os.environ.get("DRY_RUN"):
        print(json.dumps({"decision": decision.__dict__, "reason": reason}, sort_keys=True))
        return 0

    apply_label_decision(repo, decision, token=token or "")
    print(
        f"applied: add {decision.add_label}; remove {', '.join(decision.remove_labels)} "
        f"on {repo}#{decision.issue_number} ({decision.reason})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
