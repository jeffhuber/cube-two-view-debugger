#!/usr/bin/env python3
"""Apply Codex audit result labels from Codex-bot PR comments.

Parallels `devin_audit_labeler.py`. Same label state-machine shape,
different label prefix and bot author. Codex audit empirically catches
real bugs the other reviewers miss (PR #233 calibration: 6 real findings
vs Devin's final-state-PASS and Qwen's 7 false positives), and is now
the **preferred** merge-authority signal — Claude's standing in-thread
merge delegation accepts EITHER `codex-audit-done` OR `devin-audit-done`
+ CLEAN, with Codex's verdict leading when both are available. See
`tools/CODEX_AUDIT_PROTOCOL.md`.

Trailer protocol (machine-readable, authoritative over prose):
    <!-- CODEX_AUDIT_STATE: codex-audit-done -->
    <!-- CODEX_AUDIT_STATE: codex-audit-blocked -->
    <!-- CODEX_AUDIT_STATE: needs-codex-audit -->   (head changed during review)
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from urllib.parse import quote


NEEDS_LABEL = "needs-codex-audit"
DONE_LABEL = "codex-audit-done"
BLOCKED_LABEL = "codex-audit-blocked"

# Author login(s) that the Codex bridge posts audit comments under.
# Override via CODEX_BOT_AUTHORS env var (comma-separated) so users can
# point at their own service account without forking the labeler.
#
# Codex round 1 of #234 — P2: the GitHub Actions workflow always
# exports `CODEX_BOT_AUTHORS: ${{ vars.CODEX_BOT_AUTHORS }}`; when the
# repo variable is unset, GitHub passes an empty string (NOT an unset
# env var). `os.environ.get(name, default)` returns the default ONLY
# if the key is missing — an empty-string value defeats the default.
# Use `or _DEFAULT_AUTHORS` to fall back when the value is empty.
_DEFAULT_AUTHORS = "codex-audit-bot,codex-audit-bot[bot]"
CODEX_COMMENT_AUTHORS = {
    a.strip().lower()
    for a in (os.environ.get("CODEX_BOT_AUTHORS") or _DEFAULT_AUTHORS).split(",")
    if a.strip()
}


@dataclass(frozen=True)
class LabelDecision:
    issue_number: int
    add_label: str
    remove_labels: tuple[str, ...]
    reviewed_sha: Optional[str]
    reason: str


@dataclass(frozen=True)
class GitHubToken:
    name: str
    value: str


class GitHubRequestError(RuntimeError):
    def __init__(self, method: str, path: str, code: int, response_body: str) -> None:
        self.method = method
        self.path = path
        self.code = code
        self.response_body = response_body
        super().__init__(f"GitHub API {method} {path} failed: HTTP {code}\n{response_body}")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_codex_comment_author(login: str) -> bool:
    return login.lower() in CODEX_COMMENT_AUTHORS


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
    r"<!--\s*CODEX_AUDIT_STATE:\s*"
    r"(codex-audit-done|codex-audit-blocked|needs-codex-audit)"
    r"\s*-->",
    flags=re.IGNORECASE,
)


def classify_audit_comment(body: str) -> Optional[str]:
    # Machine-readable trailer is authoritative. Use the LAST trailer in
    # the body, not the first — Codex round 3 of #234 found that quoting
    # the trailer text in protocol docs / tests / earlier review prose
    # would make `search()` return the quoted occurrence, mislabeling a
    # BLOCKED result as done. The real trailer is the one `format_comment`
    # appends as the final line.
    trailers = list(TRAILER_PATTERN.finditer(body))
    if trailers:
        label = trailers[-1].group(1).lower()
        if label == "codex-audit-done":
            return "done"
        if label == "codex-audit-blocked":
            return "blocked"
        return "needs"

    # Prose fallbacks (less reliable; only used if trailer was missed).
    # The Codex CLI's stdout uses "Codex Audit: PASS" / "Codex Audit: BLOCKED"
    # in the formatted comment per `codex_audit_pr.format_comment`.
    if re.search(r"Codex Audit(?:\s+Result)?\s*[—–:-]\s*PASS\b", body, flags=re.IGNORECASE):
        return "done"
    if re.search(
        (
            r"Codex Audit(?:\s+Result)?\s*[—–:-]\s*"
            r"(BLOCKED|BLOCKER|INCOMPLETE)\b"
        ),
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
    if not is_codex_comment_author(author):
        return None, f"ignored comment author: {author}"

    body = comment.get("body") or ""
    status = classify_audit_comment(body)
    if status is None:
        return None, "comment is not a final Codex audit result"

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
        raise GitHubRequestError(method, path, exc.code, response_body) from exc


def github_request_with_fallback(
    method: str,
    path: str,
    *,
    tokens: Sequence[GitHubToken],
    body: Optional[Dict[str, Any]] = None,
    allow_missing: bool = False,
) -> Any:
    """Use the optional PAT first, then fall back to GITHUB_TOKEN on auth errors.

    CODEX_AUDIT_LABEL_TOKEN is useful when GitHub's built-in token cannot write
    labels for a deployment, but an under-scoped PAT should not make the
    otherwise-valid built-in token unreachable. Only auth/permission failures
    fall through; semantic API failures such as a missing label still fail fast.
    """
    token_list = tuple(tokens)
    last_error: Optional[GitHubRequestError] = None
    for index, token in enumerate(token_list):
        try:
            return github_request(
                method,
                path,
                token=token.value,
                body=body,
                allow_missing=allow_missing,
            )
        except GitHubRequestError as exc:
            last_error = exc
            if exc.code not in {401, 403}:
                raise
            suffix = (
                "; trying next token"
                if index < len(token_list) - 1
                else "; no more tokens"
            )
            print(
                f"warning: {method} {path} failed with HTTP {exc.code} using "
                f"{token.name}{suffix}",
                file=sys.stderr,
            )
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"no GitHub tokens available for {method} {path}")


def fetch_pull_request(
    repo: str,
    issue_number: int,
    *,
    token: Optional[str] = None,
    tokens: Optional[Sequence[GitHubToken]] = None,
) -> Dict[str, Any]:
    path = f"/repos/{repo}/pulls/{issue_number}"
    if tokens is not None:
        return github_request_with_fallback("GET", path, tokens=tokens)
    if token is None:
        raise ValueError("token or tokens is required")
    return github_request("GET", path, token=token)


def apply_label_decision(
    repo: str,
    decision: LabelDecision,
    *,
    token: Optional[str] = None,
    tokens: Optional[Sequence[GitHubToken]] = None,
) -> None:
    if tokens is None:
        if token is None:
            raise ValueError("token or tokens is required")
        tokens = (GitHubToken("GITHUB_TOKEN", token),)
    github_request_with_fallback(
        "POST",
        f"/repos/{repo}/issues/{decision.issue_number}/labels",
        tokens=tokens,
        body={"labels": [decision.add_label]},
    )
    for label in decision.remove_labels:
        github_request_with_fallback(
            "DELETE",
            f"/repos/{repo}/issues/{decision.issue_number}/labels/{quote(label, safe='')}",
            tokens=tokens,
            allow_missing=True,
        )


def github_tokens_from_env() -> tuple[GitHubToken, ...]:
    tokens = []
    seen = set()
    for name in ("CODEX_AUDIT_LABEL_TOKEN", "GITHUB_TOKEN"):
        value = (os.environ.get(name) or "").strip()
        if not value or value in seen:
            continue
        tokens.append(GitHubToken(name, value))
        seen.add(value)
    return tuple(tokens)


def main() -> int:
    event_path = Path(os.environ["GITHUB_EVENT_PATH"])
    repo = os.environ["GITHUB_REPOSITORY"]
    event = load_json(event_path)

    current_head_sha = os.environ.get("DRY_RUN_HEAD_SHA")
    tokens = github_tokens_from_env()
    if not os.environ.get("DRY_RUN"):
        if not tokens:
            print("error: GITHUB_TOKEN or CODEX_AUDIT_LABEL_TOKEN is required", file=sys.stderr)
            return 1
        # Codex round 1 of #234 — P3: the workflow fires on every
        # `issue_comment`, including comments on regular (non-PR) issues.
        # Skip the `/pulls/{n}` fetch when the issue isn't a PR — otherwise
        # the call 404s and the Action fails for unrelated issue comments.
        issue = event.get("issue") or {}
        issue_number = int(issue.get("number", 0))
        is_pr = "pull_request" in issue
        if issue_number and is_pr:
            current_head_sha = fetch_pull_request(repo, issue_number, tokens=tokens)["head"]["sha"]

    decision, reason = resolve_label_decision(event, current_head_sha=current_head_sha)
    if decision is None:
        print(f"skip: {reason}")
        return 0

    if os.environ.get("DRY_RUN"):
        print(json.dumps({"decision": decision.__dict__, "reason": reason}, sort_keys=True))
        return 0

    apply_label_decision(repo, decision, tokens=tokens)
    print(
        f"applied: add {decision.add_label}; remove {', '.join(decision.remove_labels)} "
        f"on {repo}#{decision.issue_number} ({decision.reason})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
