#!/usr/bin/env python3
"""Apply Gitar audit result labels from Gitar-bot dashboard comments.

Gitar (https://gitar.ai) is an AI code-review GitHub App. UNLIKE the
Greptile labeler (which fires on `pull_request_review` events and reads
per-line P-badges), Gitar posts a single ISSUE COMMENT — a "dashboard"
comment — that contains a collapsible Code Review block whose verdict is
encoded as an HTML `<kbd>` badge in the summary line, e.g.:

    <details>
    <summary><b>Code Review</b> <kbd>✅ Approved</kbd></summary>

    ...prose...

    </details>

So this labeler fires on `issue_comment` events from `gitar-bot`, parses
that badge, and applies one of:

- `gitar-audit-done`     (verdict badge says "Approved")
- `gitar-audit-blocked`  (verdict badge present but NOT "Approved")
- `needs-gitar-audit`    (Code Review block present but badge unparseable)

## How this lane differs from the Greptile lane (deliberately)

1. **Act-on-all, not opt-in.** Greptile is a PAID lane gated behind
   `needs-greptile-audit` to control spend. Gitar's review is free and
   auto-runs on every PR, so we WANT the signal on every PR. There is no
   opt-in gate: the labeler acts on any Gitar Code Review comment. The
   `needs-gitar-audit` label is used only for the fail-closed
   (unparseable-verdict) case and as a manual "look again" marker.

2. **No stale-SHA / clear-on-push job.** The Codex lane has an elaborate
   `clear-stale-on-push` handler because a stale `codex-audit-done`
   could falsely authorize a merge. The Gitar lane is INFORMATIONAL
   ONLY (never gates merge, exactly like Greptile), so a momentarily
   stale label is harmless. Gitar also re-reviews on every push and
   edits/reposts its dashboard comment, so the label naturally tracks
   the latest review (the labeler also fires on `edited`).

3. **Verdict lives in the comment body**, not in fetched inline review
   comments, so there is no pagination / no separate fetch — the
   `issue_comment` event payload already carries everything we need.

## Fail-closed posture

- A Gitar comment that is NOT a Code Review verdict comment (e.g. an
  in-progress "Reviewing your code" badge with no Code Review summary,
  a reply, or a CI comment) is SKIPPED — it must never move a label.
- A Code Review comment whose badge cannot be read is requeued to
  `needs-gitar-audit` rather than guessed.
- Any badge that is present but does NOT say "Approved" is treated as
  `gitar-audit-blocked` (the safe direction for an informational lane:
  surface that Gitar flagged something rather than silently mark done).
  The exact non-approved badge taxonomy is refined as real non-approved
  verdicts are observed; see GITAR_AUDIT_PROTOCOL.md.

## Verdict source: native `<kbd>` badge only

The verdict is read solely from Gitar's own Code Review `<kbd>` badge (its
authoritative dashboard output). We previously also accepted an optional
`<!-- GITAR_AUDIT_STATE: ... -->` trailer Gitar could be instructed to
append, but Gitar does not honor that instruction in practice (confirmed
across several PRs) and the trailer is PR-influenced content that proved a
recurring spoofing surface, so it was removed. PR-controlled text echoed
inside code fences/spans is blanked before parsing so a quoted badge
cannot be mistaken for Gitar's own verdict.

A first-class machine-readable verdict (a read-only API field, or a
verdict label decoupled from a code-host approval) would be preferable,
but on the Pro plan the verdict API is Enterprise-gated and the native
`gitar-approved` label is coupled to Gitar submitting a GitHub approval
(which conflicts with this lane's advisory-only posture). Badge parsing is
the correct integration for now; see GITAR_AUDIT_PROTOCOL.md.
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
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote


NEEDS_LABEL = "needs-gitar-audit"
DONE_LABEL = "gitar-audit-done"
BLOCKED_LABEL = "gitar-audit-blocked"

# Author login(s) that Gitar posts comments under. Override via
# GITAR_BOT_AUTHORS for installs that use a different bot identity.
# Default-empty env var must not defeat the default (the
# codex/greptile labeler #234 round-1 P2 pattern). Confirmed login
# `gitar-bot` from real Gitar reviews on cube-snap #270 / ctvd #411.
_DEFAULT_AUTHORS = "gitar-bot,gitar-bot[bot],gitar-ai[bot]"
GITAR_COMMENT_AUTHORS = {
    a.strip().lower()
    for a in (os.environ.get("GITAR_BOT_AUTHORS") or _DEFAULT_AUTHORS).split(",")
    if a.strip()
}


@dataclass(frozen=True)
class LabelDecision:
    issue_number: int
    add_label: str
    remove_labels: tuple
    reason: str


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_gitar_comment_author(login: str) -> bool:
    return login.lower() in GITAR_COMMENT_AUTHORS


# ----- Verdict parsing -----

# Capture the verdict badge that immediately follows the bold
# "Code Review" label inside Gitar's dashboard `<summary>`:
#
#     <summary><b>Code Review</b> <kbd>✅ Approved</kbd></summary>
#
# Anchoring on `Code Review</b>` deliberately excludes the transient
# in-progress badge Gitar prepends while it works, e.g.
# `<kbd><img ...gitar-spin.svg...> Reviewing your code</kbd>`, which is
# NOT preceded by the Code Review label.
# Detect the presence of Gitar's Code Review verdict block (the bold
# "Code Review" label in its dashboard <summary>), independent of the
# badge markup. Used to distinguish "not a verdict comment" (skip) from
# "verdict comment whose badge we could not parse" (fail closed) — Codex
# P2 on cube-snap#271.
_CODE_REVIEW_LABEL_RE = re.compile(r"Code Review\s*</b>", re.IGNORECASE)

# Badge extractor. `<kbd[^>]*>` tolerates attribute drift (e.g.
# `<kbd class="...">`) so a benign markup change degrades to fail-closed
# (needs) rather than silently mismatching.
_CODE_REVIEW_BADGE_RE = re.compile(
    r"Code Review\s*</b>\s*<kbd[^>]*>\s*(?P<badge>.*?)\s*</kbd>",
    re.IGNORECASE | re.DOTALL,
)


def has_code_review_block(comment_body: str) -> bool:
    """True if the comment contains Gitar's Code Review verdict block."""
    return bool(_CODE_REVIEW_LABEL_RE.search(comment_body))


def parse_code_review_badge(comment_body: str) -> Optional[str]:
    """Return the Code Review verdict badge text, or None if no
    `Code Review</b> <kbd>...</kbd>` badge could be extracted (no block at
    all, or the badge markup drifted). The string may be empty if the
    <kbd> tag was present but empty. Callers pair this with
    `has_code_review_block` to fail closed on drift rather than skip."""
    m = _CODE_REVIEW_BADGE_RE.search(comment_body)
    if not m:
        return None
    # Strip any nested tags (e.g. an <img> inside the badge) so the
    # classifier sees only the textual verdict.
    badge = re.sub(r"<[^>]*>", "", m.group("badge"))
    return badge.strip()


def classify_badge(badge: Optional[str]) -> Optional[str]:
    """Map a parsed badge to 'done' | 'blocked' | 'needs' | None.

    - None badge  -> None  (not a Code Review verdict comment; skip)
    - ""          -> 'needs' (verdict block present but empty; fail closed)
    - contains 'approved' (case-insensitive) -> 'done'
    - any other non-empty verdict text       -> 'blocked' (safe direction)
    """
    if badge is None:
        return None
    if not badge:
        return "needs"
    # Exact-match the verdict word, ignoring a leading emoji/symbol and
    # surrounding whitespace (e.g. "✅ Approved" -> "approved"). A substring
    # check is unsafe: "Not Approved" / "Unapproved" contain "approved" but
    # are NOT approvals and must fail closed to blocked (Codex P2 on
    # ctvd#413). Only the exact `Approved` verdict yields done.
    if re.sub(r"[^a-z]", "", badge.lower()) == "approved":
        return "done"
    return "blocked"


# Gitar echoes PR-controlled content (command examples, quoted snippets)
# inside code fences and inline-code spans, so a Code Review badge quoted
# there must NOT be mistaken for Gitar's own verdict. `_strip_code` blanks
# code to spaces (preserving newlines/columns) before badge parsing.
# Run-aware: a Markdown code span/fence is an opening run of N backticks
# closed by a run of N backticks (single `x`, double ``x``, fenced
# ```...```), so `(`+).*?\1` covers them ALL; ~~~ fences too.
_CODE_FENCE_RE = re.compile(r"(`+).*?\1|~~~.*?~~~", re.DOTALL)


def _strip_code(text: str) -> str:
    """Blank code spans/fences to spaces, preserving newlines and column
    positions, so a Code Review badge quoted inside PR-controlled code is
    not read as Gitar's own verdict."""
    return _CODE_FENCE_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def classify_gitar_comment(comment_body: str) -> Tuple[Optional[str], str]:
    """Classify a Gitar comment into ('done'|'blocked'|'needs'|None, detail).

    Reads Gitar's native `<kbd>` Code Review badge. None means this is not
    a Gitar verdict comment at all and should be skipped; a Code Review
    block whose badge can't be parsed fails closed to 'needs'.
    """
    # Blank code (fenced + inline) so a Code Review block/badge quoted in
    # PR-controlled code is not mistaken for Gitar's own verdict.
    body = _strip_code(comment_body)
    if not has_code_review_block(body):
        return None, "no Code Review block"
    badge = parse_code_review_badge(body)
    if not badge:
        # Block present but the badge is missing/empty or its markup
        # drifted: fail closed to needs rather than silently skipping and
        # leaving a stale label (Codex P2 on cube-snap#271).
        return "needs", "Code Review block present but badge unparseable"
    return classify_badge(badge), f"Code Review badge {badge!r}"


# ----- GitHub API helpers -----


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
        raise RuntimeError(
            f"GitHub API {method} {path} failed: HTTP {exc.code}\n{response_body}"
        ) from exc


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


# ----- Decision logic -----


def resolve_label_decision(
    event: Dict[str, Any],
) -> Tuple[Optional[LabelDecision], str]:
    """Compute the label transition for a Gitar issue_comment event.

    Pure function over the event payload — no GitHub calls. Returns
    (decision, reason); decision None means 'do nothing'.

    Gates:
      1. Action must be created/edited.
      2. The issue must be a pull request.
      3. The comment author must be a Gitar bot.
      4. The comment must carry a native Code Review <kbd> badge; else
         skip — Gitar posts many non-verdict comment types.
      5. Verdict: 'Approved' badge -> done; other non-empty badge ->
         blocked; badge present but unparseable -> needs (fail closed).
    """
    if event.get("action") not in ("created", "edited"):
        return None, f"unsupported issue_comment action: {event.get('action')}"

    issue = event.get("issue") or {}
    if "pull_request" not in issue:
        return None, "comment is on an issue, not a pull request"

    issue_number = int(issue.get("number") or 0)
    if not issue_number:
        return None, "event has no issue/PR number"

    comment = event.get("comment") or {}
    author = (comment.get("user") or {}).get("login", "")
    if not is_gitar_comment_author(author):
        return None, f"ignored comment author: {author}"

    status, detail = classify_gitar_comment(comment.get("body") or "")

    if status is None:
        return None, "no Gitar Code Review verdict badge — skipping"

    if status == "needs":
        return LabelDecision(
            issue_number=issue_number,
            add_label=NEEDS_LABEL,
            remove_labels=(DONE_LABEL, BLOCKED_LABEL),
            reason=(
                "Gitar Code Review block present but verdict was "
                f"empty/unparseable ({detail}) — failing closed"
            ),
        ), "format unknown — re-queue"

    if status == "blocked":
        return LabelDecision(
            issue_number=issue_number,
            add_label=BLOCKED_LABEL,
            remove_labels=(NEEDS_LABEL, DONE_LABEL),
            reason=f"Gitar verdict: blocked ({detail})",
        ), "label blocked"

    return LabelDecision(
        issue_number=issue_number,
        add_label=DONE_LABEL,
        remove_labels=(NEEDS_LABEL, BLOCKED_LABEL),
        reason=f"Gitar verdict: done ({detail})",
    ), "label done"


# ----- CLI entry point (run inside the GitHub Action) -----


def _apply_or_log(repo: str, decision: LabelDecision, *, token: str) -> None:
    """Apply a label decision; treat failures as non-fatal.

    The Gitar lane is INFORMATIONAL ONLY (per CLAUDE.md merge policy).
    Label-application failures (e.g. a PAT permission gap on
    `POST /repos/.../issues/N/labels`) must not fail the workflow check,
    otherwise PRs go UNSTABLE and the lane becomes an accidental merge
    gate. Log the verdict + any failure; the caller exits 0 regardless.
    (Same posture as the Greptile labeler's `_apply_or_log`.)
    """
    try:
        apply_label_decision(repo, decision, token=token)
        print(
            f"applied: add {decision.add_label}; remove "
            f"{', '.join(decision.remove_labels)} "
            f"on {repo}#{decision.issue_number} ({decision.reason})"
        )
    except Exception as exc:
        print(
            f"verdict (label apply skipped — informational lane is "
            f"non-blocking): add {decision.add_label}; remove "
            f"{', '.join(decision.remove_labels)} "
            f"on {repo}#{decision.issue_number} ({decision.reason})"
        )
        print(f"warn: could not apply label: {exc}", file=sys.stderr)


def main() -> int:
    event_path = Path(os.environ["GITHUB_EVENT_PATH"])
    repo = os.environ["GITHUB_REPOSITORY"]
    event = load_json(event_path)

    token = os.environ.get("GITHUB_TOKEN")
    if not os.environ.get("DRY_RUN") and not token:
        print("error: GITHUB_TOKEN is required", file=sys.stderr)
        return 1

    decision, reason = resolve_label_decision(event)

    if decision is None:
        print(f"skip: {reason}")
        return 0

    if os.environ.get("DRY_RUN"):
        print(json.dumps({"decision": decision.__dict__, "reason": reason}, sort_keys=True))
        return 0

    _apply_or_log(repo, decision, token=token or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
