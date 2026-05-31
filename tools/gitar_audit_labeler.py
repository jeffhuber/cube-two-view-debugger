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

## Trailer protocol (preferred) with native-badge fallback

Gitar can be configured (via its dashboard "Custom instructions" or a
`.gitar/` rule) to append our `<!-- GITAR_AUDIT_STATE: ... -->` trailer,
exactly like the Codex/Devin lanes. When that trailer is present this
labeler treats it as AUTHORITATIVE (last trailer wins). When it is
absent — Gitar honoring the instruction is best-effort and not
guaranteed — the labeler falls back to parsing the native `<kbd>` Code
Review badge. The two-layer design means the lane works whether or not
Gitar emits the trailer, and auto-upgrades to trailer-grade reliability
if it does.
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
_CODE_REVIEW_BADGE_RE = re.compile(
    r"Code Review\s*</b>\s*<kbd>\s*(?P<badge>.*?)\s*</kbd>",
    re.IGNORECASE | re.DOTALL,
)


def parse_code_review_badge(comment_body: str) -> Optional[str]:
    """Return the Code Review verdict badge text, or None if this
    comment has no Code Review verdict block (e.g. an in-progress or
    non-review Gitar comment). The returned string may be empty if the
    badge tag was present but empty — the caller fails closed on that."""
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
# inside code fences and inline-code spans. A `<!-- GITAR_AUDIT_STATE: ... -->`
# planted there must NOT be trusted, so fenced/inline code is stripped
# before the trailer search (Codex P2 on ctvd#413).
_CODE_FENCE_RE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]*`", re.DOTALL)

# Authoritative trailer Gitar can be configured to emit (preferred over
# the native badge). It must be ALONE on its line — anchored ^...$ in
# MULTILINE — the way an appended HTML trailer is emitted, so an
# inline/quoted trailer with other text on the line is not authoritative
# (Codex P2 on ctvd#413). Strict three-value alternation so a malformed
# `<!-- GITAR_AUDIT_STATE: foo -->` does not match and falls through to
# the badge parse (mirrors the Codex labeler's TRAILER_PATTERN
# tightening, cube-snap#206 / ctvd#373).
_STATE_TRAILER_RE = re.compile(
    r"^\s*<!--\s*GITAR_AUDIT_STATE:\s*"
    r"(gitar-audit-done|gitar-audit-blocked|needs-gitar-audit)\s*-->\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_state_trailer(comment_body: str) -> Optional[str]:
    """Return 'done' | 'blocked' | 'needs' from the last standalone-line
    GITAR_AUDIT_STATE trailer in non-fenced text, or None if absent."""
    body = _CODE_FENCE_RE.sub("", comment_body)
    matches = _STATE_TRAILER_RE.findall(body)
    if not matches:
        return None
    tag = matches[-1].lower()  # last trailer wins
    if tag == DONE_LABEL:
        return "done"
    if tag == BLOCKED_LABEL:
        return "blocked"
    return "needs"


def classify_gitar_comment(comment_body: str) -> Tuple[Optional[str], str]:
    """Classify a Gitar comment into ('done'|'blocked'|'needs'|None, detail).

    Trailer-first: if Gitar emitted our authoritative
    `<!-- GITAR_AUDIT_STATE: ... -->` trailer, use it. Otherwise fall
    back to the native `<kbd>` Code Review badge. None means this is not
    a Gitar verdict comment at all and should be skipped.
    """
    trailer = parse_state_trailer(comment_body)
    if trailer is not None:
        return trailer, "GITAR_AUDIT_STATE trailer"
    badge = parse_code_review_badge(comment_body)
    status = classify_badge(badge)
    if status is None:
        return None, "no trailer and no Code Review badge"
    return status, f"Code Review badge {badge!r}"


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
      4. The comment must carry a verdict (our GITAR_AUDIT_STATE trailer
         OR a native Code Review <kbd> badge); else skip — Gitar posts
         many non-verdict comment types.
      5. Verdict: trailer wins; else 'Approved' badge -> done; other
         non-empty badge -> blocked; empty badge -> needs (fail closed).
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
        return None, "no Gitar verdict (trailer or Code Review badge) — skipping"

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
