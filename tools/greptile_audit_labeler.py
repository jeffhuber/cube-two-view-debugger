#!/usr/bin/env python3
"""Apply Greptile audit result labels from Greptile-bot PR reviews.

UNLIKE Devin/Qwen/Codex labelers (which fire on issue_comment events),
the Greptile labeler fires on `pull_request_review` events from
`greptile-apps[bot]`. Greptile's review style is different:

- It posts a single PR REVIEW submission via the GitHub Reviews API
  with `state: COMMENTED` (NOT approved, NOT changes_requested) and
  an empty review body.
- The actual findings live as INLINE REVIEW COMMENTS on changed lines.
- Each finding starts with a severity badge encoded as an HTML img tag:
    <a href="#"><img alt="P1" src="https://greptile-static-assets.s3.amazonaws.com/badges/p1.svg?v=7" ...></a>

This labeler reads the review's inline comments, counts P0/P1 badges
(which we treat as blockers), and applies one of:

- `greptile-audit-done`
- `greptile-audit-blocked`
- `needs-greptile-audit` (re-queue if format unrecognized or head stale)

## Four defensive gates (per Codex's tightening — see GREPTILE_AUDIT_PROTOCOL.md)

1. **Opt-in gate.** Only flip labels on PRs that currently carry the
   `needs-greptile-audit` label. Greptile auto-fires on every PR — but
   only PRs we explicitly opt in to the bake-off get labeled.

2. **Stale-HEAD gate.** Compare `review.commit_id` (the SHA Greptile
   actually reviewed) to the PR's current head SHA. If they differ,
   keep / re-apply `needs-greptile-audit`.

3. **Severity parse, fail closed.** Preferred: extract severity from
   `alt="P[N]"` in inline-comment HTML img tags. Cross-check via the
   badge URL pattern `greptile-static-assets.../badges/p{N}.svg`. If
   inline comments exist but NO severity markers of any tier can be
   parsed, fail closed: keep `needs-greptile-audit` rather than
   defaulting to a verdict we can't trust.

4. **Verdict.** Any P0 or P1 → `greptile-audit-blocked`. Zero P0/P1
   (P2/P3 allowed; zero badges of any tier with zero inline comments
   means clean review) → `greptile-audit-done`.

## Trailer protocol

Greptile does NOT emit our `<!-- *_AUDIT_STATE: ... -->` trailer
(its output format is its own). This labeler is the authoritative
classifier; there's no trailer to fall back on.

If Greptile ever supports custom output config that emits our
trailer, we can layer it on the badge-count parse. Until then, the
labeler keys on severity-badge markers.
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


NEEDS_LABEL = "needs-greptile-audit"
DONE_LABEL = "greptile-audit-done"
BLOCKED_LABEL = "greptile-audit-blocked"

# Author login(s) that Greptile posts reviews under. Override via
# GREPTILE_BOT_AUTHORS for self-hosted Greptile installs that use a
# different bot identity. Default-empty env var doesn't defeat the
# default (codex_audit_labeler.py #234 round-1 P2 fix pattern).
_DEFAULT_AUTHORS = "greptile-apps[bot],greptile-apps"
GREPTILE_REVIEW_AUTHORS = {
    a.strip().lower()
    for a in (os.environ.get("GREPTILE_BOT_AUTHORS") or _DEFAULT_AUTHORS).split(",")
    if a.strip()
}


@dataclass(frozen=True)
class LabelDecision:
    issue_number: int
    add_label: str
    remove_labels: tuple
    reviewed_sha: Optional[str]
    reason: str


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_greptile_review_author(login: str) -> bool:
    return login.lower() in GREPTILE_REVIEW_AUTHORS


# ----- Severity parsing -----

# Preferred: extract from `alt="P0"` / `alt="P1"` etc. — Codex's scoping
# recommendation. The alt attribute is more semantic and stable than the
# badge URL (which could change with CDN versioning).
_ALT_P_TAG_RE = re.compile(r'alt="P([0-3])"', flags=re.IGNORECASE)

# Cross-check via the badge URL pattern as a fallback (in case Greptile
# emits the img without an alt, or with a different alt value).
_BADGE_URL_RE = re.compile(
    r"greptile-static-assets\.s3\.amazonaws\.com/badges/p([0-3])\.svg",
    flags=re.IGNORECASE,
)


def severity_of(comment_body: str) -> Optional[int]:
    """Return 0/1/2/3 if this inline comment has a P-badge, else None.

    Prefer the `alt="P[N]"` form; fall back to the badge URL pattern.
    Returns None if neither is present — caller treats that as
    'format unknown', fails closed in the verdict logic.
    """
    m = _ALT_P_TAG_RE.search(comment_body)
    if m:
        return int(m.group(1))
    m = _BADGE_URL_RE.search(comment_body)
    if m:
        return int(m.group(1))
    return None


@dataclass
class GreptileVerdict:
    """Aggregated parse of a Greptile review's inline comments."""
    p0_count: int = 0
    p1_count: int = 0
    p2_count: int = 0
    p3_count: int = 0
    unparsed_count: int = 0  # inline comments with no recognizable badge

    @property
    def total_comments(self) -> int:
        return (self.p0_count + self.p1_count + self.p2_count
                + self.p3_count + self.unparsed_count)

    @property
    def blocker_count(self) -> int:
        # Policy: P0 / P1 = blocker. P2 / P3 = concern (non-blocking).
        return self.p0_count + self.p1_count

    def classify(self) -> str:
        """Return 'done', 'blocked', or 'needs' (the latter means
        fail-closed because format couldn't be parsed).

        Codex review of PR #235 — P2 (round 1): ANY unparsed comment
        triggers fail-closed, not only the all-unparsed case. Each
        unparsed inline comment is unknown signal — could be a
        malformed P0/P1 the regex missed. Requeueing is the
        conservative choice; the original 'use what we got'
        semantics could silently ignore a drifted blocker if the
        same review also contained a parseable P2/P3 alongside it.
        """
        if self.unparsed_count > 0:
            return "needs"
        if self.blocker_count > 0:
            return "blocked"
        return "done"


def parse_review_comments(comments: List[Dict[str, Any]]) -> GreptileVerdict:
    """Walk inline review comments and produce a GreptileVerdict."""
    verdict = GreptileVerdict()
    for c in comments:
        body = c.get("body") or ""
        sev = severity_of(body)
        if sev is None:
            verdict.unparsed_count += 1
        elif sev == 0:
            verdict.p0_count += 1
        elif sev == 1:
            verdict.p1_count += 1
        elif sev == 2:
            verdict.p2_count += 1
        elif sev == 3:
            verdict.p3_count += 1
    return verdict


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


def fetch_pull_request(repo: str, pr_number: int, *, token: str) -> Dict[str, Any]:
    return github_request("GET", f"/repos/{repo}/pulls/{pr_number}", token=token)


class ReviewCommentsTruncated(RuntimeError):
    """Raised when fetch_review_comments hits the safety page-cap.
    Caller treats this as fail-closed → `needs-greptile-audit`."""


# Safety cap. A real Greptile review has on the order of 1-20 findings;
# a real PR has a few dozen inline comments. We raise this cap if the
# decommissioning ever produces a counterexample. The orchestrator
# treats hitting it as fail-closed via the `ReviewCommentsTruncated`
# exception, not as silent truncation.
_PAGE_CAP = 50  # 5000 comments


def fetch_review_comments(
    repo: str, pr_number: int, review_id: int, *, token: str,
) -> List[Dict[str, Any]]:
    """Fetch inline review comments that belong to a specific review.

    GitHub's `/pulls/{n}/comments` lists ALL inline comments on the PR
    (paginated, 100/page). Each has a `pull_request_review_id` field;
    we filter to the ones that belong to the review we're labeling for.

    Codex PR #235 — P2 (round 1): paginate. Codex PR #235 — P2
    (round 2): if we hit the safety page cap without exhausting the
    list, raise `ReviewCommentsTruncated` so the orchestrator can
    fail closed to `needs-greptile-audit` — silently returning the
    partial list could cause a P0/P1 on page 51+ to be invisible
    and the audit labeled `done`.
    """
    all_comments: List[Dict[str, Any]] = []
    page = 1
    while page <= _PAGE_CAP:
        chunk = github_request(
            "GET",
            f"/repos/{repo}/pulls/{pr_number}/comments?per_page=100&page={page}",
            token=token,
        ) or []
        if not chunk:
            # Fully exhausted (empty page) — done.
            return [c for c in all_comments if c.get("pull_request_review_id") == review_id]
        all_comments.extend(chunk)
        if len(chunk) < 100:
            # Last (partial) page — exhausted.
            return [c for c in all_comments if c.get("pull_request_review_id") == review_id]
        page += 1
    # Hit the safety cap without exhausting. Fail closed.
    raise ReviewCommentsTruncated(
        f"hit pagination cap of {_PAGE_CAP} pages "
        f"({_PAGE_CAP * 100} comments) for {repo}#{pr_number} "
        f"review {review_id}; refusing to classify on partial data"
    )


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


def sha_matches(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    return a == b or b.startswith(a) or a.startswith(b)


def resolve_label_decision(
    event: Dict[str, Any],
    *,
    pr_labels: List[str],
    current_head_sha: Optional[str],
    review_comments: List[Dict[str, Any]],
) -> Tuple[Optional[LabelDecision], str]:
    """Compute the label transition for a Greptile review event.

    Pure function: takes the parsed event + pre-fetched PR labels +
    current PR head SHA + the review's inline comments. No GitHub
    calls. Returns (decision, reason) where decision is None means
    'do nothing'.

    Four-gate logic (Codex's tightening per scope doc):
      1. Opt-in: PR must carry `needs-greptile-audit` (else no-op)
      2. Stale-HEAD: review.commit_id must match current head (else
         re-apply needs-greptile-audit)
      3. Severity parse: fail closed on unrecognized inline format
      4. Verdict: P0/P1 → blocked; P2/P3-only → done; clean → done
    """
    if event.get("action") not in ("submitted", "edited"):
        return None, f"unsupported pull_request_review action: {event.get('action')}"

    review = event.get("review") or {}
    author = (review.get("user") or {}).get("login", "")
    if not is_greptile_review_author(author):
        return None, f"ignored review author: {author}"

    pr = event.get("pull_request") or {}
    issue_number = int(pr.get("number") or 0)
    if not issue_number:
        return None, "event has no pull request number"

    # Gate 1: opt-in. PR must carry ANY greptile-audit-* label — the
    # initial `needs-greptile-audit` opt-in OR a prior `done`/`blocked`
    # from an earlier Greptile review (which means the PR is already
    # in the bake-off and subsequent reviews should update its state).
    #
    # Codex review of PR #235 — P2 (round 2): without including
    # done/blocked, the first review removes `needs-` and labels with
    # done/blocked. The NEXT Greptile review (after a push) would then
    # be ignored because `needs-` is gone, leaving a stale label.
    pr_labels_set = set(pr_labels or [])
    greptile_labels = {NEEDS_LABEL, DONE_LABEL, BLOCKED_LABEL}
    if not (pr_labels_set & greptile_labels):
        return None, (
            f"PR does not carry any greptile-audit-* label — Greptile "
            f"auto-reviews every PR but only opted-in PRs participate in "
            f"the bake-off. Add `{NEEDS_LABEL}` to opt in."
        )

    # Gate 2: stale-HEAD
    review_sha = review.get("commit_id") or ""
    if current_head_sha and review_sha and not sha_matches(review_sha, current_head_sha):
        return LabelDecision(
            issue_number=issue_number,
            add_label=NEEDS_LABEL,
            remove_labels=(DONE_LABEL, BLOCKED_LABEL),
            reviewed_sha=review_sha,
            reason=f"review SHA {review_sha[:8]} != current head {current_head_sha[:8]}",
        ), "stale review — re-queue"

    # Gate 3+4: parse severity badges
    verdict = parse_review_comments(review_comments)
    status = verdict.classify()

    if status == "needs":
        return LabelDecision(
            issue_number=issue_number,
            add_label=NEEDS_LABEL,
            remove_labels=(DONE_LABEL, BLOCKED_LABEL),
            reviewed_sha=review_sha,
            reason=(
                f"{verdict.total_comments} inline comments present but "
                f"no recognizable P-badges (alt='P[N]' or badge URL) — "
                f"Greptile output format may have drifted; failing closed"
            ),
        ), "format unknown — re-queue"

    if status == "blocked":
        return LabelDecision(
            issue_number=issue_number,
            add_label=BLOCKED_LABEL,
            remove_labels=(NEEDS_LABEL, DONE_LABEL),
            reviewed_sha=review_sha,
            reason=(
                f"Greptile found {verdict.blocker_count} blocker(s) "
                f"(P0={verdict.p0_count}, P1={verdict.p1_count}); "
                f"P2={verdict.p2_count}, P3={verdict.p3_count} concerns"
            ),
        ), "label blocked"

    return LabelDecision(
        issue_number=issue_number,
        add_label=DONE_LABEL,
        remove_labels=(NEEDS_LABEL, BLOCKED_LABEL),
        reviewed_sha=review_sha,
        reason=(
            f"Greptile audit clean "
            f"(0 P0/P1 blockers; "
            f"P2={verdict.p2_count}, P3={verdict.p3_count} concerns)"
        ),
    ), "label done"


# ----- CLI entry point (run inside the GitHub Action) -----


def _apply_or_log(repo: str, decision: LabelDecision, *, token: str) -> None:
    """Apply a label decision; treat failures as non-fatal.

    The Greptile lane is INFORMATIONAL ONLY (per CLAUDE.md merge
    policy). Label-application failures (e.g. PAT permission gap
    on `POST /repos/.../issues/N/labels`) must not fail the
    workflow check, otherwise PRs go UNSTABLE and the lane
    becomes an accidental merge gate. Log the verdict + any
    failure; the caller exits 0 regardless.
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
        print(
            f"warn: could not apply label: {exc}",
            file=sys.stderr,
        )


def _fail_closed_requeue(
    *,
    repo: str,
    event: Dict[str, Any],
    review: Dict[str, Any],
    pr_labels: List[str],
    reason: str,
    token: Optional[str],
) -> int:
    """Re-apply `needs-greptile-audit` after author + opt-in gates pass.

    Used by paths where we cannot fully classify the review (missing
    `review.id` in the event payload, pagination cap hit, etc.).
    Replicates the same author + opt-in gates that `resolve_label_decision`
    enforces, so a non-Greptile review or a non-opted-in PR can never
    mutate labels via the fail-closed path.

    Goes through `_apply_or_log` so label POST failures stay
    non-fatal (the informational-only contract applies here too —
    Codex caught this on the audit-lane-merge-auth PR).
    """
    review_author = (review.get("user") or {}).get("login", "")
    if not is_greptile_review_author(review_author):
        print(f"skip fail-closed: review author "
              f"{review_author!r} is not a Greptile bot")
        return 0
    pr_labels_set = set(pr_labels) if pr_labels else set()
    if not (pr_labels_set & {NEEDS_LABEL, DONE_LABEL, BLOCKED_LABEL}):
        print(f"skip fail-closed: PR has no "
              f"greptile-audit-* label (not opted in)")
        return 0
    issue_number = (event.get("pull_request") or {}).get("number")
    if issue_number:
        _apply_or_log(
            repo,
            LabelDecision(
                issue_number=int(issue_number),
                add_label=NEEDS_LABEL,
                remove_labels=(DONE_LABEL, BLOCKED_LABEL),
                reviewed_sha=(review.get("commit_id") or None),
                reason=reason,
            ),
            token=token or "",
        )
    return 0


def main() -> int:
    event_path = Path(os.environ["GITHUB_EVENT_PATH"])
    repo = os.environ["GITHUB_REPOSITORY"]
    event = load_json(event_path)

    token = os.environ.get("GITHUB_TOKEN")
    if not os.environ.get("DRY_RUN") and not token:
        print("error: GITHUB_TOKEN is required", file=sys.stderr)
        return 1

    pr = event.get("pull_request") or {}
    pr_number = int(pr.get("number") or 0)
    if not pr_number:
        print("skip: event has no pull request number")
        return 0

    # Fetch the PR's current labels + head SHA. (The event includes a
    # snapshot of `pr` but it could be slightly stale.)
    pr_labels: List[str] = []
    current_head_sha: Optional[str] = os.environ.get("DRY_RUN_HEAD_SHA")
    review_comments: List[Dict[str, Any]] = []

    if os.environ.get("DRY_RUN"):
        # In dry-run we read labels/comments from the event for testability
        pr_labels = [l.get("name", "") for l in (pr.get("labels") or [])]
        review_comments = event.get("_dry_run_review_comments") or []
    else:
        pr_current = fetch_pull_request(repo, pr_number, token=token or "")
        pr_labels = [l.get("name", "") for l in (pr_current.get("labels") or [])]
        current_head_sha = pr_current.get("head", {}).get("sha")
        review = event.get("review") or {}
        review_id = review.get("id")
        if not review_id:
            # Devin audit on cube-snap #145 — missing `review.id` in the
            # event payload (malformed event, mock, redelivery oddity)
            # means we cannot fetch the inline comments needed to score
            # severity. Previously `review_comments` stayed `[]` and
            # fell through to `resolve_label_decision`, which classifies
            # empty comments as a clean review and applies
            # `greptile-audit-done` — false-PASS on incomplete data.
            # Fail closed to needs-greptile-audit instead.
            print("warn: review.id missing from event — "
                  "cannot fetch inline comments", file=sys.stderr)
            return _fail_closed_requeue(
                repo=repo,
                event=event,
                review=review,
                pr_labels=pr_labels,
                reason="review.id missing from event payload",
                token=token,
            )
        try:
            review_comments = fetch_review_comments(
                repo, pr_number, review_id, token=token or "",
            )
        except ReviewCommentsTruncated as exc:
            # Codex PR #235 — P2 (round 2): hitting the pagination
            # safety cap means we cannot classify on complete data.
            # Fail closed to needs-greptile-audit, surfacing the
            # cause in the apply log.
            #
            # Codex PR #235 — P3 (round 3): the fallback must
            # respect the same author/opt-in gates that
            # `resolve_label_decision` enforces. Otherwise a
            # non-Greptile review or non-opted-in PR could
            # mutate labels on the truncation path.
            print(f"warn: review comments truncated — {exc}", file=sys.stderr)
            return _fail_closed_requeue(
                repo=repo,
                event=event,
                review=review,
                pr_labels=pr_labels,
                reason=f"pagination truncated: {exc}",
                token=token,
            )

    decision, reason = resolve_label_decision(
        event,
        pr_labels=pr_labels,
        current_head_sha=current_head_sha,
        review_comments=review_comments,
    )

    if decision is None:
        print(f"skip: {reason}")
        return 0

    if os.environ.get("DRY_RUN"):
        print(json.dumps({"decision": decision.__dict__, "reason": reason}, sort_keys=True))
        return 0

    # Greptile lane is INFORMATIONAL ONLY (per CLAUDE.md merge
    # policy). Label-application failures (e.g. PAT permission gap
    # on `POST /repos/.../issues/N/labels`) must not fail the
    # workflow check, otherwise PRs go UNSTABLE and the lane
    # becomes an accidental merge gate. `_apply_or_log` handles
    # the try/except + verdict logging.
    _apply_or_log(repo, decision, token=token or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
