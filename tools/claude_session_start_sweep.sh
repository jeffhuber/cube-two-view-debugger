#!/usr/bin/env bash
# Session-start queue sweep. Fires on startup/resume/compact via .claude/settings.json hook.
#
# Why: Monitor tasks die when a Claude session is interrupted, resumed, or compacted.
# The new session starts with zero inherited background work, so any PR that landed in
# `needs-claude-review` (or any audit-log event) while the previous session was dead is
# invisible until I explicitly poll. This script polls automatically on every session start
# and prints the result as system-reminder context so I see it before my first action.
#
# Output shape: always prints the exact Monitor command to re-arm. If queue is non-empty,
# it also prints PR list per repo + last few audit-log events.
#
# Hard 3s wall-clock budget — never block session startup on flaky network / rate limits.

set -u
LOG="$HOME/.cache/cube-agent-audits/events.jsonl"

# Fast-fail silently if prereqs missing.
command -v gh >/dev/null 2>&1 || exit 0
command -v jq >/dev/null 2>&1 || exit 0

run_with_timeout() {
  seconds="$1"
  shift

  if command -v timeout >/dev/null 2>&1; then
    timeout "$seconds" "$@"
    return $?
  fi
  if command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$seconds" "$@"
    return $?
  fi

  "$@" &
  cmd_pid=$!
  (
    sleep "$seconds"
    kill "$cmd_pid" 2>/dev/null
  ) &
  watchdog_pid=$!
  wait "$cmd_pid"
  rc=$?
  kill "$watchdog_pid" 2>/dev/null
  wait "$watchdog_pid" 2>/dev/null || true
  return "$rc"
}

CS_TMP=$(mktemp -t cs-pending.XXXX)
CTVD_TMP=$(mktemp -t ctvd-pending.XXXX)
CS_STATUS=$(mktemp -t cs-status.XXXX)
CTVD_STATUS=$(mktemp -t ctvd-status.XXXX)
trap 'rm -f "$CS_TMP" "$CTVD_TMP" "$CS_STATUS" "$CTVD_STATUS"' EXIT

jq_count_or_zero() {
  count=$(jq 'length' < "$1" 2>/dev/null)
  case "$count" in
    ''|*[!0-9]*) echo 0 ;;
    *) echo "$count" ;;
  esac
}

fetch_queue() {
  repo="$1"
  out="$2"
  status="$3"
  tmp="$out.raw"

  if run_with_timeout 2 gh pr list --repo "$repo" --state open \
    --label needs-claude-review \
    --json number,headRefOid,updatedAt 2>/dev/null > "$tmp" &&
    jq empty "$tmp" >/dev/null 2>&1; then
    mv "$tmp" "$out"
    echo ok > "$status"
  else
    rm -f "$tmp"
    : > "$out"
    echo failed > "$status"
  fi
}

# Parallel fetch, 2s each. Fetch failures are reported as unknown, not empty.
{
  fetch_queue jeffhuber/cube-snap "$CS_TMP" "$CS_STATUS" &
  fetch_queue jeffhuber/cube-two-view-debugger "$CTVD_TMP" "$CTVD_STATUS" &
  wait
}

CS_FETCH=$(cat "$CS_STATUS" 2>/dev/null || echo failed)
CTVD_FETCH=$(cat "$CTVD_STATUS" 2>/dev/null || echo failed)
CS_COUNT=0
CTVD_COUNT=0
if [ "$CS_FETCH" = ok ]; then
  CS_COUNT=$(jq_count_or_zero "$CS_TMP")
fi
if [ "$CTVD_FETCH" = ok ]; then
  CTVD_COUNT=$(jq_count_or_zero "$CTVD_TMP")
fi
TOTAL=$((CS_COUNT + CTVD_COUNT))

# Cross-check the audit log for `review_requested` events that have NO
# later `finished` at the same head SHA. This catches the race window
# between Codex writing a review_requested event AND the GitHub label
# landing: gh pr list above only sees the label, so a freshly-requested
# review can be invisible to the gh-list call by tens of seconds even
# though the audit-log event is already there. The audit-log tail is the
# leading indicator; the GitHub label is the lagging one.
#
# Scan a 100-event window (≈ last hour of typical activity). For each
# review_requested in that window, check whether a subsequent finished
# event in the window has the same (repo, pr, head). If not, emit it as
# a pending entry.
#
# Worst-case work: 100-event jq pass. Cheap relative to the 2s gh
# budget already accepted above.
AUDIT_PENDING_TMP=$(mktemp -t audit-pending.XXXX)
trap 'rm -f "$CS_TMP" "$CTVD_TMP" "$CS_STATUS" "$CTVD_STATUS" "$AUDIT_PENDING_TMP"' EXIT
if [ -f "$LOG" ]; then
  tail -100 "$LOG" 2>/dev/null | jq -s -r '
    . as $events
    # Scope filter: the headline above is claude-review-scoped (the gh
    # pr list call uses --label needs-claude-review), so the audit-log
    # cross-check must match the same scope. Surfacing codex-audit /
    # codex-review / qwen-audit requests here would tell Claude to
    # review PRs that are in another lane entirely — false positives
    # caught by Codex P2 on snap#201 round 2.
    | [$events[] | select(.event == "review_requested"
                          and (.lane // "") == "claude-review")
       | {repo, pr,
          head: (.head // ""),
          time: (.time.pt // "—"),
          time_utc: (.time.utc // ""),
          lane: (.lane // "—")}] as $reviews
    | [$events[] | select(.event == "finished")
       | {repo: (.repo // .lock.repo // ""),
          pr: (.pr // .lock.pr // null),
          head: (.head // .lock.head // ""),
          # lane field. post_review.sh writes it at top level
          # (claude-review / codex-review); the codex-audit wrapper
          # run_codex_audit_pr.sh nests it inside .lock. Without the
          # .lock.lane fallback, every codex-audit finish would have
          # lane == "" and never match a claude-review request, so
          # all claude reviews would look perpetually pending. Caught
          # empirically on the snap#201 / ctvd#366 captured dumps.
          lane: (.lane // .lock.lane // ""),
          # UTC time of the finish event (top-level on both event
          # shapes). Used below to require finish AFTER the request
          # so an earlier-round finish on the same head does not
          # falsely clear a fresh request.
          time_utc: (.time.utc // "")}] as $finishes
    | $reviews
      | map(
          . as $r
          | select([
              $finishes[]
              # Three-part match for a "this review is done" signal.
              # NOTE: comments here cannot use apostrophes because the
              # entire jq program is a Bash single-quoted string and an
              # apostrophe inside it terminates the quote (Codex P1
              # caught the previous version which used wed/dont/etc).
              #
              # 1. Same (repo, pr, head[:7]) — the request and finish
              #    are about the same PR head. 7-char SHA prefix
              #    because the audit log mixes 40-char and 7-char
              #    head encodings (post_review.sh shortens; the
              #    wrappers do not); slicing beyond string length
              #    returns the string unchanged so [0:12] would
              #    falsely reject 40-char-vs-7-char matches.
              #
              # 2. Same lane — a codex-audit finish must NOT clear a
              #    claude-review request (the two lanes are
              #    independent workflows). Codex P2 on
              #    cube-snap#201 / ctvd#366: without the lane filter,
              #    the same head being audited would suppress a
              #    freshly-requested review on the same head,
              #    defeating the cross-check.
              #
              # 3. Finish time AFTER request time — chronological
              #    order matters. An earlier-round finish on the
              #    same (repo, pr, head, lane) MUST NOT clear a new
              #    request for that same head (e.g. when a fix is
              #    pushed-then-reverted-then-rerequested at the
              #    same SHA, or when the same head is requested
              #    twice for follow-up review). Codex P2 round-1
              #    on ctvd#366 / round-2 on snap#201. UTC ISO-8601
              #    is lexicographically sortable so a plain string
              #    compare works.
              | select(
                  .repo == $r.repo
                  and .pr == $r.pr
                  and (.head // "")[0:7] == ($r.head // "")[0:7]
                  and .lane == $r.lane
                  and .time_utc > $r.time_utc
                )
            ] | length == 0)
          | "\($r.repo)\t\($r.pr)\t\(($r.head // "")[0:12])\t\($r.lane)\t\($r.time)"
        )
      | .[]
  ' 2>/dev/null > "$AUDIT_PENDING_TMP"
fi
AUDIT_PENDING_COUNT=$(wc -l < "$AUDIT_PENDING_TMP" 2>/dev/null | tr -d ' ')
case "$AUDIT_PENDING_COUNT" in ''|*[!0-9]*) AUDIT_PENDING_COUNT=0 ;; esac

# Headline. The fetch-failure case still wins (we can't tell what's
# pending if gh is down). Otherwise: if EITHER the gh-list count OR
# the audit-log cross-check is non-zero, surface specific PR numbers in
# the headline itself so the model can't gloss over a count.
if [ "$CS_FETCH" != ok ] || [ "$CTVD_FETCH" != ok ]; then
  echo "[Claude queue sweep — queue fetch incomplete; re-arm Monitor and poll GitHub manually]"
elif [ "$TOTAL" -eq 0 ] && [ "$AUDIT_PENDING_COUNT" -eq 0 ]; then
  echo "[Claude queue sweep — no PRs awaiting Claude review at session start]"
else
  # Build a compact "snap#N, ctvd#M" list from gh-list + audit-log
  # cross-check, deduplicating by (repo, pr).
  HEADLINE_PRS=$(
    {
      [ "$CS_COUNT" -gt 0 ] && jq -r '.[] | "jeffhuber/cube-snap\t\(.number)"' < "$CS_TMP"
      [ "$CTVD_COUNT" -gt 0 ] && jq -r '.[] | "jeffhuber/cube-two-view-debugger\t\(.number)"' < "$CTVD_TMP"
      [ "$AUDIT_PENDING_COUNT" -gt 0 ] && awk -F'\t' '{print $1 "\t" $2}' < "$AUDIT_PENDING_TMP"
    } | sort -u | awk -F'\t' '{
        repo = $1; sub("jeffhuber/", "", repo)
        short = (repo == "cube-snap") ? "snap" : (repo == "cube-two-view-debugger" ? "ctvd" : repo)
        printf "%s%s#%s", (NR>1 ? ", " : ""), short, $2
      } END { print "" }'
  )
  echo "[Claude queue sweep — PENDING: $HEADLINE_PRS]"
fi
echo ""

if [ "$CS_FETCH" != ok ]; then
  echo "cube-snap: fetch failed or timed out"
fi
if [ "$CTVD_FETCH" != ok ]; then
  echo "cube-two-view-debugger: fetch failed or timed out"
fi

if [ "$CS_COUNT" -gt 0 ]; then
  echo "cube-snap (label needs-claude-review):"
  jq -r '.[] | "  - #\(.number) @ \(.headRefOid[0:7]) (updated \(.updatedAt))"' < "$CS_TMP"
fi

if [ "$CTVD_COUNT" -gt 0 ]; then
  echo "cube-two-view-debugger (label needs-claude-review):"
  jq -r '.[] | "  - #\(.number) @ \(.headRefOid[0:7]) (updated \(.updatedAt))"' < "$CTVD_TMP"
fi

if [ "$AUDIT_PENDING_COUNT" -gt 0 ]; then
  echo "Audit-log pending (review_requested with no matching finished):"
  awk -F'\t' '{
    repo = $1; sub("jeffhuber/", "", repo)
    short = (repo == "cube-snap") ? "snap" : (repo == "cube-two-view-debugger" ? "ctvd" : repo)
    printf "  - %s#%s @ %s lane=%s (review_requested at %s)\n", short, $2, $3, $4, $5
  }' < "$AUDIT_PENDING_TMP"
  echo "  ^ The audit-log event preceded the GitHub label flip — leading indicator."
fi

echo ""
if [ -f "$LOG" ]; then
  echo "Recent audit-log events (last 5):"
  tail -5 "$LOG" 2>/dev/null | jq -r '
    . as $event |
    ($event.lock // $event.active // $event.stale // {}) as $nested |
    "  \(($event.time.pt // $event.started.pt // $nested.started.pt // "—")) \(($event.actor // $nested.actor // "—")) \($event.event // "—") \((($event.repo // $nested.repo // "—") | sub("jeffhuber/"; "")))#\($event.pr // $nested.pr // "?") @ \((($event.head // $nested.head // "—")[0:7])) verdict=\($event.verdict // "—")"
  ' 2>/dev/null
fi

echo ""
echo "ACTION REQUIRED: previous session's Monitor is dead. Re-arm before processing the queue:"
echo ""
echo "  Monitor(persistent=true, timeout_ms=3600000, command='tail -F ~/.cache/cube-agent-audits/events.jsonl 2>/dev/null | grep --line-buffered -E '\\''\"event\":\\s*\"(finished|duplicate_refused|review_requested|stale_lock_reaped)\"'\\'')"
echo ""
echo "Then process each pending PR in order (oldest first). PR sweep at session start is "
echo "the durable fix for the recurring 'Monitor died, missed PR #N' failure mode."
