#!/usr/bin/env bash
# Session-start queue sweep. Fires on startup/resume/compact via .claude/settings.json hook.
#
# Why: Monitor tasks die when a Claude session is interrupted, resumed, or compacted.
# The new session starts with zero inherited background work, so any PR that landed in
# `needs-claude-review` (or any audit-log event) while the previous session was dead is
# invisible until I explicitly poll. This script polls automatically on every session start
# and prints the result as system-reminder context so I see it before my first action.
#
# Output shape: silent if queue is empty (so it doesn't spam routine resumes); otherwise
# prints PR list per repo + last few audit-log events + the exact Monitor command to re-arm.
#
# Hard 3s wall-clock budget — never block session startup on flaky network / rate limits.

set -u
LOG="$HOME/.cache/cube-agent-audits/events.jsonl"

# Fast-fail silently if prereqs missing.
command -v gh >/dev/null 2>&1 || exit 0
command -v jq >/dev/null 2>&1 || exit 0

CS_TMP=$(mktemp -t cs-pending.XXXX)
CTVD_TMP=$(mktemp -t ctvd-pending.XXXX)
trap 'rm -f "$CS_TMP" "$CTVD_TMP"' EXIT

# Parallel fetch, 2s each. Empty file on timeout/failure is fine — jq length returns 0.
{
  timeout 2 gh pr list --repo jeffhuber/cube-snap --state open \
    --label needs-claude-review \
    --json number,headRefOid,title,updatedAt 2>/dev/null > "$CS_TMP" &
  timeout 2 gh pr list --repo jeffhuber/cube-two-view-debugger --state open \
    --label needs-claude-review \
    --json number,headRefOid,title,updatedAt 2>/dev/null > "$CTVD_TMP" &
  wait
}

CS_COUNT=$(jq 'length' < "$CS_TMP" 2>/dev/null || echo 0)
CTVD_COUNT=$(jq 'length' < "$CTVD_TMP" 2>/dev/null || echo 0)
TOTAL=$((CS_COUNT + CTVD_COUNT))

if [ "$TOTAL" -eq 0 ]; then
  # Silent on empty queue — no need to spam every session start with "nothing to do."
  exit 0
fi

echo "[Claude queue sweep — $TOTAL PR(s) awaiting Claude review at session start]"
echo ""

if [ "$CS_COUNT" -gt 0 ]; then
  echo "cube-snap:"
  jq -r '.[] | "  - #\(.number) @ \(.headRefOid[0:7]) — \(.title) (updated \(.updatedAt))"' < "$CS_TMP"
fi

if [ "$CTVD_COUNT" -gt 0 ]; then
  echo "cube-two-view-debugger:"
  jq -r '.[] | "  - #\(.number) @ \(.headRefOid[0:7]) — \(.title) (updated \(.updatedAt))"' < "$CTVD_TMP"
fi

echo ""
if [ -f "$LOG" ]; then
  echo "Recent audit-log events (last 5):"
  tail -5 "$LOG" 2>/dev/null | jq -r '"  \(.time.pt // "—") \(.actor // "—") \(.event // "—") \(.repo // "—" | sub("jeffhuber/"; ""))#\(.pr // "?") @ \((.head // "—")[0:7]) verdict=\(.verdict // "—")"' 2>/dev/null
fi

echo ""
echo "ACTION REQUIRED: previous session's Monitor is dead. Re-arm before processing the queue:"
echo ""
echo "  Monitor(persistent=true, timeout_ms=3600000, command='tail -F ~/.cache/cube-agent-audits/events.jsonl 2>/dev/null | grep --line-buffered -E '\\''\"event\":\\s*\"(finished|duplicate_refused|review_requested|stale_lock_reaped)\"'\\'')"
echo ""
echo "Then process each pending PR in order (oldest first). PR sweep at session start is "
echo "the durable fix for the recurring 'Monitor died, missed PR #N' failure mode."
