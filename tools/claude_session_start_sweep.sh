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

if [ "$CS_FETCH" != ok ] || [ "$CTVD_FETCH" != ok ]; then
  echo "[Claude queue sweep — queue fetch incomplete; re-arm Monitor and poll GitHub manually]"
elif [ "$TOTAL" -eq 0 ]; then
  echo "[Claude queue sweep — no PRs awaiting Claude review at session start]"
else
  echo "[Claude queue sweep — $TOTAL PR(s) awaiting Claude review at session start]"
fi
echo ""

if [ "$CS_FETCH" != ok ]; then
  echo "cube-snap: fetch failed or timed out"
fi
if [ "$CTVD_FETCH" != ok ]; then
  echo "cube-two-view-debugger: fetch failed or timed out"
fi

if [ "$CS_COUNT" -gt 0 ]; then
  echo "cube-snap:"
  jq -r '.[] | "  - #\(.number) @ \(.headRefOid[0:7]) (updated \(.updatedAt))"' < "$CS_TMP"
fi

if [ "$CTVD_COUNT" -gt 0 ]; then
  echo "cube-two-view-debugger:"
  jq -r '.[] | "  - #\(.number) @ \(.headRefOid[0:7]) (updated \(.updatedAt))"' < "$CTVD_TMP"
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
