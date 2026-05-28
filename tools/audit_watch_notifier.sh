#!/usr/bin/env bash
# audit_watch_notifier.sh — long-running tailer that posts macOS
# notifications on each `review_requested` event in the shared local
# audit log. Designed to be started by a launchd agent (see
# tools/install_audit_watch_agent.sh) so the operator sees pending
# Claude reviews even when no Claude Code session is open.
#
# Without this, a `review_requested` event written by Codex while no
# Claude session is running goes unobserved until the operator next
# resumes a session — the SessionStart sweep hook catches it at that
# point, but if the operator doesn't open Claude for hours, the PR
# sits unreviewed. This script closes that observability gap by
# surfacing pending reviews as macOS notifications the operator can
# act on out-of-band.
#
# Filter scope: only `lane=claude-review` events. Codex audits, Devin
# audits, and other lanes are ignored — this notifier is specifically
# for "Claude needs to review something."
#
# Mirror-invariant: snap + ctvd copies of this script must stay
# byte-identical. Verify with `diff` before changing either side.

set -euo pipefail

LOG_FILE="${HOME}/.cache/cube-agent-audits/events.jsonl"

# Ensure the file exists so `tail -F` doesn't fail at startup. The
# parent dir is created at audit-log first-write time, but we may be
# starting before the first audit ever ran.
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

# Skip everything that's already in the log — only notify on NEW
# events appended after the agent starts. Without this offset,
# every agent restart (e.g. after the user logs in fresh) would
# re-notify the entire event history.
start_offset=$(wc -c <"$LOG_FILE" | tr -d ' ')

# De-dupe: don't fire two notifications for the same
# (repo, pr, head) tuple in a row. Codex frequently re-requests
# review on the same PR after a force-push or wording-fix push;
# the rapid re-request shouldn't fire a second notification before
# the operator has a chance to act on the first.
last_key=""

# Use `tail -c +N` to seek to the byte offset, then `-F` to follow
# rotations + appends. `--line-buffered` on grep so events surface
# within ~100ms instead of pipe-buffered minutes.
exec tail -F -c "+$((start_offset + 1))" "$LOG_FILE" 2>/dev/null \
  | grep --line-buffered '"event":[[:space:]]*"review_requested"' \
  | while IFS= read -r line; do
    # Parse fields with grep -oE / sed so we don't depend on jq.
    repo=$(grep -oE '"repo":[[:space:]]*"[^"]+"' <<<"$line" \
      | sed -E 's/.*"repo":[[:space:]]*"([^"]+)".*/\1/')
    pr=$(grep -oE '"pr":[[:space:]]*[0-9]+' <<<"$line" \
      | sed -E 's/.*"pr":[[:space:]]*//')
    head=$(grep -oE '"head":[[:space:]]*"[^"]+"' <<<"$line" \
      | sed -E 's/.*"head":[[:space:]]*"([^"]+)".*/\1/')
    lane=$(grep -oE '"lane":[[:space:]]*"[^"]+"' <<<"$line" \
      | sed -E 's/.*"lane":[[:space:]]*"([^"]+)".*/\1/')

    # Scope to Claude review only.
    [[ "$lane" == "claude-review" ]] || continue

    # Dedup against the most-recent fired event.
    key="${repo}#${pr}@${head:0:12}"
    if [[ "$key" == "$last_key" ]]; then
      continue
    fi
    last_key="$key"

    repo_short=$(basename "$repo")
    head_short="${head:0:7}"

    # osascript notification. `Glass` is a built-in sound on macOS.
    # Title shows the verb so the lock-screen / Notification Center
    # entry is recognizable at a glance.
    osascript -e "display notification \"#${pr} on ${repo_short} (head ${head_short}) needs Claude review\" with title \"Claude review requested\" sound name \"Glass\""
  done
