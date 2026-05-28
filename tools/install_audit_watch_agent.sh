#!/usr/bin/env bash
# install_audit_watch_agent.sh — install (or re-install) the launchd
# agent that runs tools/audit_watch_notifier.sh in the background.
#
# Idempotent: safe to re-run after the notifier script changes. Will
# unload any prior version first, then re-load with the new plist.
#
# The plist is generated at install time rather than committed to the
# repo so the absolute path to the notifier script is correct for the
# current checkout location.
#
# After install:
#   - Check status:   launchctl list | grep cubesnap
#   - View stdout:    tail -f ~/.cache/cube-agent-audits/audit-watch.log
#   - View stderr:    tail -f ~/.cache/cube-agent-audits/audit-watch.err
#   - Uninstall:      launchctl unload ~/Library/LaunchAgents/com.cubesnap.audit-watch.plist
#
# Mirror-invariant: snap + ctvd copies of this script must stay
# byte-identical. Verify with `diff` before changing either side.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTIFIER_SCRIPT="${SCRIPT_DIR}/audit_watch_notifier.sh"
LABEL="com.cubesnap.audit-watch"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
LOG_DIR="${HOME}/.cache/cube-agent-audits"

if [[ ! -x "$NOTIFIER_SCRIPT" ]]; then
  if [[ -f "$NOTIFIER_SCRIPT" ]]; then
    chmod +x "$NOTIFIER_SCRIPT"
  else
    echo "ERROR: notifier script not found at $NOTIFIER_SCRIPT" >&2
    exit 1
  fi
fi

mkdir -p "${HOME}/Library/LaunchAgents" "$LOG_DIR"

# Unload any prior version. Ignore errors — the agent may not be
# loaded yet (first install) or may already have been removed.
launchctl unload "$PLIST_PATH" 2>/dev/null || true

# Write the plist with the resolved absolute path to the notifier.
cat >"$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${NOTIFIER_SCRIPT}</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/audit-watch.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/audit-watch.err</string>
</dict>
</plist>
PLIST

launchctl load "$PLIST_PATH"

cat <<EOF
Installed launchd agent: ${PLIST_PATH}
Notifier script:        ${NOTIFIER_SCRIPT}
Logs:                   ${LOG_DIR}/audit-watch.{log,err}

Verify it's running:
  launchctl list | grep cubesnap

Test it:
  echo '{"event":"review_requested","lane":"claude-review","repo":"jeffhuber/cube-snap","pr":999,"head":"deadbeefcafebabe1234567890abcdef12345678","time":{"pt":"test"}}' >> ~/.cache/cube-agent-audits/events.jsonl
  # You should see a macOS notification within ~1s.
EOF
