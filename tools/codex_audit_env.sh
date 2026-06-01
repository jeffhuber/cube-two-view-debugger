# Source this before running local structured Codex audits on macOS.
#
# The audit wrapper still refuses ambient python3; this file only gives
# agents an explicit, reviewable way to opt into the known-good local
# interpreter and Codex CLI paths.

if [ -z "${CODEX_AUDIT_PYTHON:-}" ] && [ -x /usr/bin/python3 ]; then
  export CODEX_AUDIT_PYTHON=/usr/bin/python3
fi

if [ -z "${CODEX_CLI_PATH:-}" ] && [ -x /Applications/Codex.app/Contents/Resources/codex ]; then
  export CODEX_CLI_PATH=/Applications/Codex.app/Contents/Resources/codex
fi
