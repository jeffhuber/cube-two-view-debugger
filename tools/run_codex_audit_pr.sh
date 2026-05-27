#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(CDPATH= cd -- "${script_dir}/.." && pwd)"
audit_script="${script_dir}/codex_audit_pr.py"
handoff_log_script="${script_dir}/audit_handoff_log.py"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

# Parse CLI args FIRST so Python discovery (below) can fall back to the
# --repo-paths CLI arg when CODEX_AUDIT_REPO_PATHS env var isn't set and
# the script's own repo lacks .venv (the common case: invoking the wrapper
# from cube-snap to audit a cube-two-view-debugger PR, where ctvd has the
# venv but cube-snap doesn't).
audit_args=("$@")
repo_arg=""
pr_arg=""
repo_paths_arg=""
help_requested=""
idx=0
while [ "${idx}" -lt "${#audit_args[@]}" ]; do
  arg="${audit_args[${idx}]}"
  case "${arg}" in
    --repo)
      idx=$((idx + 1))
      repo_arg="${audit_args[${idx}]:-}"
      ;;
    --repo=*)
      repo_arg="${arg#--repo=}"
      ;;
    --pr)
      idx=$((idx + 1))
      pr_arg="${audit_args[${idx}]:-}"
      ;;
    --pr=*)
      pr_arg="${arg#--pr=}"
      ;;
    --repo-paths)
      idx=$((idx + 1))
      repo_paths_arg="${audit_args[${idx}]:-}"
      ;;
    --repo-paths=*)
      repo_paths_arg="${arg#--repo-paths=}"
      ;;
    --help|-h)
      help_requested=1
      ;;
  esac
  idx=$((idx + 1))
done

choose_python_from_entries() {
  local entries entry path candidate
  entries="$1"
  [ -n "${entries}" ] || return 1

  IFS=',' read -r -a repo_entries <<< "${entries}"
  for entry in "${repo_entries[@]}"; do
    path="${entry#*:}"
    [ "${path}" != "${entry}" ] || continue
    candidate="${path}/.venv/bin/python"
    if [ -x "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

if [ -x "${repo_root}/.venv/bin/python" ]; then
  python_bin="${repo_root}/.venv/bin/python"
elif [ -n "${CODEX_AUDIT_PYTHON:-}" ]; then
  python_bin="${CODEX_AUDIT_PYTHON}"
  [ -x "${python_bin}" ] || die "CODEX_AUDIT_PYTHON is not executable: ${python_bin}"
elif python_bin="$(choose_python_from_entries "${CODEX_AUDIT_REPO_PATHS:-}")"; then
  printf 'warning: %s has no local .venv/bin/python; using %s from CODEX_AUDIT_REPO_PATHS env\n' \
    "${repo_root}" "${python_bin}" >&2
elif python_bin="$(choose_python_from_entries "${repo_paths_arg}")"; then
  printf 'warning: %s has no local .venv/bin/python; using %s from --repo-paths CLI arg\n' \
    "${repo_root}" "${python_bin}" >&2
else
  die "no controlled Python found. Create ${repo_root}/.venv, set CODEX_AUDIT_PYTHON=/path/to/venv/bin/python, pass --repo-paths <owner/repo>:<path-with-.venv>, or set CODEX_AUDIT_REPO_PATHS in the env. Refusing to use ambient python3."
fi

lock_id=""
if [ -n "${repo_arg}" ] && [ -n "${pr_arg}" ]; then
  set +e
  lock_id="$(
    "${python_bin}" "${handoff_log_script}" start \
      --lane codex-audit \
      --repo "${repo_arg}" \
      --pr "${pr_arg}" \
      --trigger "tools/run_codex_audit_pr.sh" \
      --pid "$$" \
      --cwd "${PWD}" \
      --command "$0 $*"
  )"
  lock_rc=$?
  set -e
  if [ "${lock_rc}" -ne 0 ]; then
    exit "${lock_rc}"
  fi

  finish_lock() {
    # CRITICAL: capture $? in the SAME statement as `local`, not afterward.
    # `local rc; rc=$?` always captures 0 because `local` is a builtin that
    # succeeds, resetting $? before we read it. That bug caused exit code 1
    # from `codex_audit_pr.py` (missing --repo-paths, etc.) to be silently
    # logged as exitCode=0/status=completed and the wrapper to exit 0 — the
    # canonical "success and failure look the same" failure mode. Caught on
    # cube-snap#184 where three audit runs all reported success but never
    # posted a GitHub comment because Python exited 1 each time. See
    # ~/.claude/projects/-Users-jhuber-cube-snap/memory/feedback_silent_success_silent_failure.md.
    local rc=$?
    local status="completed"
    if [ "${rc}" -ne 0 ]; then
      status="failed"
    fi
    "${python_bin}" "${handoff_log_script}" finish \
      --lock-id "${lock_id}" \
      --status "${status}" \
      --exit-code "${rc}" >/dev/null || true
    exit "${rc}"
  }
  trap finish_lock EXIT
fi

# codex_audit_pr.py requires GITHUB_TOKEN for the GitHub API calls
# (PR metadata fetch, comment post). Fail with an actionable
# message rather than letting the Python script die later with a
# generic "env var required" error.
#
# Placed AFTER the lock/trap setup above (Codex P2 audit on
# cube-two-view-debugger#354): if this early-exit ran before the
# trap, the failure would not be logged to events.jsonl and the
# existing test_wrapper_propagates_python_failure_exit_code_to_audit_log
# regression test would break. Now the trap catches `exit 1` here
# and writes a `finished` event with status='failed'.
#
# Deliberately NOT auto-sourcing via `gh auth token`: that would
# put a write-capable GitHub credential into a process whose
# subprocesses include `codex review` running PR-controlled
# tooling. Even with the codex_audit_pr.py env sanitization
# stripping GITHUB_TOKEN/GH_TOKEN from the subprocess env, the
# subprocess inherits HOME and PATH and can recover the token by
# invoking `gh auth token` itself or by reading `gh`'s credential
# store directly. The cleanest mitigation is to force callers to
# explicitly opt in by setting GITHUB_TOKEN in their own env, so
# the credential-sharing decision is deliberate. See Codex P1
# audits on cube-snap#194 and cube-two-view-debugger#354 for the
# full reasoning behind reverting the auto-fallback.
if [ -z "${help_requested}" ] && [ -z "${GITHUB_TOKEN:-}" ]; then
  printf 'error: GITHUB_TOKEN env var is required.\n' >&2
  printf '\n' >&2
  printf '  To use your local gh CLI credential (one-shot):\n' >&2
  printf '    GITHUB_TOKEN="$(gh auth token)" %s %s\n' "$0" "$*" >&2
  printf '\n' >&2
  printf '  Or export once for the shell session:\n' >&2
  printf '    export GITHUB_TOKEN="$(gh auth token)"\n' >&2
  printf '\n' >&2
  printf '  (Not auto-sourced: would expose the credential to the\n' >&2
  printf '   untrusted codex review subprocess. See commit message\n' >&2
  printf '   on this script for details.)\n' >&2
  printf '\n' >&2
  printf '  To inspect CLI options without a token, use --help or -h.\n' >&2
  exit 1
fi

# Capture GITHUB_TOKEN/GH_TOKEN into local bash variables and remove
# them from THIS shell's environment. Without this, the audit
# subprocess (codex review running PR-controlled tooling) can walk up
# the process tree via /proc/<bash_pid>/environ on Linux (or
# equivalent same-user introspection) and recover the credential
# despite Python's _build_subprocess_env sanitization. Codex P1 audit
# on cube-snap#194 and cube-two-view-debugger#354 caught this.
#
# We re-export the token only as an inline-scoped env var on the
# python invocation below, so the python child sees it briefly until
# its own _extract_and_clear_github_token() pops it from os.environ.
# After that point, neither this shell's env nor python's env contain
# the token, and the codex subprocess walking up either pid finds
# nothing. The user's interactive shell may still hold the token —
# that's the user's deliberate export and outside this wrapper's
# scope to clear.
_token_to_pass=""
if [ -n "${GITHUB_TOKEN:-}" ]; then
  _token_to_pass="${GITHUB_TOKEN}"
  unset GITHUB_TOKEN
fi
unset GH_TOKEN  # strip alt-name even if unused, for completeness

if [ -n "${_token_to_pass}" ]; then
  GITHUB_TOKEN="${_token_to_pass}" "${python_bin}" "${audit_script}" "${audit_args[@]}"
else
  # No token in env — this is the --help/-h bypass path. Let python
  # show its argparse usage; no token needed for that.
  "${python_bin}" "${audit_script}" "${audit_args[@]}"
fi
