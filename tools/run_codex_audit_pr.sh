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
token_from_stdin=""
# Filtered args = audit_args with --token-from-stdin (a wrapper-only flag)
# stripped. We pass --read-token-from-stdin to python instead.
filtered_args=()
idx=0
while [ "${idx}" -lt "${#audit_args[@]}" ]; do
  arg="${audit_args[${idx}]}"
  case "${arg}" in
    --repo)
      idx=$((idx + 1))
      repo_arg="${audit_args[${idx}]:-}"
      filtered_args+=("${arg}" "${audit_args[${idx}]:-}")
      ;;
    --repo=*)
      repo_arg="${arg#--repo=}"
      filtered_args+=("${arg}")
      ;;
    --pr)
      idx=$((idx + 1))
      pr_arg="${audit_args[${idx}]:-}"
      filtered_args+=("${arg}" "${audit_args[${idx}]:-}")
      ;;
    --pr=*)
      pr_arg="${arg#--pr=}"
      filtered_args+=("${arg}")
      ;;
    --repo-paths)
      idx=$((idx + 1))
      repo_paths_arg="${audit_args[${idx}]:-}"
      filtered_args+=("${arg}" "${audit_args[${idx}]:-}")
      ;;
    --repo-paths=*)
      repo_paths_arg="${arg#--repo-paths=}"
      filtered_args+=("${arg}")
      ;;
    --help|-h)
      help_requested=1
      filtered_args+=("${arg}")
      ;;
    --token-from-stdin)
      # Wrapper-only flag: tells the wrapper the caller is piping the
      # token via stdin. Strip it before passing to python (python uses
      # its own --read-token-from-stdin which the wrapper adds).
      token_from_stdin=1
      ;;
    *)
      filtered_args+=("${arg}")
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

# Token-resolution decision tree. Three modes:
#
# 1. --token-from-stdin (RECOMMENDED): caller pipes the token in
#    via stdin. The wrapper never touches GITHUB_TOKEN in env, never
#    has it in argv. No process in the wrapper's tree exposes the
#    token via /proc/<pid>/environ. This is the only mode that
#    fully isolates from the /proc-environ exposure path on Linux.
#
# 2. GITHUB_TOKEN env var (legacy / backward-compat): caller has
#    GITHUB_TOKEN exported. The wrapper captures, unsets from env,
#    pipes to python via stdin. BUT the wrapper's own
#    /proc/<bash_pid>/environ block was captured at exec time with
#    GITHUB_TOKEN in it; `unset` doesn't scrub that. On Linux,
#    PR-controlled code in the codex review subprocess can read
#    /proc/<wrapper_bash_pid>/environ and recover the token for the
#    full lifetime of the audit. We print a warning to stderr.
#    Same applies recursively to the user's interactive shell
#    above the wrapper — outside the wrapper's scope.
#
# 3. --help / -h: no token needed; argparse prints usage and exits.
#
# Modes 2 + 3 deliberately do NOT auto-source via `gh auth token`:
# that would put a write-capable credential into a process whose
# subprocess (codex review) executes PR-controlled tooling. The
# explicit-opt-in design makes credential sharing the caller's
# deliberate choice, not the wrapper's default.
#
# Codex P1 audits on cube-snap#194/195 and cube-two-view-debugger
# #354 caught successive /proc-exposure layers (subprocess env,
# Python's initial env, wrapper bash's initial env). Mode 1 is the
# only one that fully closes that chain *for this wrapper's
# process tree*. Closing the user's interactive shell's exposure
# is outside this wrapper's scope (would require user to invoke
# from a fresh subshell with a sanitized env).
if [ -n "${help_requested}" ]; then
  # --help: skip all token handling, let python show usage.
  "${python_bin}" "${audit_script}" "${filtered_args[@]}"
elif [ -n "${token_from_stdin}" ]; then
  # Mode 1: caller pipes the token in. Wrapper passes through
  # stdin unmodified; python reads first line as token.
  # We do NOT inspect or rewrite stdin, do NOT touch GITHUB_TOKEN
  # in env (which should not be set in this mode anyway — if it
  # is, that's the caller's choice and the legacy /proc exposure
  # applies to whatever exported it).
  "${python_bin}" "${audit_script}" --read-token-from-stdin "${filtered_args[@]}"
elif [ -n "${GITHUB_TOKEN:-}" ]; then
  # Mode 2: legacy env-var path. WARN about /proc exposure, then
  # do best-effort scrubbing + pipe to python.
  printf 'warning: GITHUB_TOKEN-in-env mode exposes the token via\n' >&2
  printf '  /proc/<bash_pid>/environ for the lifetime of this audit.\n' >&2
  printf '  For full isolation, use:\n' >&2
  printf '    printf %%s "$(gh auth token)" | %s --token-from-stdin %s\n' "$0" "$*" >&2
  printf '\n' >&2
  _token_to_pass="${GITHUB_TOKEN}"
  unset GITHUB_TOKEN
  unset GH_TOKEN  # strip alt-name for completeness
  printf '%s' "${_token_to_pass}" | "${python_bin}" "${audit_script}" --read-token-from-stdin "${filtered_args[@]}"
else
  # Mode 4: no token, no --help — error out with actionable message.
  printf 'error: no GitHub token provided.\n' >&2
  printf '\n' >&2
  printf '  Preferred (fully isolates from /proc/<pid>/environ exposure):\n' >&2
  printf '    printf %%s "$(gh auth token)" | %s --token-from-stdin %s\n' "$0" "$*" >&2
  printf '\n' >&2
  printf '  Or legacy env-var (warns about /proc exposure but works):\n' >&2
  printf '    GITHUB_TOKEN="$(gh auth token)" %s %s\n' "$0" "$*" >&2
  printf '\n' >&2
  printf '  To inspect CLI options without a token, use --help or -h.\n' >&2
  exit 1
fi
