#!/usr/bin/env bash
set -euo pipefail

# Sanitized re-exec: if the caller invoked us with --token-from-stdin
# (or the python-name alias --read-token-from-stdin), AND GITHUB_TOKEN /
# GH_TOKEN are in our env, re-exec ourselves with those vars stripped.
#
# Why: bash's /proc/<pid>/environ on Linux exposes the *initial exec
# environment block* for the life of the process. `unset GITHUB_TOKEN`
# inside the script removes it from bash's variable table but NOT from
# that kernel-exposed block. A PR-controlled subprocess walked from
# `codex review` can ascend its ancestor chain (codex review → python →
# bash wrapper → ...) and `cat /proc/<bash_pid>/environ` to recover the
# token even though Python already popped it from os.environ.
#
# Re-exec replaces this bash process with a new one whose initial env
# does not contain those vars at all. CUBE_SNAP_AUDIT_SANITIZED guards
# against infinite re-exec loops. `exec` preserves stdin (the piped
# token) and argv, so the new bash sees the same invocation.
#
# This only runs in the --token-from-stdin code path because:
#  - Mode 2 (legacy GITHUB_TOKEN env) DELIBERATELY uses that env var;
#    sanitizing would break it. Mode 2 carries its own /proc warning.
#  - --help bypasses token handling entirely.
#  - No-token mode errors out before any subprocess is spawned.
#
# Codex P1 audit on cube-snap#195 / ctvd#354 caught the mixed-mode
# hole: --token-from-stdin + exported GITHUB_TOKEN still leaked via
# /proc. This is the structural fix.
if [ -z "${CUBE_SNAP_AUDIT_SANITIZED:-}" ] \
   && { [ -n "${GITHUB_TOKEN:-}" ] || [ -n "${GH_TOKEN:-}" ]; }; then
  _has_stdin_flag=""
  for _arg in "$@"; do
    case "${_arg}" in
      --token-from-stdin|--read-token-from-stdin)
        _has_stdin_flag=1
        break
        ;;
    esac
  done
  unset _arg
  if [ -n "${_has_stdin_flag}" ]; then
    exec env -u GITHUB_TOKEN -u GH_TOKEN CUBE_SNAP_AUDIT_SANITIZED=1 "$0" "$@"
  fi
  unset _has_stdin_flag
fi

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
    --token-from-stdin|--read-token-from-stdin)
      # Wrapper-only flag (with python-name alias --read-token-from-stdin
      # so users following the argparse `--help` output land in the same
      # mode). Tells the wrapper the caller is piping the token via
      # stdin. Strip it before passing to python — the wrapper always
      # adds `--read-token-from-stdin` itself in mode 1 below, so leaving
      # it in filtered_args would double-pass.
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
  #
  # By the time control reaches here, the sanitized re-exec at the
  # top of this script has already stripped GITHUB_TOKEN / GH_TOKEN
  # from the wrapper bash's initial environ block — so this exec
  # gives Python an env block free of the token. The `unset` calls
  # below are defense-in-depth: redundant after the re-exec, but
  # cheap, and they cover the (impossible-without-tampering)
  # scenario where the sentinel CUBE_SNAP_AUDIT_SANITIZED is set
  # while the tokens are also in env.
  unset GITHUB_TOKEN
  unset GH_TOKEN
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
