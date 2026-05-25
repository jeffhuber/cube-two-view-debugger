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

choose_python_from_repo_paths() {
  local entries entry path candidate
  entries="${CODEX_AUDIT_REPO_PATHS:-}"
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
elif python_bin="$(choose_python_from_repo_paths)"; then
  printf 'warning: %s has no local .venv/bin/python; using %s from CODEX_AUDIT_REPO_PATHS\n' \
    "${repo_root}" "${python_bin}" >&2
else
  die "no controlled Python found. Create ${repo_root}/.venv, set CODEX_AUDIT_PYTHON=/path/to/venv/bin/python, or include a repo with .venv in CODEX_AUDIT_REPO_PATHS. Refusing to use ambient python3."
fi

audit_args=("$@")
repo_arg=""
pr_arg=""
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
  esac
  idx=$((idx + 1))
done

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
    local rc status
    rc=$?
    status="completed"
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

"${python_bin}" "${audit_script}" "${audit_args[@]}"
