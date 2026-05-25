#!/usr/bin/env bash
# Post a cross-review (Claude→Codex or Codex→Claude) with the three
# discipline steps bundled in the right order:
#
#   1. Post the review comment via tools/safe_gh_comment.py (no shell
#      interpretation of Markdown).
#   2. Remove the routing label (needs-claude-review /
#      needs-codex-review) so the other agent's queue sweep notices.
#   3. Append a `finished` event to the shared local audit log
#      (~/.cache/cube-agent-audits/events.jsonl) so the other agent's
#      Monitor on that log catches the verdict in real time.
#
# The third step is what gives both agents symmetric ~0s fast-pickup on
# reviews — without it, only audit runs flow through the shared log and
# Claude reviews are invisible to a Codex-side Monitor (and vice versa).
# GitHub remains the source of truth; the log entry is a local-only
# fast-signal mirror.
#
# Usage:
#   tools/post_review.sh \
#     --lane claude-review \
#     --repo jeffhuber/cube-snap --pr 172 \
#     --head 6abe1c7 \
#     --verdict pass \
#     --label needs-claude-review \
#     --body-file /tmp/claude-review-snap172.md
#
# All four operations are independent gh + python calls — there's no
# atomic guarantee. If step 1 succeeds but step 2 or 3 fails, the
# comment is on the PR but the label / log is inconsistent. Re-run the
# remaining steps manually in that case.
#
# Lane names should match what the queue-sweep protocol expects:
#   claude-review        Claude cross-reviews a Codex-authored PR
#   codex-review         Codex manual cross-reviews a Claude-authored PR
#   codex-audit          Codex automated audit (set by the wrapper,
#                        not this script — use tools/run_codex_audit_pr.sh
#                        for that path)
#
# Mirror-invariant with the ctvd copy. Edits land in lockstep.

set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(CDPATH= cd -- "${script_dir}/.." && pwd)"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

LANE=""
REPO=""
PR=""
HEAD=""
VERDICT=""
LABEL=""
BODY_FILE=""
ACTOR="${AUDIT_ACTOR:-${USER:-unknown}}"

while [ $# -gt 0 ]; do
  case "$1" in
    --lane) LANE="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --pr) PR="$2"; shift 2 ;;
    --head) HEAD="$2"; shift 2 ;;
    --verdict) VERDICT="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --body-file) BODY_FILE="$2"; shift 2 ;;
    --actor) ACTOR="$2"; shift 2 ;;
    --help|-h)
      sed -n '1,/^set -/p' "$0" | sed -e 's/^# \{0,1\}//' -e '$d'
      exit 0
      ;;
    *) die "unknown arg: $1 (use --help for usage)" ;;
  esac
done

[ -n "${LANE}" ]      || die "--lane is required (e.g. claude-review, codex-review)"
[ -n "${REPO}" ]      || die "--repo is required (owner/name)"
[ -n "${PR}" ]        || die "--pr is required"
[ -n "${HEAD}" ]      || die "--head is required (current PR head SHA)"
[ -n "${VERDICT}" ]   || die "--verdict is required (pass, blocked, etc.)"
[ -n "${LABEL}" ]     || die "--label is required (label to remove, e.g. needs-claude-review)"
[ -n "${BODY_FILE}" ] || die "--body-file is required"
[ -r "${BODY_FILE}" ] || die "body file not readable: ${BODY_FILE}"

# Python selection mirrors tools/run_codex_audit_pr.sh: prefer
# <repo>/.venv/bin/python, then CODEX_AUDIT_PYTHON, then a venv found
# via CODEX_AUDIT_REPO_PATHS (entries are owner/repo:/path).
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
  [ -x "${python_bin}" ] || die "CODEX_AUDIT_PYTHON not executable: ${python_bin}"
elif python_bin="$(choose_python_from_repo_paths)"; then
  :
else
  die "no controlled Python found. Set CODEX_AUDIT_PYTHON or create ${repo_root}/.venv. Refusing to use ambient python3."
fi

# Step 1: post the review comment (no shell interpretation of Markdown).
"${python_bin}" "${script_dir}/safe_gh_comment.py" \
  --repo "${REPO}" --pr "${PR}" --body-file "${BODY_FILE}"

# Step 2: remove the routing label so the other agent's queue sweep
# notices the lane is done. If the label was already removed (e.g.
# someone else picked it up), `gh pr edit` returns nonzero — we treat
# that as non-fatal so step 3 still runs.
gh pr edit "${PR}" --repo "${REPO}" --remove-label "${LABEL}" \
  || printf 'warning: failed to remove label %s (may already be gone)\n' \
            "${LABEL}" >&2

# Step 3: log the event so the other agent's Monitor catches the
# verdict in real time. Lock-free `record` subcommand (added alongside
# this helper).
"${python_bin}" "${script_dir}/audit_handoff_log.py" record \
  --lane "${LANE}" \
  --repo "${REPO}" \
  --pr "${PR}" \
  --head "${HEAD}" \
  --event finished \
  --verdict "${VERDICT}" \
  --actor "${ACTOR}"

printf 'posted review: %s#%s @ %s verdict=%s lane=%s actor=%s\n' \
  "${REPO}" "${PR}" "${HEAD:0:7}" "${VERDICT}" "${LANE}" "${ACTOR}"
