#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${RAILWAY_PROJECT_ID:-b1c7d2c0-6efd-4128-acd4-4f0975cc6e16}"
SERVICE="${RAILWAY_SERVICE:-ctvd-recognizer}"
ENVIRONMENT="${RAILWAY_ENVIRONMENT:-production}"
REF="${1:-origin/main}"
if [[ $# -gt 0 ]]; then
  shift
fi
MESSAGE="${*:-Deploy ctvd recognizer from clean worktree (${REF})}"

if ! command -v railway >/dev/null 2>&1; then
  echo "railway CLI is required but was not found on PATH" >&2
  exit 127
fi

repo_root="$(git rev-parse --show-toplevel)"
tracked_status="$(git -C "$repo_root" status --porcelain=v1 --untracked-files=no)"
if [[ -n "$tracked_status" ]]; then
  echo "Warning: tracked local changes exist in $repo_root." >&2
  echo "This script deploys the committed ref '$REF' from a temporary worktree, not local edits." >&2
fi

git -C "$repo_root" fetch --prune origin main

tmp_parent="$(mktemp -d "${TMPDIR:-/tmp}/ctvd-railway-deploy.XXXXXX")"
deploy_dir="$tmp_parent/worktree"
cleanup() {
  git -C "$repo_root" worktree remove --force "$deploy_dir" >/dev/null 2>&1 || true
  rm -rf "$tmp_parent"
}
trap cleanup EXIT

git -C "$repo_root" worktree add --detach "$deploy_dir" "$REF"

if [[ -e "$deploy_dir/.worktrees" ]]; then
  echo "Refusing to deploy: temporary worktree unexpectedly contains .worktrees/" >&2
  exit 1
fi

if [[ ! -f "$deploy_dir/railway.json" ]]; then
  echo "Refusing to deploy: railway.json is missing from $REF" >&2
  exit 1
fi

echo "Deploying $REF to Railway service '$SERVICE' in '$ENVIRONMENT' from $deploy_dir"
(
  cd "$deploy_dir"
  railway up \
    --project "$PROJECT_ID" \
    --service "$SERVICE" \
    --environment "$ENVIRONMENT" \
    --detach \
    --message "$MESSAGE"
)

echo
echo "After deploy, verify the latest deployment includes configFile=/railway.json and builder=NIXPACKS:"
echo "  railway deployment list --service $SERVICE --environment $ENVIRONMENT --json --limit 1"
echo "Then verify health:"
echo "  curl -fsS https://ctvd-recognizer-production.up.railway.app/api/diag"
