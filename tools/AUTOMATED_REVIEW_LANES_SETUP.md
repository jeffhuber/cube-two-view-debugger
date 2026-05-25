# Automated Review Lanes — Setup + Ops Guide

This doc covers the auto-trigger GitHub Actions that fire when a PR
gets a review-request label applied. There are two lanes:

| Label | Action | Runner | Auth |
|---|---|---|---|
| `needs-claude-review` | `claude-review-action.yml` | GitHub-hosted (`ubuntu-latest`) | `ANTHROPIC_API_KEY` secret + Claude GitHub App |
| `needs-codex-audit` | `codex-review-action.yml` | Self-hosted Mac (`self-hosted, macos, codex`) | `codex` CLI session on the runner host |

Both mirror the protocol in `CLAUDE.md` (Claude cross-review lane,
Codex audit lane). The actions are the auto-trigger end; the existing
labeler workflows (`codex-audit-labeler.yml`, etc.) handle the
verdict-to-terminal-label flip.

**Mirror invariant:** both `.github/workflows/*-review-action.yml` and
this doc MUST stay byte-identical across `cube-snap` and
`cube-two-view-debugger`. Verify with `diff` before changing either
side; PRs land in lockstep.

---

## Initial setup

### Claude lane

1. **Get an Anthropic API key** at https://console.anthropic.com.
2. **Add `ANTHROPIC_API_KEY` repo secret** in each repo:
   - https://github.com/jeffhuber/cube-snap/settings/secrets/actions
   - https://github.com/jeffhuber/cube-two-view-debugger/settings/secrets/actions
3. **Install the Claude GitHub App** on both repos:
   https://github.com/apps/claude → Configure → select both repos.
   Required permissions: Contents (read), Issues (read+write), Pull
   requests (read+write).
4. **Verify** by applying `needs-claude-review` to a test PR. The
   action should fire within 30 seconds; check the Actions tab.

### Codex lane

1. **Register a self-hosted runner** on each repo:
   - Settings → Actions → Runners → New self-hosted runner → macOS
   - Run the registration command on the developer's Mac (typically
     `~/actions-runner/config.sh ...`)
   - Apply labels during config: `self-hosted, macos, codex`
   - **Run as a service** so it survives logout:
     `./svc.sh install && ./svc.sh start`
   - The runner shares the login user's environment, including the
     authenticated `codex` CLI session and `.venv` Python.
2. **Set up the runner's environment** so `tools/run_codex_audit_pr.sh`
   can find a controlled Python interpreter. Either:
   - Symlink or check out the repo at a path with a `.venv` (the
     script's auto-discovery works), OR
   - Set `CODEX_AUDIT_PYTHON=/path/to/venv/bin/python` in the runner's
     environment (e.g. `~/.actions-runner/.env` or wherever the runner
     loads env from), OR
   - Set `CODEX_AUDIT_REPO_PATHS=owner/repo:/path/to/repo,...`
3. **Verify** by applying `needs-codex-audit` to a test PR. The
   action queues until the runner picks it up (seconds when online,
   indefinite when offline).

---

## Cost model

- **Claude lane:** every review burns Anthropic API tokens. Typical
  review of a 200-line PR with `claude-opus-4-7` runs roughly $0.50–
  $1.50; with `claude-sonnet-4-6` roughly $0.10–$0.30. Adjust
  `--max-turns` in the workflow to bound cost per review (`20` is the
  current default).
- **Codex lane:** uses the developer's existing Codex CLI session —
  no extra cost per review beyond the subscription that already
  authenticates the local CLI.
- **GitHub Actions minutes:** Claude lane uses GitHub-hosted minutes
  (~2-5 min per review). Codex lane uses self-hosted minutes (free).

To keep cost predictable, apply review labels deliberately. The
existing `CLAUDE.md` Paid Review Budget Policy applies to these
labels too: don't fire reviews during normal iteration.

---

## Operations

### When Claude action posts nothing or fails

1. Check Actions tab → `Claude review action` → latest run.
2. Common failures:
   - **"Anthropic API key not configured"** → secret missing or
     mistyped. Re-add `ANTHROPIC_API_KEY` to repo secrets.
   - **"GitHub App not installed"** → install `apps/claude` on the
     repo (see Initial setup step 3).
   - **Empty/no comment posted** → check the action's logs for the
     model's final output; often a prompt-following failure. Adjust
     the `prompt` block in the workflow if recurring.
3. **Manual fallback:** there is none for the Claude side — Claude
   cross-review is otherwise done by a human invoking the CLI.

### When Codex action stays queued

1. Check the runner status: Actions → Runners → look for the
   `self-hosted macos codex` runner. Idle vs Offline.
2. If offline: the developer's Mac is asleep, logged out, or the
   runner service stopped. SSH in or wake the Mac; the runner
   reconnects automatically and picks up queued jobs.
3. **Manual fallback** (always works): run the audit locally from
   any checkout of the repo:
   ```
   bash tools/run_codex_audit_pr.sh \
     --repo jeffhuber/cube-two-view-debugger --pr 295 \
     --repo-paths jeffhuber/cube-two-view-debugger:/Users/jhuber/cube-two-view-debugger,jeffhuber/cube-snap:/Users/jhuber/cube-snap
   ```
   The existing labeler workflow flips the terminal label exactly
   the same way regardless of whether the script ran via the action
   or via CLI.

### When you want to skip the action for one PR

- Don't apply the trigger label. The actions only fire on
  `needs-claude-review` / `needs-codex-audit`.

### When you want to disable an action entirely

- Disable the workflow in Actions → workflow → "..." → Disable
  workflow. The label still gets applied by humans but no review
  fires. Re-enable when ready.

---

## Security notes

- Both actions use `pull_request_target` (not `pull_request`) so they
  can post comments on PRs from forks. **The Claude action checks out
  the PR head SHA at the labeled time**, which means a malicious PR
  could in principle modify the prompt before the action runs. The
  `if:` prefilter requires the label to be applied by someone with
  repo permissions (the `labeled` event's actor must be a maintainer),
  but this is worth understanding before turning on for a public-fork
  repo.
- The Codex action runs on a self-hosted runner with the developer's
  environment, including any credentials in `~/.zshrc`, etc. **Do not
  enable the Codex action on repos that accept PRs from untrusted
  contributors** without sandboxing the runner.
- Neither action uses `--no-verify` or bypasses pre-commit hooks
  during runs (they read code, post comments — they don't push
  commits).

---

## What this replaces

- **Claude lane:** previously manual. Codex (or a human) would
  apply `needs-claude-review` and either ping Claude directly in
  chat or wait for the next session-open. With this action, the
  label triggers an immediate auto-review.
- **Codex lane:** previously manual. A human would invoke
  `tools/run_codex_audit_pr.sh` after applying `needs-codex-audit`.
  With this action, the label triggers an immediate audit on the
  self-hosted runner — same script, same comment trailer, same
  labeler-driven verdict flip. Manual CLI invocation still works as
  the documented fallback when the runner is offline.

---

## Future work

- **Codex CLI in CI without a self-hosted runner.** OpenAI ships an
  official `openai/codex-action` (it doesn't exist as of 2026-05),
  swap the runner from `[self-hosted, macos, codex]` to
  `ubuntu-latest` and use the action instead of
  `tools/run_codex_audit_pr.sh` directly. The trailer / labeler
  contract stays unchanged.
- **Cost-tier the Claude model per PR size.** Currently always
  Opus 4.7. A small wrapper could pick Sonnet for tiny diffs and
  Opus for large ones. Probably not worth the complexity until
  costs become a real number.
- **Drift detection between lanes.** When Claude and Codex both
  review the same PR (rare today), have a meta-step compare verdicts
  and flag disagreement for human review.
