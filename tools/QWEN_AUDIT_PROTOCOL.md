# Qwen audit protocol (parallel calibration phase)

## Status

**Informational only.** Qwen runs in parallel with Devin. Claude's
standing in-thread merge delegation authorizes merge on
`devin-audit-done` + CLEAN — it does NOT yet authorize merge on
`qwen-audit-done`.

## Goal

Calibrate whether a locally-served Qwen3-Coder-Next can replace Devin
as the trusted PR auditor for the merge-delegation contract. Costs
less, runs locally, faster turnaround — but unproven on this
codebase. Calibration window: ~10–20 PRs of parallel audits to
compare Qwen's verdicts to Devin's.

After calibration, the user decides whether to:

- promote Qwen to merge-authority (update `CLAUDE.md`'s standing
  delegation to read `devin-audit-done` OR `qwen-audit-done`), OR
- demote Devin to spot-check / cost-saving mode, OR
- retire one or the other entirely.

## Components

The review work lives in `tools/qwen_audit_pr.py` (CLI + Python
module). The daemon at `tools/qwen_audit_bridge.py` is a thin
label-state-machine scheduler on top of the CLI: it polls, dedupes by
head SHA, and shells out to the CLI for each PR that needs review.

This split lets Claude / Codex invoke the reviewer on demand for a
specific PR without waiting for the daemon's next poll — and keeps
all the review-quality logic (full-file-content context, per-file
chunking, three-tier severity, stale-HEAD detection) in one place.

### 1. Reviewer CLI: `tools/qwen_audit_pr.py`

Reviews ONE PR end-to-end. Per Codex's design feedback (cube-snap
follow-up to #142 / ctvd #230):

1. Fetches PR metadata + per-file diffs.
2. For each changed file, fetches the **full file content at the PR
   head SHA** (not just diff text — "diff-only review is too weak").
3. Per-file Qwen pass: prompt sees full file + diff hunks + PR
   title/body. Returns JSON `{"blockers": [...], "concerns": [...]}`.
4. Synthesis Qwen pass: gathers per-file findings + cross-cutting
   concerns (test coverage, doc-index, lane discipline) → final
   verdict.
5. Refetches head SHA at end. If it changed mid-review, emits the
   `needs-qwen-audit` (stale) trailer instead of PASS/BLOCKED.
6. Posts the comment with the authoritative trailer.

**Severity contract** (in the prompt):

- `BLOCKER` — must fix before merge (correctness, missing test for new
  behavior, schema break, secret, claim in PR body that can't be
  verified from visible code — Codex's "don't approve by vibes" rule)
- `CONCERN` — non-blocking observation (style, naming, refactor idea)
- `NONE` — no issues to report

Verdict logic: any BLOCKER (per-file OR cross-cutting) → BLOCKED.
Else → PASS. CONCERNs are surfaced in the comment body but don't gate.

CLI usage:
```bash
GITHUB_TOKEN=<bot_pat> python3 tools/qwen_audit_pr.py \
    --repo jeffhuber/cube-snap --pr 142
```

Add `--dry-run` to print the audit comment to stdout instead of
posting. Exit codes: 0 (success), 1 (error), 2 (stale head — caller
may requeue).

### 2. Polling daemon: `tools/qwen_audit_bridge.py`

Long-lived loop that polls GitHub for `needs-qwen-audit` labels and
delegates each new PR to `qwen_audit_pr.audit_pr()`. Dedupes by
`(repo, pr_number, head_sha)` so the same head doesn't trigger
twice. State persists in `~/.config/qwen-audit-bridge/state.json`
across restarts.

Run manually for one pass:
```bash
GITHUB_TOKEN=<bot_pat> python3 tools/qwen_audit_bridge.py --once
```

Run as a long-lived daemon (e.g., inside tmux or under launchd/systemd):
```bash
GITHUB_TOKEN=<bot_pat> python3 tools/qwen_audit_bridge.py
```

Required environment (shared between bridge and CLI):

| Variable | Default | Purpose |
|---|---|---|
| `GITHUB_TOKEN` | (none — required) | PAT under the bot account whose comments the labeler trusts |
| `QWEN_API_BASE` | `http://localhost:1234/v1` | OpenAI-compatible base URL. LM Studio default; for ollama use `http://localhost:11434/v1` |
| `QWEN_API_MODEL` | `qwen3-coder-next` | Model name |
| `QWEN_API_KEY` | `EMPTY` | Bearer for local serving (most servers accept any string) |
| `QWEN_AUDIT_REPOS` | `jeffhuber/cube-snap,jeffhuber/cube-two-view-debugger` | Comma-separated `owner/repo` list to poll |
| `QWEN_POLL_INTERVAL` | `60` | Seconds between polls |
| `QWEN_AUDIT_STATE_PATH` | `~/.config/qwen-audit-bridge/state.json` | Persisted dedupe state (per-head-SHA) |
| `QWEN_AUDIT_DRY_RUN` | unset | If set, logs but doesn't post |

### 2. Labeler: `tools/qwen_audit_labeler.py` + `.github/workflows/qwen-audit-labeler.yml`

Runs as a GitHub Action. Fires on `issue_comment` events authored by
the Qwen bot account, parses the trailer, and applies one of:

- `qwen-audit-done`
- `qwen-audit-blocked`
- `needs-qwen-audit` (re-queue if head SHA changed mid-review)

If `github.token` cannot label the bot's comments, configure a
`QWEN_AUDIT_LABEL_TOKEN` repo secret (fine-grained, Issues read/write).

### 3. Labels

Three labels per repo (need to be created via `gh label create`
before the first audit lands):

```bash
gh label create needs-qwen-audit  --color FBCA04 --description "Current PR head SHA needs Qwen review"
gh label create qwen-audit-done   --color 0E8A16 --description "Qwen reviewed current PR head SHA with no blockers"
gh label create qwen-audit-blocked --color B60205 --description "Qwen found blockers or could not complete review"
```

## Trailer protocol (authoritative)

The labeler treats one of these final-line trailers as authoritative
over any prose in the comment body:

```
<!-- QWEN_AUDIT_STATE: qwen-audit-done -->
<!-- QWEN_AUDIT_STATE: qwen-audit-blocked -->
<!-- QWEN_AUDIT_STATE: needs-qwen-audit -->
```

The third form is for cases where the head SHA changed mid-review —
Qwen detects via `HEAD_CHANGED_DURING_REVIEW: reviewed <old>, current <new>`
and emits the re-queue trailer.

If no trailer is present, the labeler falls back to prose patterns
like `Qwen Audit: PASS` / `Qwen Audit: BLOCKED`, but those are less
reliable. The audit prompt in `qwen_audit_bridge.py` instructs Qwen
to always include the trailer.

## Bot account setup

The labeler trusts comments from logins in the `QWEN_BOT_AUTHORS`
list (env var on the workflow; default: `qwen-audit-bot,qwen-audit-bot[bot]`).

Recommended setup:

1. Create a dedicated GitHub user account (e.g., `qwen-audit-bot`) —
   not the user's primary identity. Audit comments will be
   attributed to it, which keeps the GitHub UI clear about what's
   automated.
2. Generate a fine-grained PAT under that account with `repo` scope
   for both `cube-snap` and `cube-two-view-debugger`.
3. Use the PAT as `GITHUB_TOKEN` for the daemon.
4. Add the bot account as a collaborator on both repos with
   read access to issues + write access to comments (the
   collaborator + label workflow handles labels).

Alternative: use the user's own account during early calibration to
avoid the bot-account overhead. Set `QWEN_BOT_AUTHORS=jeffhuber` (or
your login) — at the cost of needing to be careful not to confuse
manual user comments with audit results. Not recommended past
calibration.

## Triggering an audit

To request a Qwen audit on a PR, apply the `needs-qwen-audit` label:

```bash
gh pr edit <N> --add-label needs-qwen-audit
```

For the calibration phase, **both** `needs-devin-audit` AND
`needs-qwen-audit` should be applied to every Claude PR so we can
compare verdicts. Update Claude's PR-creation routine to apply both
once this protocol lands.

## Comparing Qwen vs Devin

After calibration period, run `tools/qwen_devin_calibration_report.py`
(future) to produce an agreement matrix:

| | Devin PASS | Devin BLOCKED |
|---|---|---|
| **Qwen PASS** | agree (both clear) | Qwen missed a blocker |
| **Qwen BLOCKED** | Qwen false positive | agree (both blocked) |

Decision criterion: Qwen is ready for promotion when it agrees with
Devin on every Devin-blocker AND its false-positive rate is below
some threshold (TBD; probably 20%).

## Differences from the Devin protocol

| Aspect | Devin | Qwen |
|---|---|---|
| Trigger | GitHub webhook to cloud session | Polling daemon on user's machine |
| Model | Devin's hosted (paid) | Local Qwen3-Coder-Next (free per audit) |
| Latency | 2–10 min typical | TBD (depends on local serving + model size) |
| Merge authority | YES (per CLAUDE.md) | NO (calibration phase) |
| Mirror in both repos | Required (byte-identical) | Required (byte-identical) |
| Watchdog cron | 5-min schedule | Daemon poll loop is the cron equivalent |
| Re-trigger on label | `labeled needs-devin-audit` event | Daemon picks up label on next poll |

## Decommissioning

If calibration shows Qwen isn't ready, the cleanup is:

```bash
rm tools/qwen_audit_bridge.py tools/qwen_audit_labeler.py tools/qwen_audit_pr.py
rm .github/workflows/qwen-audit-labeler.yml
rm tools/QWEN_AUDIT_PROTOCOL.md
rm tests/test_qwen_audit_pr.py
gh label delete needs-qwen-audit
gh label delete qwen-audit-done
gh label delete qwen-audit-blocked
```

…in both repos.

## See also

- `tools/devin_audit_bridge.py` / `tools/devin_audit_labeler.py` — the
  Devin protocol this one parallels.
- `CLAUDE.md` "Devin PR audit routing" section — the merge-delegation
  contract that currently authorizes only Devin's labels.
- `COORDINATION.md` Decision Log entry on the standing in-thread merge
  delegation.
