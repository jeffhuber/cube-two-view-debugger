# Codex audit protocol (calibration phase — informational only)

## Status

**Informational only.** Codex runs as a 4th audit lane in parallel with
Devin (cloud webhook), Qwen (local LM Studio), and Greptile (SaaS GitHub
App). Claude's standing in-thread merge delegation authorizes merge on
`devin-audit-done` + CLEAN — it does NOT yet authorize merge on
`codex-audit-done`. The user may extend the delegation after enough
calibration data justifies it.

## Why a fourth lane

PR #233 (Phase 2B v2) produced concrete calibration data across three
reviewers on the same buggy commit (`a2ddd70`):

| Reviewer | Real findings | False positives | Style |
|---|---|---|---|
| Codex | **6** (all confirmed by Devin) | 0 | Code tracing |
| Devin | 0 (PASS) → 6 confirmed on follow-up | 0 | Final-state QA |
| Qwen | 0 | 7 | Pattern matching |

Codex demonstrably catches logic bugs the other two miss. Devin is
strongest at "does this match the claims" verification. The two are
complementary; Qwen's calibration is too noisy to add value yet.

## Components

### 1. Reviewer CLI: `tools/codex_audit_pr.py`

Reviews ONE PR end-to-end by invoking the locally-installed Codex CLI
(`/Applications/Codex.app/Contents/Resources/codex` by default).
Pipeline:

1. Resolve the LOCAL repo path for `owner/repo` from
   `CODEX_AUDIT_REPO_PATHS` env var. The CLI requires a real git
   checkout to run `codex review --base origin/main` against.
2. Fetch PR head SHA via GitHub API.
3. `git fetch origin pull/<N>/head` to ensure the SHA is local; create
   a temporary detached worktree at that SHA.
4. Run `codex review --base origin/main` from the worktree, capturing
   stdout/stderr.
5. Parse the output: extract the final verdict block (the last
   `\ncodex\n` marker in the log), count `[P0]/[P1]/[P2]/[P3]`
   severity tags, classify PASS vs BLOCKED.
6. Stale-head check: refetch PR head. If it changed mid-review, emit
   the `needs-codex-audit` (stale) trailer.
7. Post the parsed review as a PR comment with authoritative trailer.
8. Clean up the worktree.

CLI usage:
```bash
CODEX_AUDIT_REPO_PATHS=jeffhuber/cube-snap:/Users/jhuber/cube-snap,jeffhuber/cube-two-view-debugger:/Users/jhuber/cube-two-view-debugger \
GITHUB_TOKEN=<bot_pat> \
python3 tools/codex_audit_pr.py --repo jeffhuber/cube-two-view-debugger --pr 233
```

Add `--dry-run` to print the audit comment to stdout instead of posting.

Exit codes: 0 (success), 1 (error), 2 (stale head — caller may requeue).

### Severity policy (PASS / BLOCKED)

Codex tags findings inline with `[P0]`, `[P1]`, `[P2]`, `[P3]` brackets:

- **P0** — critical (security, broken main flow)
- **P1** — high (real correctness bug)
- **P2** — medium (correctness issue, should fix)
- **P3** — low (nit, style, doc)

This labeler treats **P0 / P1 / P2 as blockers**, P3 as concerns. This
matches the empirical severity breakdown on #233 where every P2
finding was a real bug worth fixing. You can override the policy by
forking the parser; the verdict logic lives in
`tools/codex_audit_pr.py::parse_codex_output`.

### 2. Labeler: `tools/codex_audit_labeler.py` + `.github/workflows/codex-audit-labeler.yml`

Runs as a GitHub Action. Fires on `issue_comment` events. The author
check is in Python (not the YAML's `if`) so the `CODEX_BOT_AUTHORS`
override actually works for early-calibration users who post under
their personal login. Applies one of:

- `codex-audit-done`
- `codex-audit-blocked`
- `needs-codex-audit` (re-queue if head SHA changed mid-review)

If `github.token` cannot label the bot's comments, configure a
`CODEX_AUDIT_LABEL_TOKEN` repo secret (fine-grained, Issues read/write).

### 3. Labels

Three labels per repo (need to be created via `gh label create`
before the first audit lands — same colors as Devin / Qwen for UI
consistency):

```bash
gh label create needs-codex-audit  --color FBCA04 --description "Current PR head SHA needs Codex review"
gh label create codex-audit-done   --color 0E8A16 --description "Codex reviewed current PR head SHA with no P0/P1/P2 findings"
gh label create codex-audit-blocked --color B60205 --description "Codex found P0/P1/P2 blocker findings"
```

## Trailer protocol (authoritative)

The labeler treats one of these final-line trailers as authoritative
over any prose in the comment body:

```
<!-- CODEX_AUDIT_STATE: codex-audit-done -->
<!-- CODEX_AUDIT_STATE: codex-audit-blocked -->
<!-- CODEX_AUDIT_STATE: needs-codex-audit -->
```

The third form is for cases where the head SHA changed mid-review.

Prose fallback (less reliable): `Codex Audit: PASS` or
`Codex Audit: BLOCKED`. Audit comments produced by the CLI always
include the trailer.

## Bot account setup

The labeler trusts comments from logins in the `CODEX_BOT_AUTHORS`
list (env var on the workflow; default: `codex-audit-bot,codex-audit-bot[bot]`).

For early calibration: set `CODEX_BOT_AUTHORS=<your-login>` as a repo
variable so the labeler accepts comments posted under your account.
This avoids the bot-account setup overhead during the calibration
phase. Not recommended past calibration (manual comments could get
classified as audit results).

For production: create a dedicated `codex-audit-bot` GitHub user
account, generate a fine-grained PAT with `repo` scope, set
`GITHUB_TOKEN=<bot_pat>` when invoking the CLI. Comments post under
that identity and the labeler picks them up.

## Triggering an audit

The CLI is invoked manually (no polling daemon yet). To audit a PR:

```bash
CODEX_AUDIT_REPO_PATHS=... GITHUB_TOKEN=... \
python3 tools/codex_audit_pr.py --repo OWNER/REPO --pr N
```

Apply `needs-codex-audit` as a marker label when a PR is ready for
calibration. The CLI itself doesn't read the label — it's purely a
manual trigger for now. (A polling daemon analogous to
`qwen_audit_bridge.py` would close the loop; not built yet.)

## Differences from the other protocols

| Aspect | Devin | Qwen | Codex |
|---|---|---|---|
| Trigger | GitHub webhook (cloud) | Polling daemon (local LM Studio) | Manual CLI invocation (local subprocess) |
| Model | Devin's hosted (paid) | Local Qwen3-Coder-Next (free) | OpenAI Codex (paid per OpenAI's pricing; uses user's `~/.codex/auth.json`) |
| Code access | Diff text via webhook | File contents via GitHub API at head SHA | Real git worktree at head SHA |
| Latency | 2–10 min typical | ~30s per file × N files (chunked) | 5–10 min typical for `codex review` |
| Review style | Final-state QA + checklist | Per-file LLM review with synthesis | Whole-repo code tracing |
| Merge authority | YES (per CLAUDE.md) | NO (calibration phase) | NO (calibration phase) |
| Mirror in both repos | Required (byte-identical) | Required (byte-identical) | Required (byte-identical) |
| Audit at a SHA different from HEAD | No (always current) | No (always current) | Yes (worktree at any SHA) |

## Calibration plan

After 10–20 PRs of Codex audits in parallel with Devin:

- If Codex's findings on Devin-PASS PRs are mostly real (>80% true positive
  rate when independently checked), promote Codex to merge-authority
  eligibility (update `CLAUDE.md` to also accept `codex-audit-done`).
- If Codex's false-positive rate is high, tighten the prompt or fall
  back to "Codex as iterative reviewer, Devin as final-state gate."
- Independent of merge authority, Codex is already valuable as an
  iterative-review tool BEFORE landing — catches issues that would
  otherwise reach Devin's review queue.

## Decommissioning

If Codex calibration doesn't pan out, cleanup is:

```bash
rm tools/codex_audit_pr.py tools/codex_audit_labeler.py
rm .github/workflows/codex-audit-labeler.yml
rm tools/CODEX_AUDIT_PROTOCOL.md
rm tests/test_codex_audit_pr.py
gh label delete needs-codex-audit
gh label delete codex-audit-done
gh label delete codex-audit-blocked
```

…in both repos.

## See also

- `tools/devin_audit_bridge.py` / `tools/devin_audit_labeler.py` — the
  Devin protocol this one parallels.
- `tools/qwen_audit_pr.py` / `tools/qwen_audit_labeler.py` — the Qwen
  protocol this one parallels.
- `tools/GREPTILE_AUDIT_PROTOCOL.md` (when landed) — the Greptile
  protocol this one parallels.
- `CLAUDE.md` "Devin PR audit routing" section — the merge-delegation
  contract that currently authorizes only Devin's labels.
