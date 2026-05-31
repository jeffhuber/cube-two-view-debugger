# Gitar audit lane

[Gitar](https://gitar.ai) is an AI code-review GitHub App that runs as an
**informational** review lane, a peer to Greptile. It never gates merge.

## What Gitar does in this repo

- Installed as a GitHub App on `cube-snap` and `cube-two-view-debugger`.
- **Auto-reviews every PR** on open and on each push (Gitar's default;
  there is no opt-in label). Review is **free** (the paid tier is its
  autofix / CI-resolution, which we keep disabled).
- Posts a single **dashboard issue-comment** whose verdict is an HTML
  `<kbd>` badge in the "Code Review" summary, e.g.
  `<summary><b>Code Review</b> <kbd>✅ Approved</kbd></summary>`.
- Severity taxonomy (per Gitar docs): Critical / Important / Suggestion,
  categorized Security / Bug / Performance / Edge Case / Code Quality.

## Advisory-only requirement (safety)

Gitar's GitHub App has write access to code, PRs, and workflows and can
auto-apply fixes, approve, and merge by default. That conflicts with the
project's invariants (humans are sole merge authority; agent comments are
advisory). Gitar MUST be kept comment-only via BOTH:

1. **Dashboard settings** (Gitar site → Configuration): Block merges
   "Never"; Auto-approve off; autofix / auto-merge disabled; "Enhance PR
   summaries" off (it edits PR descriptions); "Retry unrelated CI
   failures" off.
2. **Version-controlled backstop**: `.gitar/rules/advisory-only.md` in
   each repo restates the comment-only posture so it survives a dashboard
   change.

Verified on cube-snap #270 / ctvd #411: Gitar posted only an issue
comment, submitted no GitHub review/approval, and pushed no commits.

## State machine (labels)

- `needs-gitar-audit` — fail-closed marker (verdict present but
  unparseable) and a manual "look again" request.
- `gitar-audit-done` — Gitar's Code Review verdict is "Approved".
- `gitar-audit-blocked` — Gitar's verdict is present but not "Approved".

Informational only: a `gitar-audit-blocked` label does NOT block merge.
It surfaces that Gitar flagged something worth a human glance.

## Verdict contract (native `<kbd>` badge)

`tools/gitar_audit_labeler.py` classifies each Gitar comment from Gitar's
own Code Review `<kbd>` badge:

1. **Badge.** It parses the `<kbd>` badge that follows `Code Review</b>`.
   Exactly "Approved" (ignoring a leading emoji/whitespace) → done; any
   other non-empty badge → blocked (the safe direction). PR-controlled
   text echoed inside code fences/spans is blanked first so a quoted badge
   cannot be read as Gitar's own verdict.
2. **Fail closed.** A Code Review block whose badge is missing/empty or
   whose markup drifted → needs (rather than silently skipping).
3. **Skip.** A Gitar comment with no Code Review block (an in-progress
   "Reviewing your code" comment, a reply, a CI comment) never moves a
   label.

> Note: the exact non-"Approved" badge strings are confirmed as real
> non-approved verdicts are observed. Until then, non-"Approved" → blocked
> is the deliberate fail-closed default.
>
> We previously also accepted an optional `<!-- GITAR_AUDIT_STATE: ... -->`
> trailer Gitar could be instructed to emit, but Gitar does not honor that
> instruction in practice (confirmed across cube-snap #270/#271 and
> ctvd #411) and the trailer is PR-influenced content that was a recurring
> spoofing surface, so it was removed. A first-class verdict (read-only API
> field, or a verdict label decoupled from a code-host approval) would be
> better, but on the Pro plan the verdict API is Enterprise-gated and the
> native `gitar-approved` label is coupled to Gitar submitting a GitHub
> approval (conflicts with advisory-only). Badge parsing is the right
> integration for now.

## Trigger model

`.github/workflows/gitar-audit-labeler.yml` fires on `issue_comment`
(created / edited) and:

- checks out **default-branch** labeler code (a PR cannot modify the
  script that judges it),
- has a **bootstrap guard** (skips if the labeler is not yet on `main`),
- runs `tools/gitar_audit_labeler.py`.

There is **no `clear-stale-on-push` job** (unlike the Codex lane). That
job exists to protect Codex's merge-authority labels from going stale; the
Gitar lane is informational, so a momentarily stale label is harmless, and
Gitar re-reviews on every push (the labeler also fires on `edited`).

Label-application failures are non-fatal (the labeler logs the verdict and
exits 0), so the lane can never become an accidental merge gate.

## Bot identity

The labeler's `is_gitar_comment_author()` reads `GITAR_BOT_AUTHORS` (repo
variable, comma-separated; override is non-additive). Default accepts
`gitar-bot` / `gitar-bot[bot]` / `gitar-ai[bot]`. The author gate is
enforced in Python so the override works.

## Token

If GitHub's built-in workflow token cannot label the bot's comments, set
repo secret `GITAR_AUDIT_LABEL_TOKEN` to a fine-grained PAT with **Issues:
read + write**. The workflow falls back to `github.token` when absent.

## Mirror invariant

These files MUST stay byte-identical across `cube-snap` and
`cube-two-view-debugger`:

- `tools/gitar_audit_labeler.py`
- `tools/GITAR_AUDIT_PROTOCOL.md`
- `.github/workflows/gitar-audit-labeler.yml`

The `.gitar/rules/advisory-only.md` backstop is also byte-identical across
both repos.

Tests live in `tests/test_gitar_audit_labeler.py` (both repos,
byte-identical). Run with `.venv/bin/pytest tests/test_gitar_audit_labeler.py`.

## How this lane differs from Greptile

| | Greptile | Gitar |
|---|---|---|
| Event | `pull_request_review` | `issue_comment` |
| Verdict | per-line P-badges | dashboard `<kbd>` badge |
| Cost | paid (opt-in label) | free (act-on-all) |
| Gating | informational | informational |

Both are informational and never required for merge.
