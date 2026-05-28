# Codex audit protocol (merge authority — preferred)

## Status

**Merge authority granted, and preferred over Devin.** Codex runs as
a 3rd audit lane in parallel with Devin (cloud webhook) and Greptile
(SaaS GitHub App). Qwen (local LM Studio) was the original 2nd lane
but is currently **paused** — its calibration showed too many false
positives to justify the LM Studio runtime cost; see
`tools/QWEN_AUDIT_PROTOCOL.md` and the inert files under
`tools/qwen_audit_*.py` if reviving it.

Claude's standing in-thread merge delegation now accepts EITHER
`codex-audit-done` OR `devin-audit-done` + CLEAN, and **Codex is
preferred** — when both are available the Codex verdict is the
leading signal. Calibration on PR #233 (Phase 2B v2 audit of commit
`a2ddd70`) showed Codex catching 6 real bugs Devin's first-pass
PASSed; on this codebase Codex's findings are higher-signal.

Greptile remains strictly informational — never required for merge.

## Why a 3rd lane (originally the 4th)

PR #233 (Phase 2B v2) produced concrete calibration data across three
reviewers on the same buggy commit (`a2ddd70`):

| Reviewer | Real findings | False positives | Style |
|---|---|---|---|
| Codex | **6** (all confirmed by Devin) | 0 | Code tracing |
| Devin | 0 (PASS) → 6 confirmed on follow-up | 0 | Final-state QA |
| Qwen (paused) | 0 | 7 | Pattern matching |

Codex demonstrably catches logic bugs Devin misses. Devin is strongest
at "does this match the claims" verification. The two are complementary.
Qwen's calibration was too noisy to justify the LM Studio runtime cost;
the lane was paused — Greptile took its bake-off slot.

## Components

### 1. Reviewer wrapper + CLI: `tools/run_codex_audit_pr.sh`

Use the wrapper, not `python3 tools/codex_audit_pr.py` directly. The
wrapper selects a controlled Python interpreter for the audit script
itself, avoiding macOS/system-Python certificate-store failures before
the review even starts. It also creates a local active-audit lock via
`tools/audit_handoff_log.py`, so Claude and Codex refuse to start a
duplicate audit for the same repo/PR/head while another local process is
already running. It uses `<repo>/.venv/bin/python` when present, then
`CODEX_AUDIT_PYTHON`, then a venv discovered from
`CODEX_AUDIT_REPO_PATHS`; if none exists, it refuses to run rather than
silently falling back to ambient `python3`.

The underlying CLI reviews ONE PR end-to-end by invoking the
locally-installed Codex CLI
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
tools/run_codex_audit_pr.sh --repo jeffhuber/cube-two-view-debugger --pr 233
```

Add `--dry-run` to print the audit comment to stdout instead of posting.

Exit codes: 0 (success), 1 (error), 2 (stale head — caller may requeue),
20 (active matching audit already running).

### Local audit handoff log / duplicate guard

`tools/audit_handoff_log.py` stores local audit events and active locks
outside the repository, by default under:

```text
~/.cache/cube-agent-audits/events.jsonl
~/.cache/cube-agent-audits/locks/*.json
```

Set `AUDIT_HANDOFF_LOG_DIR=/path/to/dir` to override the location
(mainly for tests). Locks are keyed by lane + repo + PR + current head
SHA. `tools/run_codex_audit_pr.sh` creates a `codex-audit` lock before
running `codex_audit_pr.py`, appends a `started` event, and removes the
lock with a `finished` event on exit. If a matching lock exists and its
PID is still alive, the wrapper exits 20 instead of starting another
review.

Useful preflight:

```bash
tools/audit_handoff_log.py status --repo OWNER/REPO --pr N
```

This complements the chat visibility rule in `CLAUDE.md`: the log/lock
is the machine-readable local coordination record; the chat timestamp is
the human-readable handoff/return record.

### Python interpreter and venv injection

There are two separate Python controls:

1. `tools/run_codex_audit_pr.sh` controls the interpreter that runs the
   audit script itself. This prevents GitHub API calls from failing
   before review due to a broken ambient Python certificate store.
2. `--venv-path` / `CODEX_AUDIT_VENV_PATH` controls the Python PATH
   injected into the `codex review` subprocess inside the temporary PR
   worktree.

The `codex review` subprocess inherits the parent's PATH, which on a
typical macOS dev box resolves to a system anaconda Python instead of
the canonical `.venv/`. This causes per-pixel drift in any audited tool
whose output is sensitive to numpy/Pillow version
(`tools/build_oracle_rectified_faces.py`,
`tools/probe_center_color_phase_metric.py`, etc.), and surfaced as a
real Codex P3 finding on PR #262.

Auto-discovery: `audit_pr()` looks for `<local_repo>/.venv/bin/python`.
If present, prepends `<local_repo>/.venv/bin` to PATH and exports
`VIRTUAL_ENV` for the `codex review` subprocess. No-op for repos
without a `.venv/` (e.g. cube-snap, which uses npm/vitest for tests).

Explicit override (rarely needed):
- `--venv-path /path/to/venv` to pin a specific venv
- `--venv-path ""` to disable injection entirely
- `CODEX_AUDIT_VENV_PATH=...` env var (same semantics)

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
`CODEX_AUDIT_LABEL_TOKEN` repo secret. The token must be a fine-grained
PAT with **both**:

- **Issues**: read + write (to apply/remove labels)
- **Pull requests**: read (for the labeler's stale-SHA detection,
  which fetches `GET /repos/{repo}/pulls/{n}` to compare the
  reviewed SHA against current PR head)

Codex round 5 of #234 — P2 caught that an Issues-only PAT would
fail the `/pulls/{n}` fetch before any label is applied.

### 3. Labels

Three labels per repo (need to be created via `gh label create`
before the first audit lands — same colors as Devin for UI
consistency):

```bash
gh label create needs-codex-audit  --color FBCA04 --description "Current PR head SHA needs Codex review"
gh label create codex-audit-done   --color 0E8A16 --description "Codex reviewed current PR head SHA with no P0/P1/P2 findings"
gh label create codex-audit-blocked --color B60205 --description "Codex found P0/P1/P2 blocker findings"
```

## Shared event schema (audit + review log)

The shared local audit log at `~/.cache/cube-agent-audits/events.jsonl`
is a JSONL stream that BOTH agents (Claude and Codex) read and
write. Without a documented schema, each agent invents its own
verbs and the other agent's Monitor filter silently misses events
— this happened on 2026-05-25 when Claude's Monitor missed a
Codex audit because Codex's direct-write code path emitted
`"event": "review_requested"` while Claude's filter only matched
`"event": "finished"`. This section is the contract that prevents
that drift.

### Top-level fields (all events)

| Field | Type | Required | Notes |
|---|---|---|---|
| `event` | string | **yes** | One of the canonical verbs below. New verbs require a docs PR + Monitor-filter update on both sides. |
| `lane` | string | **yes** | One of the canonical lane values below. |
| `repo` | string | **yes** | `owner/name`, e.g. `jeffhuber/cube-snap`. |
| `pr` | int | **yes** | PR number. |
| `head` | string | **yes** | PR head SHA at the time of the event. `"unknown"` if not resolvable. |
| `actor` | string | recommended | `claude`, `codex`, or shell `$USER`. Helps disambiguate when the same agent runs under multiple identities. |
| `time` | object | auto | `{"utc": ISO8601, "pt": "YYYY-MM-DD HH:MM:SS PT"}` — added by `append_event` automatically. |
| `schemaVersion` | int | optional today, required at v2 | Reserved for breaking changes. Currently implicit v1; add `"schemaVersion": 2` when shape changes. |

### Canonical event verbs

| Verb | Fires when | Required extra fields | Optional extra fields |
|---|---|---|---|
| `started` | Audit run begins (wrapper creates a lock) | `lockId`, `pid`, `trigger`, `cwd`, `command`, `started` | — |
| `finished` | Audit or review completes | `lockId` (audits) OR `verdict` (reviews) | `status`, `exitCode`, `lock`, `notes` |
| `duplicate_refused` | Wrapper rejected a duplicate audit attempt (exit code 20) | `lockId`, `active` | — |
| `stale_lock_reaped` | Helper cleaned up an orphaned lock (process dead) | `lockId`, `stale` | — |
| `review_requested` | An agent has accepted a review request and is about to start (pre-audit signal Codex currently emits) | — | `trigger` |

**Adding a new verb**: docs PR to this file with the trigger + field requirements, mirrored in both repos. Monitor filters on both sides need to be updated to match — otherwise the new verb is invisible to the other agent. If you're writing a brand-new event type and want symmetric pickup, ALSO widen the canonical Monitor filter (`grep -E '"event": ?"(finished|duplicate_refused|review_requested)"'`) on both agents' setups.

### Canonical lane values

| Lane | Used by | Means |
|---|---|---|
| `codex-audit` | Codex via `tools/run_codex_audit_pr.sh` | Wrapper-driven automated `codex review` |
| `codex-review` | Codex via `tools/post_review.sh --lane codex-review` | Manual Codex cross-review of a Claude PR |
| `claude-review` | Claude via `tools/post_review.sh --lane claude-review` | Manual Claude cross-review of a Codex PR |
| `claude-audit` (reserved) | — | Reserved for a future Claude-side automated audit lane; not in use today |
| `devin-audit`, `greptile-audit` (reserved) | — | Paid lanes; events not currently written from local. Reserved so a future polling helper can record them without conflict. |

### Canonical verdict values

For events that carry a `verdict` field (currently `finished` from reviews):

| Verdict | Means | Side effect in `post_review.sh` |
|---|---|---|
| `pass` | No P0/P1/P2 findings | Routing label is removed |
| `blocked` | At least one P0/P1/P2 finding | Routing label is kept for follow-up |

Any value other than literal lowercase `pass` is treated as "keep the label" by `post_review.sh` (defense in depth — covers `concerns`, `deferred`, typos, etc.). See the verdict-gated label-removal regression tests in `tests/test_post_review.py` (ctvd) which pin this behavior.

### Recommended Monitor filter (catches all relevant events)

```bash
tail -F ~/.cache/cube-agent-audits/events.jsonl 2>/dev/null \
  | grep --line-buffered -E '"event": ?"(finished|duplicate_refused|review_requested)"'
```

This catches everything an agent on the other side typically wants to react to: audit completions (`finished` from `codex-audit` lane), review completions (`finished` from `*-review` lanes), duplicate-prevention notices, and pre-audit signals. The `started`/`stale_lock_reaped` verbs intentionally fall through — they're informational only.

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
tools/run_codex_audit_pr.sh --repo OWNER/REPO --pr N
```

Apply `needs-codex-audit` as a marker label when a PR is ready for
calibration. The CLI itself doesn't read the label — it's purely a
manual trigger for now. (A polling daemon would close the loop; not
built yet. The original `tools/qwen_audit_bridge.py` was that shape
for the now-paused Qwen lane, kept on disk as a starting point.)

## Differences from the other protocols

| Aspect | Devin | Codex | Greptile |
|---|---|---|---|
| Trigger | GitHub webhook when `needs-devin-audit` / `@devin audit` is used | Manual CLI invocation (local subprocess) | GitHub App review when `needs-greptile-audit` is used |
| Model | Devin's hosted (paid) | OpenAI Codex (paid per OpenAI pricing; uses user's `~/.codex/auth.json`) | Greptile's hosted (paid per review/tier) |
| Code access | Diff text via webhook | Real git worktree at head SHA | Diff + repo context via Greptile's GitHub App |
| Latency | 2–10 min typical | 5–10 min typical for `codex review` | 30s–2 min typical |
| Review style | Final-state QA + checklist | Whole-repo code tracing | Inline comments with severity badges (P0/P1/P2/P3) |
| Merge authority | YES | **YES — preferred** (per CLAUDE.md) | NO (informational only, never gating) |
| Mirror in both repos | Required (byte-identical) | Required (byte-identical) | Required (byte-identical) |
| Audit at a SHA different from HEAD | No (always current) | Yes (worktree at any SHA) | No (current PR head only) |

Qwen was the original calibration partner before being paused; see
`tools/QWEN_AUDIT_PROTOCOL.md` for that lane's design.

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
- `tools/GREPTILE_AUDIT_PROTOCOL.md` — the Greptile protocol this one
  parallels.
- `tools/qwen_audit_pr.py` / `tools/qwen_audit_labeler.py` — the
  original Qwen lane this one paralleled in design (now paused; files
  kept on disk so the lane is trivial to revive).
- `CLAUDE.md` "Devin PR audit routing" section — the merge-delegation
  contract that currently authorizes only Devin's labels.

## Why captured-PASS-via-dump is the operational norm (Codex CLI v0.133.0-alpha.1)

Empirical investigation (2026-05-27, 87 CLI-failure dumps captured during a
single multi-PR cube-snap session): the current Codex CLI release
deterministically routes the final-verdict marker to **stderr**, while the
verdict prose lands on **stdout**.

| Stat | Count | % |
|---|---|---|
| Total dumps | 87 | 100% |
| Stdout has column-0 `codex` marker | **0** | **0%** |
| Stdout missing marker | **87** | **100%** |
| Stderr has column-0 `codex` marker | 87 | 100% |

Sample dump header:

```
OpenAI Codex v0.133.0-alpha.1
--------
workdir: /private/var/folders/.../codex-audit-...
model: gpt-5.5
provider: openai
approval: never
sandbox: workspace-write [workdir, /tmp, $TMPDIR]
reasoning effort: xhigh
reasoning summaries: none
```

This is NOT intermittent (which is what the old "(the CLI flake mode)"
wording in CLAUDE.md implied) — it is consistent CLI behavior on this
release. The parser's stderr-fallback path is therefore the *primary*
signal source on this CLI version, not a rare workaround.

Implications:

- **Captured-PASS-via-dump is the operational norm for clean PASS
  audits**, not an exceptional case. The merge-auth path documented
  in CLAUDE.md ("Captured-PASS-via-dump counts as `codex-audit-done`")
  applies when an UNKNOWN-classified audit's captured stdout meets
  ALL of: (a) substantive Codex summary line, (b) zero
  `[P0]`/`[P1]`/`[P2]` finding bullets, AND (c) characteristic PASS
  verdict prose (e.g. "did not find any actionable regressions",
  "no introduced correctness issues", "no regressions introduced
  by this patch"). See `CLAUDE.md` for the authoritative checklist
  — this bullet only frames how often that path fires in practice
  on this CLI version. It does NOT apply to:
  - BLOCKED audits — stderr-fallback BLOCKED comments stay blocked
    and require fixing the findings or re-auditing on a new head.
  - UNKNOWN dumps where the captured stdout is empty, truncated,
    or lacks the substantive-PASS-prose signal — zero blocker
    bullets in those cases is the absence of a verdict, not
    evidence of a clean one. Re-run the audit.
- **A stable (non-alpha) Codex CLI release that emits the marker to
  stdout** would let the parser take the stdout-anchored path and skip
  the strict-shape validator entirely. If/when such a release is
  available, pinning to it would simplify the audit comments back to
  trailer-anchored `codex-audit-done` / `codex-audit-blocked` directly.
- **Upstream Codex CLI bug worth filing**: the column-0 `codex`
  final-verdict marker should land on the same stream as the verdict
  prose (stdout), not be split across streams. Once fixed, the
  stderr-fallback code path in `tools/codex_audit_pr.py` can be
  deprecated.

Sample data lives at `~/.cache/cube-agent-audits/cli-failures/` on the
audit-operator machine; the 87-dump corpus referenced here is from the
2026-05-27 cube-snap session ending ~23:00 PT.
