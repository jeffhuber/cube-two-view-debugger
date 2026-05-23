# Greptile audit protocol (parallel bake-off — informational only)

## Status

**Live on cube-snap (free OSS tier); ctvd not yet installed.**
Three-way bake-off with Devin + Codex during the next ~20 PRs (Qwen
was originally in the bake-off but is currently paused — see
`tools/CODEX_AUDIT_PROTOCOL.md` for the empirical reasoning).

**Informational only — strictly non-gating.** Claude's standing
in-thread merge delegation accepts `codex-audit-done` OR
`devin-audit-done` + CLEAN; Greptile's verdict never gates merge
or approval. The labeler treats label-application failures (e.g.
PAT permission gaps on `POST /repos/.../issues/N/labels`) as
non-fatal: it logs the verdict + the error to stderr and exits 0
so the workflow check stays green and the PR stays CLEAN. The
verdict is also visible in Greptile's own inline review comments
on the PR.

Greptile is a SaaS GitHub App. Once the App is installed on the repo,
Greptile auto-reviews every PR (~3 min latency) and posts findings as
inline review comments. This labeler reads those reviews and applies
`greptile-audit-{done,blocked,needs}` labels. Until the App is
installed, the labeler workflow doesn't fire (no events to react to).

## Why

Cost vs. quality trade-off:

| | Devin | Codex (local) | Qwen (paused) | Greptile |
|---|---|---|---|---|
| Cost | ~$500/mo serious use | paid per OpenAI pricing | free | $30/seat/mo (50 reviews) OR free for OSS w/ MIT |
| Where it runs | cloud | user's machine (Codex CLI subprocess) | user's machine (LM Studio) | cloud |
| Codebase context | what bridge sends | real git worktree at head SHA | what CLI sends (full files, post-v2) | **graph-indexed whole repo** |
| Convention learning | re-prompt each PR | re-prompt each PR | re-prompt each PR | claims to learn from review-comment history |
| Setup work for us | bridge + labeler + workflow | CLI + labeler + workflow | bridge + CLI + labeler + workflow | labeler only (no bridge — SaaS) |

The unique value vs. our existing two: graph-indexed cross-file context
(can catch "you changed X but didn't update its caller Y in another file"
without us packaging that context into the prompt) and learned conventions
from our existing ~200 Devin audit comments.

## Integration mechanics (confirmed)

Greptile is a GitHub App. Once installed, it auto-fires on every PR open
and push (no label trigger, no per-PR enable; ~3 min latency).

Greptile posts findings as a single **GitHub PR review** by
`greptile-apps[bot]` with `state: COMMENTED` (empty review body). The
review carries a `commit_id` field naming the SHA it reviewed.

The actual findings live as **inline review comments** on changed lines,
each starting with a severity badge encoded as an HTML img tag:

```
<a href="#"><img alt="P1" src="https://greptile-static-assets.s3.amazonaws.com/badges/p1.svg?v=7" align="top"></a>
**Round timer never reset for future proposals**
... finding details ... suggestion block ...
```

Severity (per Greptile docs / observed conventions):
- **P0** — critical (some installs reserve P0 for security / breaking)
- **P1** — blocker (correctness, real bug)
- **P2** — significant concern (non-blocking but worth fixing)
- **P3** — nit (style, naming, refactor opportunity)

The labeler treats **P0 and P1** as blockers, P2/P3 as concerns.

## Proposed architecture

**No bridge.** Greptile is SaaS; nothing to run on our side.

**One labeler workflow.** Fires on `pull_request_review` events from
`greptile-apps[bot]`. Per Codex's review feedback, the labeler has four
gating checks before flipping any label — these are intentionally
defensive because Greptile is a third-party surface and we don't want
silent false-PASS on format drift or stale reviews:

1. **Opt-in gate.** Only flip labels on PRs that currently carry the
   `needs-greptile-audit` label. Greptile will still auto-review every
   PR (we can't stop that without uninstalling the app), but the
   labeler ignores reviews on PRs we haven't asked to be audited.
   This keeps the bake-off scope controlled and avoids accidentally
   treating every random PR as part of the audit protocol.

2. **Stale-HEAD gate.** Compare `review.commit_id` to the PR's current
   head SHA. If they differ, the review is stale (head changed after
   Greptile reviewed). Keep / re-apply `needs-greptile-audit`,
   remove any prior `greptile-audit-{done,blocked}`. Same shape as
   the Devin labeler's head-SHA-mismatch logic.

3. **Severity parse, with fail-closed fallback.** Walk the review's
   inline comments. For each, extract severity from `alt="P[0-9]"` in
   the embedded `<img>` tag (preferred — more semantic than the
   badge URL, which could change with CDN versioning). Cross-check
   against the URL pattern `greptile-static-assets.../badges/p{N}.svg`
   if `alt` is missing. If a comment has neither marker, that comment
   is "format unknown" — fail closed: emit `needs-greptile-audit`
   rather than auto-PASS, because the comment format may have
   drifted and we shouldn't silently lose verdict signal.

4. **Verdict** (only reached if gates 1-3 pass cleanly):
   - any P0 or P1 finding → `greptile-audit-blocked`
   - zero P0/P1, but P2/P3 present → `greptile-audit-done` (concerns
     surfaced but non-blocking)
   - zero badges of any tier AND inline-comment count > 0 → fail
     closed (format unknown — same as gate 3)
   - zero badges AND zero inline comments → `greptile-audit-done`
     (Greptile's clean-review signal; needs fixture confirmation
     before we trust this final case — see "Fixture-driven dev"
     below)

**Comparison tool:** `tools/audit_bakeoff_compare.py` (new) pulls
Devin / Codex / Greptile verdicts from the last N closed PRs and
produces:

- 3-way agreement matrix
- per-reviewer "caught" / "missed" counts (where "ground truth" is the
  union of all reviewers' blockers, manually triaged)
- false-positive rates per reviewer

## Fixture-driven dev (Codex's pre-implementation gate)

Before writing the labeler, capture **real Greptile review payloads**
from a low-risk PR and commit them as test fixtures under
`tests/fixtures/greptile_reviews/`. Need at minimum these four shapes:

1. **No findings** — does Greptile post zero inline comments on a clean
   PR, or a summary "looks good" comment with no severity badge?
   This is the case the verdict-step gate-4 "zero badges" branch
   depends on, and it's the highest-risk unknown in the design.

2. **P1 / P0 finding** — confirms the badge-URL + `alt="..."` pattern
   matches what we coded against. P0 may not exist in practice;
   capture if available.

3. **P2 / P3-only findings** — confirms concerns-only path produces
   `greptile-audit-done`.

4. **Stale-SHA review** — push a commit after Greptile reviews;
   capture the older review's `commit_id` to verify the stale-head
   logic.

The labeler's unit tests should be fixture-driven from these real
payloads, not hand-constructed mocks. Same pattern as the existing
Devin labeler tests in `tests/fixtures/devin_comments/` (if those
exist) or the per-file Qwen response fixtures.

## Files to add

| File | Purpose |
|---|---|
| `tools/greptile_audit_labeler.py` | Parses Greptile's review, counts P1s, applies label. Same labeler-pattern shape as the Devin / Codex labelers. |
| `.github/workflows/greptile-audit-labeler.yml` | Triggers on `pull_request_review` from `greptile-apps[bot]`. |
| `tools/GREPTILE_AUDIT_PROTOCOL.md` | This document, finalized. |
| `tests/test_greptile_audit_labeler.py` | Fixture-driven unit tests on real Greptile comment bodies (with P1, P2, P3, mixed, none). |
| `tools/audit_bakeoff_compare.py` | 3-way verdict comparison tool. Standalone — runs on demand after the bake-off window. |

Mirror byte-identical across cube-snap + ctvd, per the existing
infra-mirror convention.

## Privacy / security tradeoff (explicit decision)

Greptile is cloud. Per the Greptile security page: code is stored on
encrypted servers, inference uses OpenAI / Anthropic APIs, embeddings
and docstrings are stored, and de-identified data may be used for
training and improvement. This is meaningfully different from local
Qwen (which never leaves the user's machine) and from Devin (which
sees only the diff text we explicitly send via the webhook bridge).

The privacy decisions to make explicit before installing:

- **cube-snap (MIT, public):** code is already public on GitHub. The
  marginal privacy loss from Greptile indexing it is small. Going
  ahead seems fine.
- **ctvd (private):** code goes to a third-party SaaS. The repo
  contains research notes + photo fixtures + diagnostic outputs — no
  customer data, no secrets, but is otherwise private. **This is a
  real tradeoff and should be opted into explicitly**, not inherited
  from the cube-snap decision. Skipping ctvd is a viable bake-off
  config (cube-snap-only sample is smaller but still informative).

## Setup status

Done:

1. ✅ **Greptile GitHub App installed on cube-snap** (free OSS tier —
   cube-snap is MIT-licensed, qualifies per "free for qualified
   non-commercial projects with MIT, Apache, or GPL licenses").
2. ✅ **Labels created** on cube-snap (`needs-greptile-audit` yellow,
   `greptile-audit-done` green, `greptile-audit-blocked` red — colors
   matching Devin / Codex for UI consistency). Also created on ctvd
   even though the App isn't installed there, so they're ready if the
   ctvd decision flips.
3. ✅ **Labeler shipped** with the 4 defensive gates (opt-in,
   stale-HEAD, severity-parse fail-closed, verdict). PR #145 + #145's
   bootstrap-guard follow-up + the `review.id`-missing fail-closed fix
   that Devin caught on the bootstrap PR.
4. ✅ **Test fixtures captured** in
   `tests/test_greptile_audit_labeler.py` — a real P1 finding body
   from `ssvlabs/ssv` PR #2835 plus synthetic variants for P0/P2/P3,
   no-marker, and pagination cases. The "fixture-driven dev" plan
   above is implemented in those tests.

Remaining:

1. **Decide ctvd** (see "Privacy / security tradeoff" above). Options:
   - Skip ctvd, run bake-off on cube-snap only (current default)
   - Pay $30/mo for the Pro seat (accept the cloud-storage tradeoff)
   - Apply for the early-stage startup 50% discount ($15/mo)

2. **Add `.greptile/rules.md`** with our review protocol (per Codex's
   original recommendation; not strictly required to use the lane,
   but improves signal):
   - focus on correctness, stale fixtures, generated-report
     consistency, missing tests, unsafe production behavior
   - ignore style-only nits unless they hide a bug
   - for geometry-sensitive PRs, require row-level baseline/diff
     artifacts (the regression-gate convention already documented in
     `CLAUDE.md`)
   - for production recognizer changes, flag any confident-wrong risk
   - same lane-discipline rules the Codex prompt enforces

3. **Update Claude's PR-creation routine** to apply
   `needs-greptile-audit` alongside `needs-devin-audit` and
   `needs-codex-audit` during the bake-off, so the agreement matrix
   has 3 rows per PR. **Reminder:** the labeler only flips labels
   when `needs-greptile-audit` is present (opt-in gate), so this
   step is what actually puts a PR into the bake-off.

4. **Build the comparison report** (`tools/audit_bakeoff_compare.py`)
   after 10–20 opted-in PRs have accumulated data.

## Bake-off decision criteria (after ~20 PRs)

Calibration plan (3 reviewers: Devin + Codex + Greptile):

Greptile graduates to merge-authority eligibility when:

- Agrees with Devin on every Devin-blocker (zero misses on real bugs)
- False-positive rate (Greptile blocks, Devin clears) is below ~20%
- Distinct value-add: catches at least one real bug that neither Devin
  nor Codex caught (otherwise it's redundant cost)

If Greptile graduates, the merge-delegation contract becomes
`devin-audit-done` OR `greptile-audit-done` (per-PR — whichever fires
first AND agrees with the Codex run).

If Codex ALSO graduates: Devin can be retired entirely.

## Open questions (resolve during fixture-capture phase)

1. **Does Greptile customization let us emit a trailer?** "Custom rules"
   feature lets us write what to flag in plain English, but unclear if
   output format itself is customizable. If yes, a
   `<!-- GREPTILE_AUDIT_STATE: ... -->` trailer would make the labeler
   bullet-proof against future format changes. If no, the
   `alt="P[N]"` + URL-pattern severity parse (with fail-closed
   fallback) is what we ship.

2. **Per-PR disable for sensitive PRs.** Per Codex's "opt-in only"
   gate, the labeler ignores PRs without `needs-greptile-audit`, so
   the bake-off scope is fully controlled. Open question:
   self-audit gotcha — when this very PR (or future Greptile-related
   PRs) gets reviewed by Greptile, will Greptile try to flag the
   `needs-greptile-audit`-detection prose in our own code as the
   thing it's looking for? Same shape as the Devin self-audit
   regression caught on cube-snap#130 / ctvd#118.

3. **Required-status-check behavior.** Need to confirm Greptile doesn't
   default to blocking merge via a required check (which would step on
   Devin's merge authority). Branch-protection settings check required
   after install.

4. **Clean-review signal.** What does Greptile do on a PR with no
   findings? Zero inline comments? Or a "looks good" summary comment
   without badges? The verdict-step gate-4 "zero badges + zero
   comments → done" branch depends on this. Fixture-capture phase
   confirms it.

## Risk profile

**Implementation risk: lowest of the three integrations.** No bridge.
No prompts. No local serving. The labeler is ~100 lines that walks
inline review comments, classifies severity, applies one label. With
the four defensive gates (opt-in, stale-HEAD, severity-parse fail-closed,
verdict), the labeler fails into `needs-greptile-audit` (the "we don't
know yet" state) on any format drift — never into silent PASS.

**Operational risk: medium.** Greptile is a third-party SaaS we don't
control. Format changes, billing surprises, security incidents, and
service outages are all real. The opt-in gate limits blast radius:
worst case, the labeler stops working on PRs that have
`needs-greptile-audit`, which leaves the PR's `needs-devin-audit` /
`needs-codex-audit` lanes intact.

**Privacy risk: medium for cube-snap (public anyway), real for ctvd
(private).** See "Privacy / security tradeoff" above. Listed as an
explicit decision rather than buried.

## See also

- `tools/devin_audit_*.py` — the Devin protocol this parallels
- `tools/codex_audit_*.py` + `tools/CODEX_AUDIT_PROTOCOL.md` — the
  Codex lane this runs alongside
- `tools/qwen_audit_*.py` — the original Qwen lane this paralleled in
  design (paused; files kept on disk so the lane is trivial to revive)
- `tools/audit_bakeoff_compare.py` (planned) — the 3-way comparison tool
