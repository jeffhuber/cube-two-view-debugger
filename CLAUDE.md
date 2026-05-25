# cube-two-view-debugger — Claude/Codex shared protocol

This file documents conventions and gotchas that affect anyone running
Claude Code (or Codex) against this repo. The recognizer code itself
is documented inline; this file is for procedural/tooling concerns
that aren't obvious from the source.

## Working with iPhone photos (EXIF gotcha — read before any visual debugging)

The corpus and ad-hoc debug photos live in `/Users/jhuber/Downloads/`
as iPhone JPEGs. They are stored as **landscape** pixels (4032×3024)
with `EXIF Orientation=6`, which tells viewers "rotate 90° CW to
display correctly." Preview, Photos, browsers, and the cube-snap UI
all respect this. **The Read tool does NOT** — it shows the raw stored
pixels, which means the cube appears rotated 90° CCW from what users
see in their viewer.

**This caused a real diagnostic bug on Set 39 (2026-05-12)**: Claude
"visually confirmed" image B had white on top when it actually had
yellow on top, because Claude was looking at sideways raw pixels.
Compounded by pixel-sampling code that was correctly EXIF-correcting
but sampling at the wrong y-coordinate (background area above the
cube, not the cube's top face). The user had to push back twice
before Claude caught it. See the [Set 39 thread on Issue
#30](https://github.com/jeffhuber/cube-two-view-debugger/issues/30)
for the full failure-mode discussion.

### Protocol — always do this before reading any photo

Use the helper:

```bash
.venv/bin/python tools/view_photo.py "/Users/jhuber/Downloads/Set 39 - B - white up IMG_7158.JPG"
# prints: /tmp/Set 39 - B - white up IMG_7158-corrected.jpg
```

Then Read the printed path, not the original.

Or inline if you're already in Python:

```python
from PIL import Image, ImageOps
ImageOps.exif_transpose(Image.open(path)).convert('RGB').save('/tmp/x.jpg', quality=85)
```

### For pixel sampling code, not just Read calls

`analyze_image` and the rest of the recognizer pipeline already apply
EXIF correctly via Pillow. But if you're writing AD-HOC pixel-sampling
code during debugging (e.g., "what color is at y=15% of this image?"),
**don't assume the cube starts at the top of the EXIF-corrected
frame**. The cube silhouette is typically in the middle of the
portrait image; sampling at y=0.15 often lands on background, not the
top face.

If you're trying to verify the color of a specific sticker, first
locate the cube silhouette (using `analyze_image(...).roi` if you can,
or visual inspection of the EXIF-corrected frame), then sample within
the silhouette.

## Comparative observational claims — view both sides before claiming

Before stating "**X differs from Y in way Z**" — whether X/Y are
photos, sets, JSON outputs, code paths, recognizer states, or any
other comparable artifacts — you MUST view both X and Y in the
medium of the claim. Memory of fixtures is unreliable. Mental
models drift.

Concretely:
- Claim about photo content → view both EXIF-corrected photos
- Claim about JSON output / recognizer signals → look at both JSONs
- Claim about code differences → grep/Read both files
- Claim about overlay quality or detection counts → view both overlays
- Claim about API response shape → fetch both responses

**This caused a real failure on Set 46 (2026-05-14):** Set 46
recognition failed with all-six-face-count errors. Claude diagnosed
"Set 46 uses a different cube than the corpus — Rubik's brand with
logo center and thick black bezels distinct from the corpus cubes."
The user pointed out this was factually wrong: it's been the same
Rubik's brand cube with the same logo and the same bezels across
every image set in the corpus and hard-case manifest. Claude had
never actually looked at any corpus photo — only at JSON
diagnostics, overlays, and scores — and pattern-matched against an
imagined mental model of "stickerless training fixtures." The real
cause turned out to be the wood-grain desktop background producing
60–70% saturated pixels vs Set 15's marble café table at 13.5%,
which broke `_find_cube_roi`. A 30-second side-by-side view of
Set 15 and Set 46 photos would have caught the false premise
before any diagnosis was written.

Related prior incidents:
- PR #76 review "options_b underpopulates" claim about Sets 17/21
  (turned out to be wrong; the actual signature was face-count
  failures earlier in the pipeline, visible in
  `/tmp/pr76-set17.json` had Claude checked).
- "U-anchor white reinforcement" scope draft (assumed solved cube
  has 9 white stickers on U; Codex's empirical falsification
  produced 0 improvement).

### The trigger / mechanism

When you write "*the X uses Y, not Z*" or "*X has more/fewer N
than Y*" or "*X differs from Y because…*", ask:

> Have I actually viewed both X and Y in the medium of the claim
> recently in this session?

If the answer is "no" or "I think so, from a while ago" or "I've
seen JSON for one but not the other" — **stop and look.** The view
is cheap; the false claim is expensive.

This protocol extends the "look at the photo before claiming"
discipline from the EXIF section above to *any* comparative claim.
The EXIF rule is "view the photo before reading it"; this rule is
"view both photos before comparing them."

## Geometry Labeler conventions

The Geometry Labeler is a diagnostic annotation surface only; saved
labels do not change recognizer behavior. Labels live under
`runs/labels/` and use `coordinateSpace: browser_image_natural`,
meaning the EXIF-corrected natural image size reported by the browser,
not raw sideways iPhone storage pixels.

Face labels must use canonical cube faces, not "left side of the
photo" shorthand:

- Image A: label the visible faces as `U`, `R`, and `F`.
- Image B: label the visible faces as `D`, `L`, and `B`.
- Single-image/ad-hoc labels: use whichever canonical WCA faces are
  visible in the photo.

For face quads, click the four exterior corners in perimeter order.
For cube hulls, click only the outer cube silhouette in perimeter
order; the normal three-face isometric photos usually have a six-point
hull. Do not include the inner shared three-face corner in the cube
hull. The seven-anchor template uses that shared front corner as its
center/placement handle, then derives the six-point hull and all three
visible face quads.

When evaluating labels, prefer explicit paths so the run is
reproducible:

```bash
.venv/bin/python tools/evaluate_geometry_labels.py \
  runs/labels/<label-id>.json \
  --overlay-dir /tmp/geometry-overlays
```

If the label image cannot be resolved from manifests or the saved image
metadata, pass `--image <photo.jpg>` for a single label file. Treat
`detected` counts in the evaluator output as recognizer candidates, not
physical sticker truth; three visible faces physically imply 27
stickers.

When comparing many labelled sets before/after recognizer changes, use
the label baseline tool instead of hand-maintaining ad hoc command lists:

```bash
.venv/bin/python tools/label_geometry_baseline.py \
  --set-id 46 --set-id 47 --set-id 48 --set-id 49 \
  --json-output /tmp/label-geometry-baseline.json \
  --overlay-dir /tmp/label-geometry-baseline-overlays
```

Do not treat exact `runs/labels/...` filenames as durable
documentation unless the files have been copied into a tracked fixture
or docs path; `runs/` is gitignored and those paths are often
local-session context.

## Mutable state — re-pull before asserting (parallel-agent workflow)

When the workflow involves another agent acting in parallel —
Codex on the production recognizer, Devin on PR audits, the user
merging from the GitHub UI — **state I observed earlier in the
session is not state.** Before any sentence in user-facing
output that asserts the current condition of something an
external agent can touch, run a fresh query.

The artifacts most often mutated between my turns:

- PR state: open vs merged, `mergeStateStatus`, `mergeable`,
  head SHA
- PR labels: `needs-devin-audit`, `devin-audit-done`,
  `devin-audit-blocked` — Devin's labeler flips these
  asynchronously
- CI check rollup status
- Branch tips (Codex pushes to shared branches, force-pushes to
  their own)
- Worktree list (the user or Codex may have removed a worktree)
- File contents in any path Codex owns (per `COORDINATION.md`
  Lanes) or that's listed as **Shared**

The cheap query before each shape of claim:

| Claim shape | Cheap re-check |
|---|---|
| "PR #N is still open / in flight / pending" | `gh pr view <n> --json state,labels,mergeStateStatus` |
| "Open PRs are…" | `gh pr list --state open` (each repo) |
| "Label X is on PR #N" | `gh pr view <n> --json labels` |
| "Branch X is at SHA Y" | `git fetch && git log --oneline -1 origin/<branch>` |
| "Worktree X exists / is dangling" | `git worktree list` |
| "File X says Y" (Codex-touched or Shared paths) | re-Read before quoting |

### The trigger / mechanism

Catch yourself on words that are claims about *current* state:
**"still"**, **"currently"**, **"now"**, **"in flight"**,
**"open"**, **"pending"**, **"hasn't yet"**, **"is at"**.
These all need a current read — not memory from earlier in the
session.

Within a single turn, state I just read is fresh and trustworthy.
Across user-message boundaries, treat all external state as
stale by default: the user message itself is evidence that time
has passed and other agents may have acted in that window.

### The trade-off, made explicit

Workflow is intentionally rapid (target: seconds, not minutes).
Pulling fresh state costs ~1 second per `gh` query. The cost of a
stale-state assertion is much higher: the user has to
context-switch to verify, downstream decisions may build on the
wrong premise, and trust degrades each time it happens.

### This caused a real failure on 2026-05-18

After I merged cube-snap#136, I posted "Open PRs now (2):
ctvd#146, ctvd#147" — based on my last fresh observation of those
PRs, which was when I labeled them `needs-devin-audit` 6-7
minutes earlier. Codex had merged both in parallel (#146 at
18:43:42 UTC, #147 at 18:45:59 UTC) before I posted at ~18:50.
The user caught it: *"Codex said he merged #146 and #147. Why did
you think they're still open?"* A `gh pr list` before the
report would have taken 1 second and produced an accurate
summary.

This is the parallel-agent analogue of the comparative-claims
"view both before comparing" rule: that one is pairwise (look at
A and B); this one is temporal (re-look at X now, not as you
remember it).

## Pre-commit verification — separate commands, explicit paths

Two patterns hit me on cube-snap#114 / ctvd#94 in quick
succession. The same shape applies in this repo.

### 1. Git staging: separate commands, explicit paths

**Wrong:**

```bash
git status --short && git add -A && git diff --cached --stat \
  && git commit -F msg.txt && git push
```

`git status` and `git diff --cached --stat` are *supposed* to be
verification gates, but chained into a single `&&` sequence they
execute after staging is irreversible. The output renders but
cannot fail the commit. Output as theater.

**Right:**

```bash
git status                 # separate command, read the output
git add <explicit-files>   # name what you mean, not -A / .
git diff --cached --stat   # separate command, react if surprised
git commit -F msg.txt
git push
```

`git add -A` and `git add .` sweep untracked files. Caught
cube-snap#114 with `.claude/launch.json` (per-checkout preview
config) and `eval/tsconfig.tsbuildinfo` (TS build artifact)
slipping in. The Bash tool description warns against `-A` / `.`
explicitly — I ignored my own guidance.

Verification needs to be a *gate*: a step where future-me gets
to react. Chaining it into the action sequence eliminates the
gate.

### 2. Read project conventions before changing them

When touching:

- `.claude/settings.json` or related operating envelope rules
- pytest configuration, `requirements.txt`, `pyproject.toml`
- `.gitignore`, CI workflow files
- The `tools/probe_*.py` script structure or output schemas
- `tests/fixtures/corpus_manifest.json` and
  `tests/fixtures/hard_case_manifest.json`

Read the **existing version** AND any **sibling README** before
adding/removing rules. The most recent miss (cube-snap#114 +
ctvd#94): broadened `Bash(curl:*)` and `Bash(python3 -c:*)` in
both repos' `.claude/settings.json` without reading this repo's
`.claude/README.md`, which explicitly calls out broad
interpreters as guardrail bypasses:

> A rule `Bash(python3:*)` allows
> `python3 -c "import os; os.system('rm -rf /')"`,
> bypassing the `rm` and `sudo` deny rules.

This is the comparative-claims "view both" rule applied to
*configs*: read what's already here before changing it.

### 3. Branching from a shared checkout: audit working tree first

Before `git checkout -b <new-branch>` from a checkout that may have
uncommitted work — especially the primary checkout that other
agents and sessions share — run `git status` as a *separate*
command and audit the output. **Uncommitted modifications and
untracked files carry over to your new branch** and will end up in
your first commit if you later `git add -A` (or any other path
that sweeps them in).

**Wrong:**

```bash
git checkout main && git pull --ff-only
git checkout -b claude/my-feature
# edit + commit + push — silently includes whatever was pending
```

**Right:**

```bash
git status                 # what's pending in the working tree?
# If anything is modified or untracked, decide *before* branching:
#   - Belongs to me / this branch? → proceed, stage explicit paths
#   - Doesn't belong and is tracked-only? →
#     git stash push -m "preserving X work"
#   - Doesn't belong and includes untracked files? →
#     git stash push --include-untracked -m "preserving X work"
#     Then branch, do my work, and pop later in the right context
#   - Unclear who owns it? → ask the user
git checkout -b claude/my-feature
# stage explicit paths only — never -A / .
```

**This caused a real failure on ctvd#98 (2026-05-16):** I cut a new
branch off the primary checkout while the user had uncommitted
edits adding a "Geometry Labeler conventions" section to
`CLAUDE.md`. My intended edit was a separate "Default to acting"
section further down. When I committed, both changes ended up in
my commit. Devin's review caught the unintended Geometry Labeler
content; the fix was a follow-up commit removing those lines so
the squash-merge diff against `main` was clean.

Same shape as the `git add -A` failure mode in rule #1:
**verification must happen before the action captures state**,
not after. Rule #1 covers staging; this rule covers the analogous
step at branch-creation time. Either misstep silently includes
someone else's pending work in your commit.

## Default to acting on non-destructive next steps

The `.claude/settings.json` allow list exists so routine operations
don't need per-step confirmation. If the next step is obviously the
right move AND it's in your allow list AND it's non-destructive,
**just do it**. Don't draft "Want me to X?" / "Should I Y?" — that's
asking the user to authorize something the settings already
authorized.

### Do without asking

- Catching up on PRs / `gh pr list` / `gh pr view` / `gh api`
- Running diagnostic probes (`tools/probe_*.py`, `tools/inspect_*.py`,
  `tools/diagnose_*.py`, `tools/evaluate_*.py`)
- Running tests, builds, typechecks (`pytest`, `npm test`,
  `npm run build`, `tsc --noEmit`)
- Reading files, JSON outputs, photos (with EXIF protocol)
- Opening, creating, and editing routine development files inside the
  active repo/worktree without confirmation, including `.py`, `.md`,
  `.txt`, `.html`, `.css`, `.js`, `.ts`, `.json`, and test/fixture
  files. Use normal read tools and `apply_patch` for manual edits.
- Querying APIs (`curl http://localhost:8080/api/diag`)
- Inspecting saved-run outputs in `runs/`
- Restarting the local dev server when it's stale (per the
  "After merging a PR that affects the running UI/API" section)
- Opening a focused PR after completing scoped implementation work
- Starting a timer/check loop after opening a PR to watch for Devin
  review comments
- Addressing clear Devin review comments
- Codex only: after independently verifying a PR, merging once Devin
  says there are no blockers and normal merge checks are green, then
  fast-forwarding `main`, restarting the local server for UI/API/server
  changes, and verifying `/api/diag` reports the merged SHA on `main`
- Posting comments on existing PRs/issues via `--body-file`
- Replying to review feedback that you've already addressed

### Ask first

- Destructive ops (`rm`, force push, `reset --hard`, `branch -D` on
  shared branches, anything outside the existing deny list that
  could reasonably surprise the user)
- Opening a PR for ambiguous scope, broad refactors, or work the user
  has not already asked you to pursue
- Claude: merging any PR, unless the user explicitly delegated that
  merge in the current thread. The standing in-thread delegation is:
  **"Keep going" / "continue" / "proceed" / similar continuation
  phrases authorize merge of any PR Claude owns that carries EITHER
  `codex-audit-done` OR `devin-audit-done` AND a CLEAN merge state.
  Codex is preferred** (its findings have been higher-signal on
  this codebase per the bake-off calibration — see
  `tools/CODEX_AUDIT_PROTOCOL.md`). Greptile is informational only
  and never required for merge. Do NOT extend this to PRs owned by
  Codex-the-collaborator (different from `codex-audit-done`), to
  PRs missing both audit-done labels, or to anything that needs
  `--admin` to bypass branch protection. If the user redirects
  elsewhere ("work on X instead"), the merge auth doesn't carry
  over to that next thing.
- Merging despite unresolved or ambiguous Devin comments, failing
  checks, merge conflicts, or anything requiring `--admin`
- Sending external messages (emails, Slack DMs to non-collaborators)
- Operations touching paths outside the active repos / worktrees /
  `/tmp` / `/private/tmp` / Codex worktrees
- Anything that costs money (cloud-LLM API calls in cube-snap's
  eval pipeline, etc.)
- Anything irreversible that you can't undo with `git reset` or a
  follow-up PR

### Sandbox prompts are not content prompts

Do not ask the user for permission just because the next step opens,
creates, or edits a normal development file (`.py`, `.md`, `.txt`,
`.html`, `.json`, tests, fixtures, docs) inside the active repo.
If a command needs escalation because it writes Git metadata
(`.git/index.lock`), touches a path outside the allowed roots, uses
network access, or performs a destructive operation, treat that as a
tool/sandbox permission issue only. Prefer a non-escalated path
(`apply_patch`, focused file reads, explicit path staging) when it
fits; otherwise request the narrow escalation and continue.

### The trigger

When you find yourself drafting "Want me to X?" / "Should I Y?" /
"Let me know if you want me to Z" — and X/Y/Z is in the "do without
asking" list — **delete the question and do it instead**. Report
the result, including the data it produced. The user can interrupt
if they wanted a different next step.

This caused real friction on 2026-05-16 during the Set 46 evaluator
analysis: I knew the next step was running
`tools/evaluate_geometry_labels.py`, the tool was in my allow list,
the data answer was the whole point of the geometry-labeling
round-trip Codex had just completed — and I asked "Want me to run
it now?" instead of running it. The user (correctly) called out
that the question itself is the cost.

### Paid review budget policy

Codex and Claude are the default iterative reviewers. Greptile and
Devin are paid lanes; use them as final/confirmatory checks or
specialist help, not as automatic reviewers for every micro-commit.

Default loop:

1. Author agent implements and runs focused local validation.
2. The other coding agent reviews the stable PR (`needs-codex-audit`
   for Codex review of Claude-owned PRs, or `needs-claude-review`
   for Claude review of Codex-owned PRs).
3. Apply a paid-review label only after that loop is stable and the
   PR risk justifies the cost.

Apply `needs-devin-audit` when the desired paid help is broad
implementation review, CI/debug triage, or "please fix comments" work.
Apply `needs-greptile-audit` when the desired paid help is a final
whole-repo/code-review sanity check. Do not apply either label during
normal iteration. Tiny docs/comment-only PRs usually need no paid
review; production behavior, recognizer geometry/scoring, auth/CI
automation, cross-repo mirrored tooling, and artifact-writing tools
usually do.

Proactive merge rule: when the PR's required review lane is satisfied,
checks pass, merge state is CLEAN, and there are no unresolved blockers,
the PR owner should merge without waiting for another user prompt. For
local, iterative, diagnostic, and tiny docs/comment-only PRs, Claude or
Codex cross-review PASS is enough. For material production changes,
auth/CI automation, mirrored tooling, and other higher-risk PRs, wait
for the selected paid final-review lane when one was requested
(`devin-audit-done`, or Greptile's clean/no-actionable-blocker result)
before merging. Do not merge if the user has explicitly asked to pause,
if review comments are unresolved, if labels indicate a stale/blocking
head, or if the branch requires a mirror PR that has not been planned.

### Claude cross-review lane (no-cost iterative review)

Claude cross-review is the no-cost counterpart to the Codex audit lane
for Codex-owned PRs. It is manual and lightweight; it does not require
a bridge or labeler.

State:

- `needs-claude-review`: current PR head SHA needs Claude review.

Protocol:

1. When a Codex-owned PR is stable enough for peer review, apply
   `needs-claude-review` and leave a short PR comment naming the review
   scope and expected output (PASS or blockers). Do not apply paid
   review labels for this step.
2. If the request is made via CLI/chat instead of only GitHub, the
   sender must also acknowledge it in the sender's chat, e.g. "Asked
   Claude via CLI to review PR #N for X; expecting PASS/blockers."
   The durable PR label/comment is still required so queue state is
   visible outside chat.
3. Claude reviews the current head, posts a PR comment with
   "Claude cross-review" plus PASS or blocker findings, and removes
   `needs-claude-review` only when no blockers remain. If blockers
   remain, keep the label until a new head is ready.
4. Claude should briefly summarize received CLI review requests and
   review outcomes in Claude's chat. Codex should do the same in Codex
   chat for review requests received via Codex CLI. This keeps the
   human aware of cross-agent work that happened outside the visible
   chat thread.

Tiny docs/comment-only PRs usually need only this cross-agent review
and no paid final-review pass.

### Devin PR audit routing

Use labels as a current-head-SHA state machine in
`jeffhuber/cube-two-view-debugger` and `jeffhuber/cube-snap`:

- `needs-devin-audit`: current PR head SHA needs Devin review.
- `devin-audit-done`: Devin reviewed current PR head SHA with no
  blockers.
- `devin-audit-blocked`: Devin found blockers or could not complete
  review.

When a PR is ready for final paid Devin review, apply
`needs-devin-audit`. When adding commits after Devin has reviewed, remove stale
`devin-audit-done` / `devin-audit-blocked` and apply
`needs-devin-audit` again only if another paid review is warranted.
Do not repeatedly ping Devin; the automation dedupes by repo + PR +
head SHA, but each new reviewed head can still consume paid review time.

Keep merge authority separate from audit authority: Devin reviews
only. Codex may merge after Devin is clear and Codex independently
verifies checks/diff. Claude asks before merging unless explicitly
delegated in-thread — and "Keep going" / "continue" / "proceed" count
as that delegation for any of Claude's own PRs that are
`devin-audit-done` + CLEAN (see "Ask first" above).

The audit labeler reacts only to Devin-authored PR comments and runs
default-branch code. If GitHub's built-in workflow token returns
`Resource not accessible by integration` while applying labels, set
the repository secret `DEVIN_AUDIT_LABEL_TOKEN` to a fine-grained
token with Issues read/write access for that repository; the labeler
falls back to `github.token` when the secret is absent.

#### Audit chain implementation

- **Bridge**: `tools/devin_audit_bridge.py` builds the webhook payload
  (with `DEVIN_INSTRUCTIONS`) and posts it to Devin. Triggered by
  `.github/workflows/devin-audit-bridge.yml` on `pull_request_target`
  events (open / synchronize / reopen / labeled `needs-devin-audit`),
  on `issue_comment` containing `@devin audit` from trusted
  commenters, and on a 5-minute `schedule` watchdog (plus
  `workflow_dispatch` for manual runs). The workflow has a job-level
  prefilter so unrelated labels/comments (for example Codex or Claude
  audit labels) skip before checkout and do not create no-op Devin
  dispatch jobs. The scheduled path calls
  `scheduled_pull_requests()` to scan open PRs with the
  `needs-devin-audit` label and dispatches each through the same
  helper; `devin_already_reviewed_sha()` dedupe ensures Devin isn't
  pinged twice for the same head SHA. The watchdog catches cases
  where the event-driven dispatch was missed (manual label
  application bypassing the `labeled` event, a workflow that errored
  out, etc.).
- **Labeler**: `tools/devin_audit_labeler.py` parses Devin's audit
  comment and applies the terminal label. Triggered by
  `.github/workflows/devin-audit-labeler.yml` on `issue_comment` from
  `devin-ai-integration[bot]` only. Checks out `default_branch` (not
  the PR branch) so a malicious PR cannot modify the script that
  judges it.
- **The contract** is the HTML-comment trailer Devin appends to every
  audit comment, on its own final line:
  - `<!-- DEVIN_AUDIT_STATE: devin-audit-done -->` (pass)
  - `<!-- DEVIN_AUDIT_STATE: devin-audit-blocked -->` (blocked or
    incomplete)
  - `<!-- DEVIN_AUDIT_STATE: needs-devin-audit -->` (head changed
    during review)

  The labeler treats this trailer as authoritative. Prose phrasing
  (headers, `Label state:`, `Intended labels:`, `Expected labels:`)
  is kept only as fallback for Devin sessions that haven't adopted
  the trailer.

`tools/devin_audit_bridge.py`, `tools/devin_audit_labeler.py`, and
both `.github/workflows/devin-audit-*.yml` files MUST stay
byte-identical across `cube-two-view-debugger` and `cube-snap`.
Verify with `diff` before changing either side; PRs land in lockstep.

Tests live in `tests/test_devin_audit_bridge.py` (pytest, bridge +
labeler combined, full coverage including trailer /
Expected-labels / HEAD_CHANGED-substring-vs-structured regression
cases). Run with `python3 -m pytest tests/test_devin_audit_bridge.py`.

**Self-audit gotcha** — when modifying `tools/devin_audit_*.py`,
remember that Devin's audit comment on your PR will likely quote
identifiers from the labeler code in prose (e.g., describing the
precedence ordering as `HEAD_CHANGED_DURING_REVIEW > trailer > prose
fallbacks`). Any labeler match that uses a bare substring check on
such an identifier will false-positive on Devin's own review of the
PR. Require structured forms — `HEAD_CHANGED_DURING_REVIEW:\s*reviewed\b`
matches the protocol form Devin is instructed to emit; the bare
identifier in prose does not. Verified by the self-audit failure on
cube-snap#130 / ctvd#118 (caught during smoke verification, fixed in
the same PRs).

### Codex audit lane (merge authority — preferred)

Codex runs as a parallel reviewer alongside Devin. **Merge
authority granted, and preferred over Devin** — the standing
in-thread merge delegation accepts EITHER `codex-audit-done` OR
`devin-audit-done` + CLEAN, and when both are available Codex's
verdict is the leading signal. Calibration on PR #233 (Phase 2B
v2 audit of commit `a2ddd70`) showed Codex catching 6 real bugs
that Devin's first-pass PASS missed; on this codebase its
findings are higher-signal.

State machine (parallels Devin):

- `needs-codex-audit`: PR head SHA needs Codex review
- `codex-audit-done`: Codex reviewed current head with no
  P0/P1/P2 findings
- `codex-audit-blocked`: Codex found P0/P1/P2 blocker findings

Trigger model: **manual CLI**, not auto-fired. Run
`tools/run_codex_audit_pr.sh --repo OWNER/REPO --pr N`
to audit one PR; the wrapper uses a controlled Python interpreter
instead of ambient `python3`, and the CLI posts a comment with the
authoritative `<!-- CODEX_AUDIT_STATE: ... -->` trailer, and
`tools/codex_audit_labeler.py` (fired by
`.github/workflows/codex-audit-labeler.yml` on `issue_comment`)
flips the label.

If the built-in workflow token can't label the bot's comments,
set repo secret `CODEX_AUDIT_LABEL_TOKEN` to a fine-grained
PAT with **Issues**: read+write and **Pull requests**: read.

Mirror invariant — these files MUST stay byte-identical across
`cube-two-view-debugger` and `cube-snap`:

- `tools/codex_audit_pr.py`
- `tools/run_codex_audit_pr.sh`
- `tools/codex_audit_labeler.py`
- `tools/CODEX_AUDIT_PROTOCOL.md`
- `.github/workflows/codex-audit-labeler.yml`

Verify with `diff` before changing either side; PRs land in
lockstep, exactly as for the Devin lane.

Tests live in `tests/test_codex_audit_pr.py` and
`tests/test_codex_audit_labeler.py` — **ctvd only**, not
mirrored to cube-snap (cube-snap doesn't have its own python
venv with pytest, so running labeler tests there is awkward;
ctvd's `.venv/bin/pytest tests/test_codex_audit*` works
against ctvd's copy of the labeler, which is byte-identical
with cube-snap's).

Full protocol: `tools/CODEX_AUDIT_PROTOCOL.md`.

### Paid final-review lane (currently Greptile — informational, non-gating)

Greptile runs as a paid SaaS GitHub App. The root `greptile.json`
config restricts automatic Greptile reviews to PRs carrying
`needs-greptile-audit`; do not add that label until the PR is stable
and worth a paid final review. **Informational only — never gates merge
or approval.** Greptile's verdict is visible in its own inline
review comments on the PR; the labeler is best-effort. If
label-application fails (e.g. PAT permission gap), the labeler
logs the verdict to workflow output and exits 0 so the
workflow check stays green and the PR stays CLEAN —
explicitly designed so this lane can never block merge.

State machine (parallels Devin/Codex):

- `needs-greptile-audit`: opt-in marker — labeler ONLY acts
  on PRs carrying this (or a prior `done`/`blocked`). The root
  `greptile.json` also uses this label as the review trigger, so adding
  it can spend a paid review.
- `greptile-audit-done`: zero P0/P1 findings on current head
- `greptile-audit-blocked`: any P0/P1 finding on current head

Trigger model: **label-filtered** via Greptile config:
`greptile.json` contains `"labels": ["needs-greptile-audit"]` and
`"triggerOnUpdates": false`. `.github/workflows/greptile-audit-labeler.yml`
fires only after Greptile posts a `pull_request_review` event from
`greptile-apps[bot]` (not `issue_comment` like the others). The
labeler has four defensive gates: opt-in, stale-HEAD,
severity-parse fail-closed, verdict.

If the built-in workflow token can't label the bot's reviews,
set repo secret `GREPTILE_AUDIT_LABEL_TOKEN` to a fine-grained
PAT with **Issues**: read+write and **Pull requests**: read.

Mirror invariant — these files MUST stay byte-identical across
`cube-two-view-debugger` and `cube-snap`:

- `tools/greptile_audit_labeler.py`
- `tools/GREPTILE_AUDIT_PROTOCOL.md`
- `.github/workflows/greptile-audit-labeler.yml`

The labeler-fail-closed-on-missing-review-id path (cube-snap
#145 fix) means any malformed event or pagination cap re-applies
`needs-greptile-audit` instead of silently PASSing.

Tests live in `tests/test_greptile_audit_labeler.py` (both
repos, byte-identical). Run with `.venv/bin/pytest
tests/test_greptile_audit_labeler.py`.

Full protocol: `tools/GREPTILE_AUDIT_PROTOCOL.md`.

### Qwen lane (paused)

The Qwen lane (local LM Studio runner) was the original 2nd
audit lane during the calibration period. Currently **paused**
— its calibration showed too many false positives to justify
the LM Studio runtime cost. `tools/qwen_audit_*.py` and
`tools/QWEN_AUDIT_PROTOCOL.md` are kept on disk so the lane
is trivial to revive (`git revert` the workflow-deletion
commits + recreate the 3 `qwen-audit-*` labels). The labeler
workflow is **removed** in both repos (ctvd #237 / cube-snap
#146), so Qwen comments are inert until the lane is restored.

## Other Claude/Codex working conventions

- **GitHub markdown bodies: body-file only.** Never pass PR, issue,
  review, or comment markdown through inline shell arguments such as
  `gh pr create --body "..."` or `gh issue comment --body "..."`.
  Shell backticks inside those strings execute as command
  substitutions, which has caused repeated accidental probe/test runs
  during PR creation. Write the markdown to a temp file, then call the
  GitHub CLI with `--body-file`, placing that flag immediately after
  the subcommand so the repo permission baseline can enforce the safe
  path. For PR/issue comments, prefer the safer wrapper
  `tools/safe_gh_comment.py`, which serializes the body as JSON and
  cannot shell-interpret backticks or `$()`:

  ```bash
  gh pr create --body-file /tmp/pr-body.md --repo jeffhuber/cube-two-view-debugger ...
  .venv/bin/python tools/safe_gh_comment.py --repo jeffhuber/cube-two-view-debugger --issue 31 --body-file /tmp/comment.md
  .venv/bin/python tools/safe_gh_comment.py --repo jeffhuber/cube-two-view-debugger --edit-comment-id 123456 --body-file /tmp/comment.md
  ```

- **Corpus probe runtime fingerprint**: the manifest pins the
  ARM64/Python 3.12.13/NumPy 2.3.5/Pillow 12.2.0 environment (see
  `supportedArchitectures.primary` in `tests/fixtures/corpus_manifest.json`).
  Running the probe on a mismatched environment emits a warning, not
  an error — but recognition outcomes ARE architecture-dependent
  (Issue #25 for the full story). Always work from the matched
  ARM64 venv at `.venv/`.

- **Speed PR convention**: each Speed PR for [Issue
  #25](https://github.com/jeffhuber/cube-two-view-debugger/issues/25)
  must preserve behavior byte-for-byte. The gate is
  `tools/probe_corpus.py --fail-on-contract` on the pinned ARM64
  environment with the current 15-pair corpus. Don't merge a Speed
  PR if any row's score / category / path / candidate count differs
  from the baseline.

- **Stacked PRs and base-branch deletion**: if you base PR B on PR
  A's branch and merge PR A with `--delete-branch`, GitHub
  auto-closes PR B. Rebase B's content onto main and re-open. See
  the PR #35 → #36/#37 incident (2026-05-12) for the messy version
  of this; the recovery pattern is in
  `tools/view_photo.py`'s neighbor files in commit history.

## Cv-local server identity (which code is :8080 serving?)

Only one cv-local server can bind port 8080 at a time. When more
than one agent (Claude / Codex / the human) has a checkout of this
repo on the same host, whichever instance restarted most recently
wins port 8080 — silently. The user-facing UI looks identical
between checkouts, but the running recognizer code can be on
different commits, branches, or even different repo paths.

A real instance of this confusion happened on 2026-05-12: Codex's
WIP branch (`codex/speed-pr11-...`) had taken over port 8080 from
my `/Users/jhuber/cube-two-view-debugger/main` server, and the only
visible clue was a SHA mismatch in `/api/diag`.

### Convention — check identity before trusting recognition output

Both `/api/diag` and the startup banner now expose three identity
fields that together answer "which code is :8080 serving?":

- `git.cwd` — the repo path the server was started from
- `git.sha` — the HEAD at request time
- `git.branch` — current branch (or `null` for detached HEAD)

Before relying on the server's output during a debug session, hit:

```bash
curl -s http://localhost:8080/api/diag | python3 -c "import json,sys; \
    d=json.load(sys.stdin); print(d.get('git'))"
```

If `git.cwd` or `git.sha` doesn't match what you expect, restart
from your own repo.

### After merging a PR that affects the running UI/API

If a merged PR changes `app.py`, `static/`, server routes, API
behavior, or anything the browser-served app depends on, make the
running server match `origin/main` before giving the user a localhost
link or asking them to refresh:

1. Fast-forward the active server checkout:
   ```bash
   cd /Users/jhuber/cube-two-view-debugger
   git fetch origin
   git pull --ff-only
   ```
2. Kill the existing :8080 web UI server and restart it from that
   checkout. The user has explicitly granted permission for this
   routine server refresh.
3. Verify `/api/diag` or the startup banner reports the merged SHA
   and `main`.
4. Only then give the user the link or ask them to refresh the page.

This avoids the stale-checkout failure mode where a newly merged UI
feature is present on GitHub but the browser still opens an older local
server.

### Canonical paths per agent

| Agent | Canonical repo path |
|---|---|
| Claude | `/Users/jhuber/cube-two-view-debugger` |
| Codex  | `/Users/jhuber/Documents/Codex/.../i-want-to-create-a-rubik` |

If you're Claude or Codex and your work needs to be the active
server, restart from your own canonical path. **Redirect stderr to
a separate file**, not the canonical log:

```bash
cd <your_canonical_path>
nohup .venv/bin/python app.py > /tmp/cv-local-stderr.log 2>&1 &
```

Two-file separation (Devin / Codex review on PR #75): the boot
banner is appended to `/tmp/cv-local-server.log` by `app.py`
itself (app-owned, append, audit trail). The shell-inherited
stderr fd goes to `/tmp/cv-local-stderr.log` (request logs,
tracebacks). Reasons to keep them separate:

1. **Truncation surface.** Sharing the path via `>` would truncate
   the canonical file before Python starts, defeating the
   "accumulates an audit trail" property.
2. **Signal separation.** Same-file redirects are brittle: `>` uses
   a non-append shell fd and can overwrite app-appended boot records;
   `>>` avoids that overwrite hazard because the shell fd also has
   `O_APPEND`, but it still mixes request logs and tracebacks into
   the app-owned boot audit file. Separating the streams keeps the
   canonical file focused on boot identity records.

Use `tail -1` to get the most recent boot's identity:

```bash
grep "identity:" /tmp/cv-local-server.log | tail -1
# [rubik-app]   identity: /Users/jhuber/cube-two-view-debugger @ d594e4a (main)
```

For request logs / tracebacks (separate concern):

```bash
tail -f /tmp/cv-local-stderr.log
```

Override the canonical log path with the `CV_LOCAL_SERVER_LOG`
environment variable if you need to (test harnesses do this).
Append mode means the canonical file accumulates an audit trail
of boots rather than overwriting; if you want a fresh log,
truncate it yourself (`: > /tmp/cv-local-server.log`) before
restarting.

**Non-default ports.** Servers started on alternate ports
(`app.py --port 8085`) write their boot record to
`/tmp/cv-local-server-<port>.log` instead, so they cannot
pollute the canonical file's `tail -1` identity for :8080.
The grep convention above is :8080-specific by design — that's
the agent contention point. For other ports, query that port's
file directly.

### When you should NOT restart someone else's server

If `git.cwd` points at another agent's repo path AND the branch
isn't main, that agent is likely actively iterating. Don't kill
their session unless you've confirmed they're not around.
Easiest signal: ask the user.

## Repository

https://github.com/jeffhuber/cube-two-view-debugger
