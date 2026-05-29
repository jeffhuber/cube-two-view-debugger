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

## Recognizer architecture: deployed constrained inference (2026-05-29)

The recognizer architecture has shifted significantly. **The product
path is now constrained cube-state inference, not LLM-as-oracle.**
Future contributors should orient around this framing rather than the
older "LLM reads colors, deterministic repair cleans up" mental model.

### The shift, in one paragraph

The old framing treated recognition as "extract 54 colored stickers
from two photos, run a solver." The LLM (claude-sonnet-rectified)
was the color extractor; deterministic post-processing was an
afterthought. That framing implicitly assumed per-sticker
independence: P(98% cube exact) requires P(99.96% sticker accuracy).
At the empirical ~96.4% per-sticker we were stuck — too few exact
cubes to ship as a no-touch recognizer.

The reframe (credit to Codex's 2026-05-26 strategic exchange with
Claude) is: treat recognition as a **constrained cube-state
inference problem**. Each sticker has top-N color candidates with
confidence — from Lab distance to known WCA centers AND from the
LLM read — two independent evidence sources. Hard constraints
(9-per-color, fixed centers, cubie legality from valid corner
orientations + edge permutation parity, two-view consistency on
shared cubies) slash the solution space. Find the most-likely valid
cube state given the noisy evidence. **LLMs become one input to a
constrained solver, not the source of truth.**

### Current deployed score

Fresh production-path score, generated 2026-05-29 against
`https://api.cubesnap.app/api/recognize?slim=1&hullLabelTier1=constrained`:

| Corpus | Exact | Within 3 stickers | Rejected |
|---|---:|---:|---:|
| 71-row local manifest | 71/71 | 71/71 | 0 |

Timing from the same deployed run, concurrency 1:

| Metric | P50 | P90 | Max |
|---|---:|---:|---:|
| End-to-end latency | 3158 ms | 3444 ms | 8217 ms |
| Server `recognizeTotal` | 2678 ms | 2882 ms | 7874 ms |

Recommended methods in that run: `canonical_count_repaired` 67,
`conservative_legal_repaired` 2, `guarded_broad_legal_repaired` 1,
`two_view_consistency_repaired` 1. Contact sheets were omitted in all
71 production responses.

The old 46-pair shadow-corpus table was a useful development snapshot,
but it is now stale and should not be used as a launch-quality number.
If you need a current accuracy claim, rerun
`tools/score_deployed_recognizer.py` against the deployed endpoint and
cite that artifact.

### What's currently wired

- **`/api/recognize?hullLabelTier1=constrained`**: the production path
  consumed by cube-snap via `api.cubesnap.app`. It runs constrained
  inference first, returns the constrained candidate when the runtime
  gate accepts it, and fast-rejects obvious non-cube/bad-orientation
  inputs without legacy fallback.
- **`/api/recognize` without the constrained query**: retained for
  compatibility/debugging. Do not infer cube-snap production behavior
  from this raw default; the web app supplies the constrained query.
- **`/api/llm-rectified-input`** (Fixer-side endpoint): serves
  hull-label-rectified panels + `deterministicColorRepair` per pair.
  Consumed by cube-snap's "claude-sonnet-rectified" Fixer option
  (cube-snap#187).
- **`tools/hull_label_color_repair.py`**: the deterministic repair
  helper. Extracted from the diagnostic into a reusable module by
  ctvd#324. Used by both the API endpoint and the 69-73 diagnostic
  so they stay aligned.
- **Diagnostic suite**: `diagnose_hull_label_color_repair.py`,
  `diagnose_hull_label_legal_repair.py`, `diagnose_hull_label_mask_thresholds.py`,
  + their committed JSON/markdown fixtures. Read these for current
  per-set behavior before opening a recognizer PR.
- **Durable metadata logging**: Railway production writes metadata-only
  recognition events to `CUBE_RECOGNITION_EVENT_DB_PATH`, currently
  `/data/recognition_events.sqlite3`, and `/api/diag` exposes aggregate
  counters. No image bytes or 54-char states are persisted there.

### Open levers (approximate descending leverage)

1. **Real-traffic telemetry review.** The durable event table now exists;
   the next leverage is watching real uploaded photos, reject reasons,
   latency tails, and per-category drift instead of relying only on the
   local manifest.
2. **Lab + LLM evidence ensemble per sticker.** Currently LLM-only;
   adding a Lab-distance read as a second independent evidence
   source would tighten the per-sticker posterior before any
   constraint solving. Cheap to add (Lab classification is already
   used in the diagnostic pipeline) and probably the next-largest
   per-sticker accuracy lift.
3. **Gloss/glint and hard-background robustness.** The local corpus is
   clean after constrained inference, so remaining quality work should
   be driven by real-photo failures and targeted hard cases.
4. **Confidence-aware user experience.** Clean results should solve
   directly; medium/low confidence should route to the Fixer/correction
   UI with the most useful uncertainty surfaced.

### What this section is NOT for

- Replacing recognizer source-code documentation. Pipeline-stage
  explanations live inline in `rubik_recognizer/`.
- Freezing the architecture. Constrained inference is the current
  best framing; if real-traffic data exposes a new failure mode, or a
  new color-evidence source materializes, the framing should evolve.
  Treat this as the 2026-05-29 snapshot.

Read this if you're about to: open a recognizer-architecture PR,
scope new diagnostic work, claim an accuracy number in external
documents, or pick the next strategic lever. Otherwise the procedural
sections below are likely more relevant to your task.

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

### Queue state is mutable too — sweep both sources

The "re-pull before asserting" rule applies to QUEUE state ("what
work exists right now"), not just per-PR state ("what's the head
SHA on PR #N"). Treating your own conversation context as
authoritative on "what's in my queue" is the same failure mode as
treating memory as authoritative on a single PR's labels — neither
survives parallel agent activity. The other agent (Codex or Claude,
depending which side you're on) may have opened, labeled, audited,
or merged work since you last looked.

Two authoritative sources to sweep, both cheap:

- **GitHub** answers "what PRs exist, what labels they carry, are
  they mergeable":
  ```
  gh pr list --repo OWNER/REPO --state open \
    --json number,title,labels,headRefOid,mergeStateStatus
  ```
  Run for both cube-snap and cube-two-view-debugger.

- **Shared local audit log** (`~/.cache/cube-agent-audits/`)
  answers "which audits are mid-flight on either agent's machine
  right now":
  ```
  tools/audit_handoff_log.py status
  ```
  This was added by ctvd#307 / cube-snap#169 and only works as a
  coordination layer if BOTH agents (a) read it before firing
  audits, and (b) write to it via `tools/run_codex_audit_pr.sh`
  (which creates locks automatically), not the raw
  `codex_audit_pr.py` script.

Sweep both at every natural transition point: after completing a
task, when a background notification arrives, before declaring
"next steps" or "holding for direction," before firing any audit.
This is what makes the standing instructions actually work:

- *"act on `needs-claude-review` / `needs-codex-audit` ASAP"* only
  fires when you observe the label; observation requires polling.
- *"proactive merge rule"* (merge own PRs that hit
  `codex-audit-done`/`devin-audit-done` + CLEAN + no explicit
  hold) only fires when you observe the audit label flipped;
  same polling requirement.

Concrete failure modes this rule prevents:

1. **Review PR lapses for hours** because the other agent opened
   it and you didn't notice. Standing instruction only fires on
   observation; observation requires polling.
2. **Mergeable PR sits open** because the audit label flipped to
   `*-audit-done` while you were doing other work and you never
   re-checked.
3. **Duplicate audit fires** on a SHA the other agent is already
   reviewing locally. The wrapper enforces this with exit code 20,
   but reading the shared log proactively keeps you from even
   starting the invocation (wastes Codex CPU / cache cycles).

Symmetric rule. The shared audit log only works as coordination
if both agents read AND write it; the queue sweep only works as a
queue if both agents do it.

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

Before any code or documentation edit, identify both the current
branch and the worktree state:

```bash
git status -sb
git branch --show-current
```

If the branch belongs to the other agent (for example Codex sees
`claude/...`, or Claude sees `codex/...`), **do not edit in that
checkout**. Create/switch to your own branch first. If the primary
checkout is on another agent's branch or has unrelated modifications,
use a separate worktree rooted at current `origin/main`:

```bash
git fetch origin main
git worktree add -b codex/<topic> /tmp/ctvd-codex-<topic> origin/main
```

Use your own prefix (`codex/...` for Codex, `claude/...` for Claude)
unless the user explicitly requests otherwise. Before committing or
opening a PR, verify the branch and the intended diff one more time:

```bash
git fetch origin main
git branch --show-current
git diff --stat origin/main...HEAD
```

If the branch prefix, owner, or diff scope is surprising, stop and fix
that before commit/PR. This is a hard gate, not a status flourish.

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

**This caused a real branch-contamination failure on 2026-05-27:**
Codex edited while the primary checkout was on Claude's
`claude/sweep-hook-audit-log-crosscheck` branch, putting the first
`tools/request_review.py` helper commit onto Claude's PR #366. The
correct fix was a separate Codex-owned PR (#367) plus a note asking
Claude to drop/rebase the accidental commit. The prevention is the
branch-owner preflight above: check `git status -sb` and
`git branch --show-current` before edits, and never do Codex work on a
`claude/...` branch.

## Worktrees — pre-flight before any Edit/Write/Bash (read once per session)

This repo uses git worktrees under `.claude/worktrees/<name>/`.
The primary checkout (`/Users/jhuber/cube-two-view-debugger`) and
any worktree are **separate working trees**: same `.git/` but
**divergent files on disk**. The primary is typically on Codex's
WIP branch (e.g. `codex/hull-label-post305-validation`,
`codex/hull-label-slot-color-diagnostic`) while a worktree is on
the branch you're actually building. Editing the wrong copy
silently splits your work AND stomps on Codex's checkout.

### Protocol — every session, before the first Edit/Write:

1. **Detect**. Run `pwd`. If the path contains `.claude/worktrees/`,
   you're in a worktree. The primary is at
   `/Users/jhuber/cube-two-view-debugger`.
2. **Root every path generator at `pwd` or a repo-relative path,
   never at the primary absolute path.** Concretely:
   - `grep -rn "..." tools/ tests/` ✅ (relative — auto-rooted at pwd)
   - `find . -name "*.py"` ✅
   - `grep -rn "..." /Users/jhuber/cube-two-view-debugger/tools` ❌
   - `find /Users/jhuber/cube-two-view-debugger -name "*.py"` ❌
3. **Before every Edit/Write, the `file_path` argument must start
   with the worktree root** (`$(pwd)` or the explicit
   `/Users/jhuber/cube-two-view-debugger/.claude/worktrees/<name>/`
   prefix). If it starts with
   `/Users/jhuber/cube-two-view-debugger/<other>` (no worktrees
   component), you're about to edit Codex's primary working tree
   — typically on Codex's WIP branch. Stop, rewrite the path,
   retry.
4. **If you must touch the primary** (rare — usually for cleaning
   up an accidental write):
   - `cd /Users/jhuber/cube-two-view-debugger` in a single Bash
     call.
   - Run `git diff -- <paths>` FIRST. Codex may have uncommitted
     work at the same paths. If the diff shows hunks that are NOT
     your accidental edit, do NOT run `git restore` — it would
     silently discard Codex's work.
   - If the diff is *only* your accidental edit, `git restore
     <paths>` is safe (or `git rm -f` for files added via
     `git checkout origin/codex/... -- <path>`).
   - Otherwise: revert only your hunk manually
     (`git checkout -p`, or `git apply -R` of a hand-extracted
     patch), or ask the user before touching anything.
   The rule of thumb: `git restore` in a primary you don't own is
   destructive by default; treat it like `rm -rf` on a directory
   whose contents you haven't audited.

### This caused real bugs TWICE on 2026-05-25

In the same session, while landing protocol mirror PRs:

- **cube-snap#169 / ctvd#307** (audit-handoff-log): ran
  `git checkout origin/codex/audit-handoff-log -- tests/...
  tools/...` into the ctvd PRIMARY (which was on
  `codex/hull-label-post305-validation`) instead of the worktree.
  Remediated via `git rm -f tests/test_audit_handoff_log.py
  tools/audit_handoff_log.py` once the user explicitly authorized.
- **cube-snap#173 / ctvd#311** (review-log-events): `Edit` call
  with `file_path=/Users/jhuber/cube-two-view-debugger/tests/
  test_audit_handoff_log.py` when the worktree path was
  `/Users/jhuber/cube-two-view-debugger/.claude/worktrees/
  review-log-events/tests/test_audit_handoff_log.py`. Same root
  cause: default-completed to the primary path. Remediated by
  copying the edits to the worktree and `git restore` in the
  primary.

Pattern: when editing files in ctvd from a session whose primary
mental model is cube-snap, hands default to the ctvd primary
path. The fix is rule 3 above — verify the prefix includes the
`.claude/worktrees/` component before passing the file_path to
Edit/Write.

### Why the existing `pwd && git branch --show-current` pre-flight
### wasn't enough

That check answers *"where am I?"* — it does not answer *"do the
file paths I'm about to pass to Edit live under here?"* Those are
different questions. The path-origin check (rule 3) is the one
that actually catches this failure mode.

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
- **Editing `.claude/settings.json` or other agent-config files** —
  the auto-mode classifier flags these as "self-modification" and
  blocks the edit without explicit user authorization. This catches
  the case where Claude would update its own allow/deny rules,
  permissions, or hook-script hashes silently. The expected
  interaction pattern: ask the user *once per distinct hash flip*
  (so you don't ask twice in a row for the same change), explain
  what's changing and why, then proceed. Examples in this codebase:
  the SessionStart-hook script hash pin in `.claude/settings.json`
  is updated whenever `tools/claude_session_start_sweep.sh` changes;
  each new hash needs fresh user OK because the classifier won't
  carry authorization across distinct hashes. See cube-snap#201 /
  cube-snap#204 for the round-by-round examples — the user
  authorized each hash bump individually, and the classifier
  blocked an unauthorized one (a fabricated `<!-- CODEX_AUDIT_STATE`
  trailer in a Claude-authored transparency comment that would
  have impersonated Codex's verdict).
- Claude: merging any PR, unless the user explicitly delegated that
  merge in the current thread. The standing in-thread delegation is:
  **"Keep going" / "continue" / "proceed" / similar continuation
  phrases authorize merge of any PR Claude owns that carries EITHER
  `codex-audit-done` OR `devin-audit-done` AND a CLEAN merge state.
  Codex is preferred** (its findings have been higher-signal on
  this codebase per the bake-off calibration — see
  `tools/CODEX_AUDIT_PROTOCOL.md`).

  **Captured-PASS-via-dump counts as `codex-audit-done`.** The
  current Codex CLI release (v0.133.0-alpha.1, model=gpt-5.5,
  reasoning effort=xhigh) deterministically routes the final
  review prose to stdout WITHOUT the column-0 `codex` marker the
  wrapper parses for. The marker lands in stderr instead. **When
  the captured stdout is a clean PASS** (substantive Codex
  summary, zero `[P0]`/`[P1]`/`[P2]` finding bullets), the
  wrapper falls to UNKNOWN and the comment carries the
  `needs-codex-audit` trailer despite the review being clean.
  Empirically observed on 87 of 87 dumps captured during the
  cube-snap session ending 2026-05-27 — see
  `tools/CODEX_AUDIT_PROTOCOL.md` for the investigation. The
  `dump_cli_failure()` instrumentation (cube-snap#202 / ctvd#368)
  captures Codex's actual stdout prose to
  `~/.cache/cube-agent-audits/cli-failures/` on every such
  occurrence. When that captured stdout (a) contains a substantive
  Codex summary line, (b) shows zero `[P0]`/`[P1]`/`[P2]` finding
  bullets, and (c) reads as a PASS verdict (e.g. "did not find any
  actionable regressions", "no introduced correctness issues", "no
  regressions introduced by this patch"), that captured prose IS
  the Codex verdict. The mechanical UNKNOWN label is the CLI
  behavior, not a missing review.

  **This path is UNKNOWN-only — it does NOT apply to
  stderr-fallback BLOCKED.** When the wrapper's stderr-fallback
  path accepts a blocker-shaped block from stderr, it posts a
  BLOCKED verdict with the `codex-audit-blocked` trailer. That
  outcome is a real BLOCKED audit and stays blocked. The
  captured-PASS escape hatch below is strictly for the
  UNKNOWN-classified case where the captured stdout has zero
  blocker findings.

  **Verify the dump matches the PR's current head before merging.**
  Both the formal `codex-audit-done` label and the captured-PASS
  dump are signals tied to a specific head SHA that can go stale
  on later pushes. The labeler workflow (cube-snap#205 / ctvd#371)
  clears `codex-audit-done` / `codex-audit-blocked` on every
  `pull_request.synchronize` event, so the formal label path is
  safely head-bound by the workflow itself. The captured-PASS dump
  lives in the operator's local `~/.cache/` and persists across
  pushes — so the operator MUST manually verify head-match before
  using it:

  ```bash
  CURRENT=$(gh pr view <N> --repo <owner>/<repo> \
    --json headRefOid --jq .headRefOid)
  ls ~/.cache/cube-agent-audits/cli-failures/<owner>_<name>_pr<N>_${CURRENT:0:12}_*.log
  grep "^# head_sha: $CURRENT$" <that-file>
  ```

  Both checks must pass. Mismatch = captured-PASS does NOT apply;
  either re-run the audit or wait for the formal label.

  **Transparency comment requirements.** Before merging on the
  captured-PASS path, post a comment that names the dump file and
  quotes Codex's actual verdict prose. Two phrasing constraints
  from the labeler-prose-fallback shape:
  - Do NOT use the exact phrase `Codex Audit: PASS` (or any
    `Codex Audit [—–:-] PASS` variant) — the labeler's prose
    fallback matches that pattern and would silently apply
    `codex-audit-done` if your login ever ended up in
    `CODEX_BOT_AUTHORS`. Use prose-only wording like "Captured
    PASS verdict on `<sha>`" instead.
  - Do NOT fabricate a `<!-- CODEX_AUDIT_STATE: codex-audit-done -->`
    trailer — that would impersonate Codex.

  Empirical basis: snap#202/#368/#201/#366/#204/#370 (three mirror
  pairs across one session, all UNKNOWN with captured PASS, all
  merged under user authorization on this exact pattern). Codex
  caught both the head-match gap (P1) and the prose-fallback gap
  (P2) on the first two rounds of cube-snap#205 / ctvd#371 itself;
  the labeler-side `synchronize` handler in the same PR pair
  closes the head-stale-label class entirely.

  Greptile is informational only and never required for merge. Do
  NOT extend this to PRs owned by Codex-the-collaborator
  (different from `codex-audit-done`), to PRs missing both
  audit-done labels (and also missing a captured-PASS dump per the
  variant above), or to anything that needs `--admin` to bypass
  branch protection. If the user redirects
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

Review handoff visibility: whenever you ask another reviewer to look at
a PR, report the handoff in your active chat with Pacific wall-clock
time, reviewer, PR number, head SHA, trigger path, and expected result.
This applies to Claude/Codex CLI handoffs, label-only handoffs
(`needs-claude-review`, `needs-codex-audit`, `needs-greptile-audit`,
`needs-devin-audit`), direct PR comments, and manual workflow dispatches.
When the reviewer returns, report the return time, elapsed time,
verdict, label state, and next action. Use a compact format such as:
`2026-05-25 09:03 PT - requested Claude review on #301 @ b29d00c via
label+CLI; expecting PASS/blockers.` and
`2026-05-25 09:08 PT - Claude PASS on #301 @ b29d00c, label removed;
elapsed 5m; merging.` If a paid lane is triggered, explicitly say that
the action may spend a review. If a labeler fails to flip labels, report
the manual correction and whether a follow-up bug is needed.

Explicit authorization comment for paid labels: adding
`needs-devin-audit` or `needs-greptile-audit` MUST be paired with an
explicit authorization PR comment posted in the same turn. The
label-only path is the silent-spend failure mode; the comment makes
the spend decision auditable in the PR record and forces a "why now"
reason at the moment of the click. Required shape (one or the other):

> Greptile paid final review requested for current head `<sha>`.
>
> Reason: `<short reason this PR is material enough to spend a paid review>`.

> Devin paid final review requested for current head `<sha>`.
>
> Reason: `<short reason>`.

If the PR head changes after a paid review lands and another paid
pass is justified, post a NEW comment naming the new head SHA +
reason BEFORE re-applying / re-triggering the paid label. Don't
reuse the prior comment — each spend event needs its own
authorization. Cross-review labels (`needs-codex-audit`,
`needs-claude-review`) imply no paid spend and do NOT require this
comment. Auto-messages from paid providers like "Required label not
found" or "PR not labeled — skipping" are NOT spend events; they
just acknowledge the opt-in gate worked. The spend event is the
label + authorization-comment pair we apply. Drafts, docs-only
micro PRs, protocol mirrors, and trivial diagnostic/report edits
should NOT carry these labels unless the user explicitly requests
the paid review. Helper:
Use `tools/request_review.py` for review requests. It generates the
PR comment inside Python, posts through `tools/safe_gh_comment.py`
(JSON-backed GitHub API, no shell interpretation of backticks or
`$()`), applies the routing label, and writes the shared
`review_requested` event in one workflow. Do not hand-roll review
request comments with shell heredocs.

### Claude cross-review lane (no-cost iterative review)

Claude cross-review is the no-cost counterpart to the Codex audit lane
for Codex-owned PRs. It is manual and lightweight; it does not require
a bridge or labeler.

State:

- `needs-claude-review`: current PR head SHA needs Claude review.

Protocol:

1. When a Codex-owned PR is stable enough for peer review, apply
   `needs-claude-review` and leave a short PR comment naming the review
   scope and expected output (PASS or blockers). Use
   `tools/request_review.py` for this step; do not use hand-written
   shell heredocs or inline `gh ... --body` comments. Do not apply
   paid review labels for this step.
2. If the request is made via CLI/chat instead of only GitHub, the
   sender must also acknowledge it in the sender's chat, e.g. "Asked
   Claude via CLI to review PR #N for X; expecting PASS/blockers."
   The durable PR label/comment is still required so queue state is
   visible outside chat.
3. Claude reviews the current head and finalizes the review via
   `tools/post_review.sh`, which bundles three required steps in
   the right order:
     (a) post the PR comment with "Claude cross-review" + PASS or
         blocker findings via `tools/safe_gh_comment.py` (no shell
         interpretation of Markdown);
     (b) remove `needs-claude-review` only when no blockers remain
         (if blockers remain, keep the label until a new head is
         ready, and skip this step);
     (c) append a `finished` event to the shared local audit log
         (`~/.cache/cube-agent-audits/events.jsonl`) so the other
         agent's Monitor on that log catches the verdict in real
         time.
   Step (c) is what makes review-side pickup symmetric with
   audit-side pickup — without it, audits flow through the shared
   log but reviews don't, leaving the other agent's Monitor blind to
   reviews. The same helper and discipline apply symmetrically to
   Codex when manually cross-reviewing a Claude-owned PR (use
   `--lane codex-review`). Example invocation:

   ```bash
   .venv/bin/python tools/request_review.py \
     --lane claude-review \
     --repo jeffhuber/cube-two-view-debugger \
     --pr 365 \
     --head "$HEAD_SHA" \
     --label needs-claude-review \
     --reviewer Claude \
     --actor codex \
     --scope "Review the current head for PASS/blockers." \
     --validation ".venv/bin/python -m pytest tests/test_request_review.py"
   ```

   Review finalization examples:

   ```bash
   # PASS — comment posted, label removed, finished event logged:
   tools/post_review.sh \
     --lane claude-review \
     --repo jeffhuber/cube-snap --pr 172 \
     --head 6abe1c7 \
     --verdict pass \
     --label needs-claude-review \
     --body-file /tmp/claude-review-snap172.md

   # BLOCKED — comment posted, label KEPT, finished event logged.
   # The label stays so the queue sweep / standing instructions
   # still see "this PR needs follow-up." Helper detects verdict
   # != "pass" and skips the label removal step.
   tools/post_review.sh \
     --lane claude-review \
     --repo jeffhuber/cube-snap --pr 172 \
     --head 6abe1c7 \
     --verdict blocked \
     --label needs-claude-review \
     --body-file /tmp/claude-review-snap172.md
   ```
   The three calls are independent (no atomic guarantee). If the
   first succeeds but later steps fail, the comment is on the PR
   but the label/log is inconsistent — re-run the remaining steps
   manually.
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
instead of ambient `python3`, creates a local audit handoff lock under
`~/.cache/cube-agent-audits` to avoid duplicate Claude/Codex audits on
the same repo/PR/head, and posts a comment with the
authoritative `<!-- CODEX_AUDIT_STATE: ... -->` trailer, and
`tools/codex_audit_labeler.py` (fired by
`.github/workflows/codex-audit-labeler.yml` on `issue_comment`)
flips the label.

Before starting a manual audit outside the wrapper, check
`tools/audit_handoff_log.py status --repo OWNER/REPO --pr N`; if a
matching active audit exists, do not start another one. Treat it as
"audit already in progress" and wait for the existing result.

If the built-in workflow token can't label the bot's comments,
set repo secret `CODEX_AUDIT_LABEL_TOKEN` to a fine-grained
PAT with **Issues**: read+write and **Pull requests**: read.

Mirror invariant — these files MUST stay byte-identical across
`cube-two-view-debugger` and `cube-snap`:

- `tools/codex_audit_pr.py`
- `tools/audit_handoff_log.py`
- `tools/run_codex_audit_pr.sh`
- `tools/codex_audit_labeler.py`
- `tools/request_review.py`
- `tools/post_review.sh`
- `tools/safe_gh_comment.py`
- `tools/claude_session_start_sweep.sh`
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
  during PR creation. Review requests must use
  `tools/request_review.py`, not a handwritten heredoc. For other
  GitHub markdown writes, write the markdown to a temp file using a
  quoted heredoc (`<<'EOF'`) or another non-interpreting writer, then
  call the GitHub CLI with `--body-file`, placing that flag immediately
  after the subcommand so the repo permission baseline can enforce the
  safe path. For PR/issue comments, prefer the safer wrapper
  `tools/safe_gh_comment.py`, which serializes the body as JSON and
  cannot shell-interpret backticks or `$()`:

  ```bash
  .venv/bin/python tools/request_review.py --repo jeffhuber/cube-two-view-debugger --pr 31 ...
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
