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
  merge in the current thread
- Merging despite unresolved or ambiguous Devin comments, failing
  checks, merge conflicts, or anything requiring `--admin`
- Sending external messages (emails, Slack DMs to non-collaborators)
- Operations touching paths outside the active repos / worktrees /
  `/tmp` / `/private/tmp` / Codex worktrees
- Anything that costs money (cloud-LLM API calls in cube-snap's
  eval pipeline, etc.)
- Anything irreversible that you can't undo with `git reset` or a
  follow-up PR

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

### Devin PR audit routing

Use labels as a current-head-SHA state machine in
`jeffhuber/cube-two-view-debugger` and `jeffhuber/cube-snap`:

- `needs-devin-audit`: current PR head SHA needs Devin review.
- `devin-audit-done`: Devin reviewed current PR head SHA with no
  blockers.
- `devin-audit-blocked`: Devin found blockers or could not complete
  review.

When a PR is ready for Devin review, apply `needs-devin-audit`. When
adding commits after Devin has reviewed, remove stale
`devin-audit-done` / `devin-audit-blocked` and apply
`needs-devin-audit` again. Do not repeatedly ping Devin; the
automation dedupes by repo + PR + head SHA.

Keep merge authority separate from audit authority: Devin reviews
only. Codex may merge after Devin is clear and Codex independently
verifies checks/diff. Claude asks before merging unless explicitly
delegated in-thread.

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
  `workflow_dispatch` for manual runs). The scheduled path calls
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

## Other Claude/Codex working conventions

- **GitHub markdown bodies: body-file only.** Never pass PR, issue,
  review, or comment markdown through inline shell arguments such as
  `gh pr create --body "..."` or `gh issue comment --body "..."`.
  Shell backticks inside those strings execute as command
  substitutions, which has caused repeated accidental probe/test runs
  during PR creation. Write the markdown to a temp file, then call the
  GitHub CLI with `--body-file`, placing that flag immediately after
  the subcommand so the repo permission baseline can enforce the safe
  path:

  ```bash
  gh pr create --body-file /tmp/pr-body.md --repo jeffhuber/cube-two-view-debugger ...
  gh issue comment --body-file /tmp/comment.md 31 --repo jeffhuber/cube-two-view-debugger
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
