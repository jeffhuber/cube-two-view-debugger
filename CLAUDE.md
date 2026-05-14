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

## Other Claude/Codex working conventions

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
