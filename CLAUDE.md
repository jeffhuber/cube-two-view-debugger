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
  environment with the current 14-pair corpus. Don't merge a Speed
  PR if any row's score / category / path / candidate count differs
  from the baseline.

- **Stacked PRs and base-branch deletion**: if you base PR B on PR
  A's branch and merge PR A with `--delete-branch`, GitHub
  auto-closes PR B. Rebase B's content onto main and re-open. See
  the PR #35 → #36/#37 incident (2026-05-12) for the messy version
  of this; the recovery pattern is in
  `tools/view_photo.py`'s neighbor files in commit history.

## Repository

https://github.com/jeffhuber/cube-two-view-debugger
