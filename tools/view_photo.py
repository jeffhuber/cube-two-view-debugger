#!/usr/bin/env python
"""
EXIF-correct an iPhone JPEG and write to /tmp so it can be Read correctly.

Usage:
    .venv/bin/python tools/view_photo.py "/Users/jhuber/Downloads/Set 39 - B - white up IMG_7158.JPG"

    # prints:
    # /tmp/Set 39 - B - white up IMG_7158-corrected.jpg

Then Read the printed path. Do NOT Read the original /Users/jhuber/Downloads
file directly when debugging — the Read tool does not apply EXIF
orientation, and iPhone JPEGs are stored as landscape pixels with
`Orientation=6` requiring a 90° CW rotation to display correctly.

This caused a real bug on Set 39 (2026-05-12) where Claude misread
"yellow on top" as "white on top" because the displayed pixels were
rotated 90° CCW from what the user saw in Preview. See CLAUDE.md for
the full incident notes.

Output is a JPEG at quality 85 (small enough to keep /tmp tidy but
high enough to preserve sticker colors for downstream visual checks).
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(
            f"usage: {Path(sys.argv[0]).name} <photo_path>",
            file=sys.stderr,
        )
        return 2

    src = Path(sys.argv[1]).expanduser()
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        return 1
    if not src.is_file():
        print(f"not a file: {src}", file=sys.stderr)
        return 1

    # Import Pillow lazily so `--help`-style early errors don't pay
    # the import cost.
    from PIL import Image, ImageOps  # type: ignore

    try:
        with Image.open(src) as image:
            corrected = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:  # noqa: BLE001 — print error and exit non-zero
        print(f"failed to read {src}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    out = Path("/tmp") / f"{src.stem}-corrected.jpg"
    corrected.save(out, quality=85)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
