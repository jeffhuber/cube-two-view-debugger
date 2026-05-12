#!/usr/bin/env python
"""
Recognize a single (imageA, imageB) pair from the command line, saving
a run under runs/pairs/<run-id>/ exactly the way POST /api/recognize
does. No HTTP server required.

Why this exists
---------------
The web UI is the operator's surface. But Claude Code / Codex sessions
debugging a recognition failure don't always want to (or can't) start
the HTTP server first. Running the same pipeline at the CLI keeps the
two paths in lock-step: the run lands in the same runs/pairs/ tree,
the saved files match, the recognizer code path is identical, and the
output is JSON-printable for piping into other tools.

Usage
-----
    .venv/bin/python tools/recognize_pair.py \\
        "/Users/jhuber/Downloads/Set 21 - A - white up.JPG" \\
        "/Users/jhuber/Downloads/Set 21 - B - white up.JPG"

    # With explicit set id (used in the run id; otherwise derived):
    .venv/bin/python tools/recognize_pair.py imageA.jpg imageB.jpg --set-id "Set 21"

    # Persist the full JSON payload to a file (mirrors --json-output
    # on probe_corpus.py for consistency):
    .venv/bin/python tools/recognize_pair.py imageA.jpg imageB.jpg \\
        --json-output /tmp/recog.json --quiet

The non-quiet output is a one-line summary suitable for live runs:

    set=Set 21 run=20260512-181522-... status=rejected reason=...
    score=0/54 confidence=0.0 candidates=9472 path=/runs/pairs/...

For the full structured response use --json-output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Importing from app.py reuses the canonical recognize-and-persist
# code path. This is intentional: we want the CLI to behave EXACTLY
# the same way HTTP POST /api/recognize behaves, just without the
# multipart wrapping. Any divergence between the two would be a bug.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import recognize_and_persist  # noqa: E402
from rubik_recognizer.dataset import (  # noqa: E402
    ImagePair,
    ImageUpload,
    normalize_set_id,
)
from rubik_recognizer.recognizer import WhiteUpRecognizer  # noqa: E402


def _summary_line(payload: dict) -> str:
    status = payload.get("status", "?")
    reason = payload.get("reason") or ""
    confidence = payload.get("confidence")
    candidates = payload.get("candidates")
    evaluation = payload.get("evaluation") or {}
    score_part = ""
    if evaluation.get("available"):
        matched = evaluation.get("matched", "?")
        score_part = f" score={matched}/54"
    confidence_part = f" confidence={confidence}" if confidence is not None else ""
    candidates_part = f" candidates={candidates}" if candidates is not None else ""
    return (
        f"set={payload.get('setId', '?')} "
        f"run={payload.get('runId', '?')} "
        f"status={status}"
        f"{score_part}"
        f"{confidence_part}"
        f"{candidates_part}"
        f" reason={reason!r}"
    )


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recognize an (imageA, imageB) Rubik's pair from disk; saves a run under runs/pairs/.",
    )
    parser.add_argument("image_a", type=Path, help="Path to image A (white face on top).")
    parser.add_argument("image_b", type=Path, help="Path to image B (yellow on top after the flip).")
    parser.add_argument(
        "--set-id",
        default=None,
        help="Explicit set id for the run (e.g. 'Set 21'). If omitted, derived from filenames.",
    )
    parser.add_argument(
        "--expected-state",
        default=None,
        help="Optional 54-char ground-truth state for evaluation against the recognized output.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write the full JSON payload to (same shape as POST /api/recognize).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable summary line on stdout. JSON output is unaffected.",
    )
    args = parser.parse_args(argv)

    for label, path in (("imageA", args.image_a), ("imageB", args.image_b)):
        if not path.exists():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 1
        if not path.is_file():
            print(f"{label} not a file: {path}", file=sys.stderr)
            return 1

    image_a_bytes = args.image_a.read_bytes()
    image_b_bytes = args.image_b.read_bytes()

    set_id = args.set_id or normalize_set_id(args.image_a.name + " " + args.image_b.name)
    pair = ImagePair(
        set_id=set_id,
        image_a=ImageUpload(args.image_a.name, image_a_bytes),
        image_b=ImageUpload(args.image_b.name, image_b_bytes),
    )

    recognizer = WhiteUpRecognizer()
    payload = recognize_and_persist(recognizer, pair, expected_state=args.expected_state)
    # Stamp setId for the summary line and JSON output. recognize_and_persist
    # plumbs set_id into the run id but doesn't re-emit it as a top-level
    # JSON field; we add it for CLI consumers.
    payload.setdefault("setId", pair.set_id)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(_summary_line(payload))

    return 0


if __name__ == "__main__":
    sys.exit(main())
