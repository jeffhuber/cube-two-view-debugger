#!/usr/bin/env python3
"""Fetch a CubeSnap iOS repro upload by ID and import it locally.

The iOS app's debug upload returns an ``uploadId``. This tool turns that ID
into a durable local repro folder under ``~/cube-corpus/ios-repro-bundles`` so
it can be replayed with ``tools/evaluate_ios_repro_bundles.py``.
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ios_repro_bundle import DEFAULT_CORPUS_ROOT, unpack_bundle


DEFAULT_BASE_URL = "https://api.cubesnap.app"
MAX_BUNDLE_BYTES = 50 * 1024 * 1024
SAFE_UPLOAD_ID_RE = re.compile(r"[\w.:-]+")


def upload_id_from_reference(reference: str) -> tuple[str, Optional[str]]:
    """Return (upload_id, base_url) from an upload id, path, or full URL."""
    value = reference.strip()
    if not value:
        raise ValueError("upload reference must not be empty")

    parsed = urllib.parse.urlparse(value)
    path = parsed.path if parsed.scheme and parsed.netloc else value
    parts = [part for part in path.split("/") if part]
    try:
        marker_index = parts.index("ios-repro-uploads")
    except ValueError:
        if "/" in value or value in {".", ".."} or ".." in value.split("/"):
            raise ValueError(f"could not find ios-repro upload id in {reference!r}")
        upload_id = value
    else:
        if marker_index + 1 >= len(parts):
            raise ValueError(f"missing ios-repro upload id in {reference!r}")
        upload_id = parts[marker_index + 1]

    if upload_id in {".", ".."} or not SAFE_UPLOAD_ID_RE.fullmatch(upload_id):
        raise ValueError(f"unsafe upload id: {upload_id!r}")
    base_url = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else None
    return upload_id, base_url


def bundle_url(upload_id: str, base_url: str = DEFAULT_BASE_URL) -> str:
    safe_id = urllib.parse.quote(upload_id, safe="")
    return f"{base_url.rstrip('/')}/runs/ios-repro-uploads/{safe_id}/bundle.json"


def download_bundle(url: str, *, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read(MAX_BUNDLE_BYTES + 1)
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError(f"downloaded bundle exceeds {MAX_BUNDLE_BYTES} bytes")
    return data


def fetch_upload(
    reference: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    out_root: Path = DEFAULT_CORPUS_ROOT,
    timeout: float = 20.0,
) -> Path:
    upload_id, inferred_base_url = upload_id_from_reference(reference)
    url = bundle_url(upload_id, inferred_base_url or base_url)
    data = download_bundle(url, timeout=timeout)
    out_dir = out_root / upload_id
    tmp = tempfile.NamedTemporaryFile(prefix=f"{upload_id}-", suffix=".json", delete=False)
    tmp_path = Path(tmp.name)
    try:
        tmp.write(data)
        tmp.close()
        target, _manifest, _decoded = unpack_bundle(tmp_path, out_dir=out_dir)
    finally:
        tmp.close()
        tmp_path.unlink(missing_ok=True)
    return target


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="Upload ID, /runs/... path, or full bundle/summary URL.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        target = fetch_upload(
            args.reference,
            base_url=args.base_url,
            out_root=args.out_root.expanduser(),
            timeout=args.timeout,
        )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"fetch_ios_repro_upload: {type(exc).__name__}: {exc}")
        return 1
    print(f"imported: {target}")
    print(f"summary: {target / 'summary.md'}")
    print(f"manifest: {target / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
