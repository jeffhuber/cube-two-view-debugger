#!/usr/bin/env python3
"""
Recognition pair audit tool — produces a deterministic per-pair report
that lets two callers (e.g. Codex and Claude) verify they are running
identical recognizer code, against identical input bytes, and reaching
identical conclusions.

When recognition results disagree, this report localises the divergence:

  same image SHA256 + same recognizer git/python/lib versions
  + same recognized state, but different scores       → ground-truth
                                                        parsing differs
  same image SHA256 + same recognizer + different
    recognized state                                  → runtime path
                                                        differs (direct vs
                                                        api adapter, etc.)
  different image SHA256                              → input bytes differ
                                                        (file path or
                                                        re-encode along
                                                        the way)

Output is plain text, one field per line, designed to diff cleanly.
The same script invocation should produce byte-identical output on
two machines if everything aligns.

Usage:

    python tools/audit_recognition_pair.py \\
      --set-id 12 \\
      --image-a "/path/to/Set 12 - A.jpg" \\
      --image-b "/path/to/Set 12 - B.jpg" \\
      --ground-truth "/path/to/Set 12 - cube-ground-truth-...json" \\
      --mode direct

    # API mode (server must be running):
    python tools/audit_recognition_pair.py \\
      --set-id 12 \\
      --image-a "..." --image-b "..." \\
      --ground-truth "..." \\
      --mode api \\
      --api-url "http://127.0.0.1:8080/api/recognize"

For the cube-snap-side audit (third path: cube-snap's adapter), pass
``--mode api --api-url "..."`` plus ``--mode-label cube-snap-adapter``.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

MIN_PYTHON = (3, 11)
CODEX_PYTHON = Path("/Users/jhuber/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3")


def _dependencies_available() -> bool:
    if sys.version_info < MIN_PYTHON:
        return False
    try:
        import numpy  # noqa: F401
        import PIL  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _candidate_runtimes(root: Path) -> List[Path]:
    candidates: List[Path] = []
    env_python = os.environ.get("CUBE_PYTHON")
    if env_python:
        candidates.append(Path(env_python))
    candidates.append(root / ".venv" / "bin" / "python")
    candidates.append(CODEX_PYTHON)
    return candidates


def _rerun_with_dependency_runtime() -> None:
    if _dependencies_available():
        return

    root = Path(__file__).resolve().parents[1]
    current = Path(sys.executable).resolve()
    for candidate in _candidate_runtimes(root):
        if not candidate.exists():
            continue
        try:
            if candidate.resolve() == current:
                continue
        except OSError:
            continue
        os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]])

    print(
        "Missing audit runtime: Python >= 3.11 with NumPy and Pillow is required.\n"
        "Create the project environment with:\n"
        "  python3 -m venv .venv\n"
        "  .venv/bin/python -m pip install -r requirements.txt\n"
        "Then run either:\n"
        "  .venv/bin/python tools/audit_recognition_pair.py ...\n"
        "or:\n"
        "  tools/audit_recognition_pair.py ...\n"
        "The executable script will prefer CUBE_PYTHON, .venv/bin/python, then the Codex bundled runtime.",
        file=sys.stderr,
    )
    raise SystemExit(2)


_rerun_with_dependency_runtime()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size(path: str) -> int:
    return os.path.getsize(path)


def parse_ground_truth(path: str) -> Tuple[str, str, str, bool]:
    """Return (sha256, raw_corrected_state, canonical_expected_state,
    canonicalization_applied).

    The Fixer's save-record JSON shape is a top-level array of objects
    each with a ``corrected`` string. The raw chunk-by-chunk URFDLB
    string in that field may not match canonical URFDLB ordering — the
    Fixer exports faces in whatever rotation the user labelled them in.
    The recognizer's ``rubik_recognizer.dataset.parse_ground_truth``
    helper canonicalises by detecting non-canonical chunk rotations and
    rotating each face until the resulting state validates as a legal
    cube. The recognizer's API and bench harness score against this
    canonical form, so the audit MUST as well — otherwise the audit
    will report bogus low scores for any pair whose corrected state
    happened to be saved in a non-canonical rotation.

    We return both forms (raw + canonical) plus a flag indicating
    whether canonicalisation actually changed anything, so the report
    makes the transform explicit."""
    sha = file_sha256(path)
    with open(path, "rb") as f:
        raw_bytes = f.read()

    # Pull the literal `corrected` string for visibility — we want this
    # in the report regardless of whether canonicalisation moves it.
    parsed = json.loads(raw_bytes.decode("utf-8"))
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        raw_state = parsed[0].get("corrected")
    elif isinstance(parsed, dict):
        raw_state = parsed.get("corrected") or parsed.get("expectedState")
    else:
        raw_state = None
    if not isinstance(raw_state, str) or len(raw_state) != 54:
        raise ValueError(
            f"Could not extract a 54-char corrected state from {path}; "
            f"got {type(raw_state).__name__}={raw_state!r}"
        )
    raw_state = raw_state.strip().upper()

    # Canonicalise via the recognizer's own helper so the audit scores
    # against exactly the same expected-state string the recognizer's
    # dataset/bench code uses.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    from rubik_recognizer.dataset import parse_ground_truth as _ds_parse  # type: ignore

    truth_map = _ds_parse(raw_bytes, os.path.basename(path))
    if not truth_map:
        raise ValueError(
            f"rubik_recognizer.dataset.parse_ground_truth returned an "
            f"empty map for {path}; cannot resolve canonical state."
        )
    if len(truth_map) == 1:
        canonical_state = next(iter(truth_map.values()))
    else:
        # Multi-record JSON: pick the one whose raw matches our chosen
        # raw_state, falling back to first-by-key.
        canonical_state = None
        for value in truth_map.values():
            if value == raw_state:
                canonical_state = value
                break
        if canonical_state is None:
            canonical_state = next(iter(truth_map.values()))

    canonical_state = canonical_state.strip().upper()
    canonicalization_applied = canonical_state != raw_state
    return sha, raw_state, canonical_state, canonicalization_applied


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for i in range(len(a)) if a[i] != b[i])


def score_match(a: str, b: str) -> int:
    if len(a) != len(b):
        return 0
    return sum(1 for i in range(len(a)) if a[i] == b[i])


# -------------------------------------------------------------- recognizers --


def recognize_direct(image_a_path: str, image_b_path: str) -> Dict[str, Any]:
    """In-process recognition via WhiteUpRecognizer.recognize. Returns
    a normalised payload mirroring what the API returns."""
    # Import lazily so api-mode users don't need the recognizer's deps.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    from rubik_recognizer.recognizer import WhiteUpRecognizer  # type: ignore
    recognizer = WhiteUpRecognizer()
    with open(image_a_path, "rb") as f:
        a_bytes = f.read()
    with open(image_b_path, "rb") as f:
        b_bytes = f.read()
    result = recognizer.recognize(a_bytes, b_bytes)
    payload = result.to_api_dict(include_overlays=False)
    return payload


def recognize_api(
    image_a_path: str, image_b_path: str, api_url: str,
) -> Dict[str, Any]:
    """Multipart POST recognition. Mirrors what cube-snap's cv-local
    adapter and bench harness do — same envelope, same content type."""
    parsed = urlparse(api_url if "://" in api_url else f"http://{api_url}")
    host = parsed.hostname or "localhost"
    port = parsed.port or 80
    # Append ?slim=1 if the URL doesn't already include a query, mirroring
    # cube-snap's behaviour. The runtime block is preserved under slim mode.
    path = parsed.path or "/api/recognize"
    if parsed.query:
        path = f"{path}?{parsed.query}"
        if "slim=" not in parsed.query:
            path = path + "&slim=1"
    else:
        path = f"{path}?slim=1"

    boundary = "----audit-recognition-pair-boundary"
    parts: List[bytes] = []
    for name, p in (("imageA", image_a_path), ("imageB", image_b_path)):
        with open(p, "rb") as f:
            data = f.read()
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"; '
            f'filename="{os.path.basename(p)}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n".encode()
        )
        parts.append(data)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    conn = http.client.HTTPConnection(host, port, timeout=600)
    conn.request(
        "POST", path, body=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return json.loads(raw.decode("utf-8"))


# --------------------------------------------------------------- environment --


def _git_sha(cwd: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def runtime_environment() -> Dict[str, Any]:
    """Mirror the recognizer's `recognitionSignals.runtime` block where
    possible. When in --mode direct, this is the only source of those
    fields; in --mode api, the server's `runtime` block is preferred
    (recorded inside the response payload)."""
    env: Dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": ".".join(str(v) for v in sys.version_info[:3]),
    }
    try:
        import PIL  # type: ignore
        env["pillow_version"] = PIL.__version__
    except Exception:
        env["pillow_version"] = "unavailable"
    try:
        import numpy  # type: ignore
        env["numpy_version"] = numpy.__version__
    except Exception:
        env["numpy_version"] = "unavailable"
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["git_sha"] = _git_sha(here)
    env["recognizer_repo_root"] = here
    return env


# ------------------------------------------------------------------- report --


def _signal_field(signals: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = signals
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def emit_report(
    *,
    set_id: str,
    invocation_path: str,
    image_a_path: str,
    image_b_path: str,
    ground_truth_path: str,
    payload: Dict[str, Any],
    raw_state: str,
    canonical_state: str,
    canonicalization_applied: bool,
    ground_truth_sha: str,
    env: Dict[str, Any],
) -> None:
    """Print the audit report to stdout. One field per line; field
    order is stable so two reports diff cleanly. The audit scores the
    recognizer's output against the CANONICAL expected state — i.e.
    the string the recognizer's own dataset helper would emit for
    this JSON file — not the raw `corrected` string. Both forms are
    printed so a reader can see the canonicalisation transform."""
    recognized = payload.get("state") or ""
    score = score_match(recognized, canonical_state)
    ham = hamming(recognized, canonical_state)
    score_vs_raw = score_match(recognized, raw_state)

    # Prefer the runtime block recorded inside the response payload (when
    # it exists — i.e. mode=api), since it captures the SERVER's env, not
    # this script's. In mode=direct the script and the recognizer share
    # an interpreter, so env is the server.
    server_runtime = payload.get("runtime") or {}
    git_sha = (
        (server_runtime.get("git") or {}).get("sha")
        if isinstance(server_runtime.get("git"), dict)
        else None
    )
    server_python = (
        (server_runtime.get("python") or {}).get("version")
        if isinstance(server_runtime.get("python"), dict)
        else None
    )
    server_pillow = (
        (server_runtime.get("libraries") or {}).get("pillow")
        if isinstance(server_runtime.get("libraries"), dict)
        else None
    )
    server_numpy = (
        (server_runtime.get("libraries") or {}).get("numpy")
        if isinstance(server_runtime.get("libraries"), dict)
        else None
    )

    signals = payload.get("recognitionSignals") or {}
    selected = signals.get("selectedRepairCandidate") or {}

    out: List[Tuple[str, Any]] = [
        ("set_id", set_id),
        ("invocation_path", invocation_path),
        ("git_sha", git_sha or env["git_sha"]),
        ("python_executable", env["python_executable"]),
        ("python_version", server_python or env["python_version"]),
        ("pillow_version", server_pillow or env["pillow_version"]),
        ("numpy_version", server_numpy or env["numpy_version"]),
        ("", ""),
        ("imageA_path", image_a_path),
        ("imageA_sha256", file_sha256(image_a_path)),
        ("imageA_size", file_size(image_a_path)),
        ("imageB_path", image_b_path),
        ("imageB_sha256", file_sha256(image_b_path)),
        ("imageB_size", file_size(image_b_path)),
        ("", ""),
        ("groundTruth_path", ground_truth_path),
        ("groundTruth_sha256", ground_truth_sha),
        ("raw_corrected_state", raw_state),
        ("canonical_expected_state", canonical_state),
        ("canonicalization_applied", canonicalization_applied),
        ("", ""),
        ("recognized_state", recognized),
        ("hamming", ham),
        ("score", f"{score}/54"),
        ("score_vs_raw", f"{score_vs_raw}/54"),
        ("status", payload.get("status")),
        ("confidence", payload.get("confidence")),
        ("reason", payload.get("reason")),
        ("repairPathUsed", signals.get("repairPathUsed")),
        ("repairCandidateCount", signals.get("repairCandidateCount")),
        (
            "selectedRepairCandidate.repairRankingPenalty",
            selected.get("repairRankingPenalty"),
        ),
        ("selectedRepairCandidate.baseConfidence", selected.get("baseConfidence")),
        ("selectedRepairCandidate.repairChanges", selected.get("repairChanges")),
        (
            "selectedRepairCandidate.preRepairConflicts.totalConflicts",
            _signal_field(selected, ["preRepairConflicts", "totalConflicts"]),
        ),
    ]
    for k, v in out:
        if k == "":
            print()
        else:
            print(f"{k}: {v}")


# --------------------------------------------------------------------- main --


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--image-a", required=True)
    p.add_argument("--image-b", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--mode", required=True, choices=("direct", "api"))
    p.add_argument(
        "--api-url",
        default="http://127.0.0.1:8080/api/recognize",
        help="API endpoint when --mode api. Default localhost:8080.",
    )
    p.add_argument(
        "--mode-label",
        default=None,
        help="Override invocation_path label (e.g. cube-snap-adapter).",
    )
    args = p.parse_args()

    if not os.path.exists(args.image_a):
        print(f"ERROR: image-a not found: {args.image_a}", file=sys.stderr)
        return 2
    if not os.path.exists(args.image_b):
        print(f"ERROR: image-b not found: {args.image_b}", file=sys.stderr)
        return 2
    if not os.path.exists(args.ground_truth):
        print(f"ERROR: ground-truth not found: {args.ground_truth}", file=sys.stderr)
        return 2

    ground_truth_sha, raw_state, canonical_state, canon_applied = parse_ground_truth(
        args.ground_truth
    )

    if args.mode == "direct":
        invocation_path = args.mode_label or "direct-python"
        payload = recognize_direct(args.image_a, args.image_b)
    else:
        invocation_path = args.mode_label or "api-multipart"
        payload = recognize_api(args.image_a, args.image_b, args.api_url)

    emit_report(
        set_id=args.set_id,
        invocation_path=invocation_path,
        image_a_path=args.image_a,
        image_b_path=args.image_b,
        ground_truth_path=args.ground_truth,
        payload=payload,
        raw_state=raw_state,
        canonical_state=canonical_state,
        canonicalization_applied=canon_applied,
        ground_truth_sha=ground_truth_sha,
        env=runtime_environment(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
