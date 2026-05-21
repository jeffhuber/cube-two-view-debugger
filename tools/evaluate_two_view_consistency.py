#!/usr/bin/env python
"""Run cv-local on the labeled corpus and report the two-view geometry
consistency signal alongside per-pair accuracy.

Validates whether the `twoViewGeometryConsistency.ratio` field added by
the recognizer's `_two_view_geometry_consistency` function is useful as
a quality signal — specifically, does a high A/B sticker-spacing ratio
correlate with bad recognition outcomes?

2026-05-22 baseline:
  full-match cases:   median ratio 1.03, max 1.19
  partial (30-53):    median ratio 1.08, max 1.14
  failure (<30):      median ratio 1.05, max 1.18
  no recognition:     median ratio 1.07, max 1.54

Conclusion: ratio > 1.20 → strong signal for "recognition will fail
entirely." Otherwise weakly predictive. Worth keeping as a signal but
not strong enough on its own to drive category decisions.

Usage:
    .venv/bin/python tools/evaluate_two_view_consistency.py

    # Override tolerance (default 1.4):
    RUBIK_TWO_VIEW_RATIO_TOLERANCE=1.2 \\
        .venv/bin/python tools/evaluate_two_view_consistency.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.recognizer import WhiteUpRecognizer  # noqa: E402


def _load_manifests() -> list:
    out = []
    for fname in ("hard_case_manifest.json", "corpus_manifest.json"):
        with (REPO_ROOT / "tests" / "fixtures" / fname).open() as f:
            out.append(json.load(f))
    return out


def _load_corrected(path: str | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data[0].get("corrected")
    except (KeyError, IndexError, json.JSONDecodeError):
        return None


def main() -> int:
    manifests = _load_manifests()
    pairs = [p for m in manifests for p in m["pairs"]]
    seen: set = set()
    unique = []
    for p in pairs:
        if p["setId"] in seen:
            continue
        seen.add(p["setId"])
        unique.append(p)

    recognizer = WhiteUpRecognizer()
    results = []
    for entry in unique:
        sid = entry["setId"]
        path_a = Path(entry["imageAPath"])
        path_b = Path(entry["imageBPath"])
        if not (path_a.exists() and path_b.exists()):
            continue
        corrected = _load_corrected(entry.get("groundTruthPath"))
        if not corrected:
            continue
        try:
            with path_a.open("rb") as f:
                bytes_a = f.read()
            with path_b.open("rb") as f:
                bytes_b = f.read()
            result = recognizer.recognize(bytes_a, bytes_b)
        except Exception:
            continue
        state = getattr(result, "state", None)
        signals = getattr(result, "recognition_signals", {}) or {}
        consistency = signals.get("twoViewGeometryConsistency", {}) or {}
        matches = (
            sum(1 for a, b in zip(state, corrected) if a == b)
            if state and len(state) == 54
            else None
        )
        results.append({
            "setId": sid,
            "matches": matches,
            "ratio": consistency.get("ratio"),
            "inconsistent": consistency.get("inconsistent", False),
            "spacing_a": consistency.get("spacingPxA"),
            "spacing_b": consistency.get("spacingPxB"),
            "tolerance": consistency.get("toleranceRatio"),
        })

    print(f"{'set':4s} {'matches':>8s} {'ratio':>7s} {'A_spc':>7s} {'B_spc':>7s} {'flag'}")
    for r in sorted(results, key=lambda r: r["ratio"] or 0):
        m = r["matches"]
        m_str = f"{m:>4d}/54" if m is not None else "  N/A "
        rt = r["ratio"]
        rt_str = f"{rt:>6.2f}" if rt is not None else "  N/A "
        a_str = f"{r['spacing_a']:>6.1f}" if r["spacing_a"] is not None else "  N/A "
        b_str = f"{r['spacing_b']:>6.1f}" if r["spacing_b"] is not None else "  N/A "
        flag = "INCONSISTENT" if r["inconsistent"] else ""
        print(f"{r['setId']:>4s} {m_str:>8s} {rt_str:>7s} {a_str:>7s} {b_str:>7s} {flag}")

    print()
    print("=== Spacing ratio distribution by accuracy tier ===")
    tiers = [
        ("full-match (54/54)", [r for r in results if r["matches"] == 54]),
        ("partial (30-53)", [r for r in results if r["matches"] is not None and 30 <= r["matches"] < 54]),
        ("failure (<30)", [r for r in results if r["matches"] is not None and r["matches"] < 30]),
        ("no recognition", [r for r in results if r["matches"] is None]),
    ]
    for name, group in tiers:
        ratios = [r["ratio"] for r in group if r["ratio"] is not None]
        if not ratios:
            continue
        print(
            f"  {name:>20s}  n={len(ratios):>2d}  ratio: median={statistics.median(ratios):.2f}  max={max(ratios):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
