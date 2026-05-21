#!/usr/bin/env python
"""Validate the per-sticker confidence signal against cv-local accuracy.

For each labeled-corpus pair: runs cv-local, extracts the new
`perStickerConfidence` recognition signal, and groups by accuracy tier
(full-match / partial / failure / no-recognition) to show whether the
signal is predictive.

2026-05-22 baseline:
    full-match (n=10):     min_conf median 0.24
    partial (n=6):         min_conf median 0.19
    failure (n=4):         min_conf median 0.15
    no-recognition (n=8):  min_conf median 0.05

Monotonic with accuracy tier — confirms the signal is meaningful even
though the dynamic range is small (0-0.55 in practice).

Usage:
    .venv/bin/python tools/evaluate_per_sticker_confidence.py

    # Override threshold:
    RUBIK_PER_STICKER_CONFIDENCE_THRESHOLD=0.20 \\
        .venv/bin/python tools/evaluate_per_sticker_confidence.py
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


def _load_manifests():
    out = []
    for fname in ("hard_case_manifest.json", "corpus_manifest.json"):
        with (REPO_ROOT / "tests" / "fixtures" / fname).open() as f:
            out.append(json.load(f))
    return out


def _load_corrected(path):
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
        pa = Path(entry["imageAPath"])
        pb = Path(entry["imageBPath"])
        if not (pa.exists() and pb.exists()):
            continue
        corrected = _load_corrected(entry.get("groundTruthPath"))
        if not corrected:
            continue
        try:
            with pa.open("rb") as f:
                bytes_a = f.read()
            with pb.open("rb") as f:
                bytes_b = f.read()
            result = recognizer.recognize(bytes_a, bytes_b)
        except Exception:
            continue
        state = getattr(result, "state", None)
        signals = getattr(result, "recognition_signals", {}) or {}
        psc = signals.get("perStickerConfidence", {}) or {}
        per_pair = psc.get("perPair", {}) or {}
        matches = (
            sum(1 for a, b in zip(state, corrected) if a == b)
            if state and len(state) == 54
            else None
        )
        results.append({
            "setId": sid,
            "matches": matches,
            "min_conf": per_pair.get("min"),
            "median_conf": per_pair.get("median"),
            "below_thresh": psc.get("belowThresholdCount"),
            "threshold": psc.get("threshold"),
        })

    print(f"{'set':4s} {'matches':>8s} {'min_conf':>9s} {'med_conf':>9s} {'below_thr':>10s}")
    for r in sorted(results, key=lambda r: r["min_conf"] or 0):
        m = r["matches"]
        m_str = f"{m:>4d}/54" if m is not None else "  N/A "
        mc = r["min_conf"]; mc_str = f"{mc:>6.2f}" if mc is not None else " N/A "
        md = r["median_conf"]; md_str = f"{md:>6.2f}" if md is not None else " N/A "
        bt = r["below_thresh"]; bt_str = f"{bt:>4d}" if bt is not None else " N/A "
        print(f"{r['setId']:>4s} {m_str:>8s} {mc_str:>9s} {md_str:>9s} {bt_str:>10s}")

    tiers = [
        ("full-match", [r for r in results if r["matches"] == 54 and r["min_conf"] is not None]),
        ("partial (30-53)", [r for r in results if r["matches"] is not None and 30 <= r["matches"] < 54 and r["min_conf"] is not None]),
        ("failure (<30)", [r for r in results if r["matches"] is not None and r["matches"] < 30 and r["min_conf"] is not None]),
        ("no-recognition", [r for r in results if r["matches"] is None and r["min_conf"] is not None]),
    ]
    print()
    print("=== Distribution by accuracy tier ===")
    for name, group in tiers:
        if not group:
            continue
        mins = [r["min_conf"] for r in group]
        meds = [r["median_conf"] for r in group]
        bts = [r["below_thresh"] for r in group]
        print(
            f"  {name:>18s} (n={len(group)}):  "
            f"min_conf median={statistics.median(mins):.2f}  "
            f"median_conf median={statistics.median(meds):.2f}  "
            f"below_thresh median={int(statistics.median(bts))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
