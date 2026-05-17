#!/usr/bin/env python3
"""Score a hand-labeled calibration packet produced by
``prepare_label_calibration.py``.

Reads /tmp/cal-set-<id>-labels.txt (filled in by the user) and
/tmp/cal-set-<id>-table.json (saved metadata) and prints:

  * hand vs multiset agreement (= are multiset-assigned labels biased?)
  * hand vs classifier agreement (= what's the real classifier accuracy?)
  * the specific stickers where multiset and classifier disagree,
    with hand's verdict on each (which one was right?)

Verdict: if multiset agreement >= 95%, label bias is small and the
broader LLM-labeling experiment can be skipped (classifier IS the lever).
If multiset agreement < 90%, labels are noisy and the LLM-labeling
experiment is worth running.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

VALID_COLORS = {"white", "yellow", "red", "orange", "green", "blue"}


def parse_labels_file(path: Path) -> Dict[int, str]:
    in_block = False
    labels: Dict[int, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped == "HAND_LABELS:":
            in_block = True
            continue
        if not in_block:
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        n_str, color = stripped.split("=", 1)
        n_str = n_str.strip()
        color = color.strip().lower()
        if not n_str.isdigit():
            continue
        if not color:
            continue  # blank => skipped
        if color not in VALID_COLORS:
            print(f"  WARN: line `{stripped}` — color '{color}' not valid; skipping", file=sys.stderr)
            continue
        labels[int(n_str)] = color
    return labels


def score(set_id: str) -> int:
    labels_path = Path(f"/tmp/cal-set-{set_id}-labels.txt")
    table_path = Path(f"/tmp/cal-set-{set_id}-table.json")
    if not labels_path.exists():
        print(f"missing {labels_path}", file=sys.stderr)
        return 2
    if not table_path.exists():
        print(f"missing {table_path}", file=sys.stderr)
        return 2

    hand = parse_labels_file(labels_path)
    table = json.loads(table_path.read_text())
    rows = table["rows"]

    if not hand:
        print(f"no hand labels found in {labels_path}", file=sys.stderr)
        print("did you fill in the HAND_LABELS: block?", file=sys.stderr)
        return 1

    # Match rows to hand labels
    matched: List[Dict] = []
    for row in rows:
        n = row["n"]
        if n not in hand:
            continue
        matched.append({
            "n": n,
            "side": row["side"],
            "face": row["face"],
            "rgb": row["rgb"],
            "multiset": row["multiset"],
            "classifier": row["classifier"],
            "hand": hand[n],
        })

    if not matched:
        print("no overlap between hand labels and table", file=sys.stderr)
        return 1

    total = len(matched)
    multiset_agree = sum(1 for m in matched if m["multiset"] == m["hand"])
    classifier_agree = sum(1 for m in matched if m["classifier"] == m["hand"])

    print(f"# Set {set_id} calibration ({total} hand-labeled stickers of {len(rows)} in table)")
    print()
    print(f"hand vs multiset:    {multiset_agree}/{total} = {multiset_agree/total:.1%}")
    print(f"hand vs classifier:  {classifier_agree}/{total} = {classifier_agree/total:.1%}")
    print()

    # Confusion matrices
    def confusion(field: str) -> Dict:
        cm: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for m in matched:
            cm[m["hand"]][m[field]] += 1
        return cm

    def print_confusion(name: str, cm: Dict) -> None:
        print(f"## {name} (rows=hand truth, cols=predicted)")
        cols = sorted({c for r in cm.values() for c in r.keys()})
        print(f"  {'truth':<8s}  " + "  ".join(f"{c:<7s}" for c in cols) + "  total")
        for truth in sorted(cm.keys()):
            row = cm[truth]
            total_truth = sum(row.values())
            print(
                f"  {truth:<8s}  " +
                "  ".join(f"{row.get(c, 0):<7d}" for c in cols) +
                f"  {total_truth}"
            )
        print()

    print_confusion("Multiset confusion", confusion("multiset"))
    print_confusion("Classifier confusion", confusion("classifier"))

    # Disagreement detail
    multi_wrong = [m for m in matched if m["multiset"] != m["hand"]]
    cls_wrong = [m for m in matched if m["classifier"] != m["hand"]]
    both_wrong = [m for m in matched if m["multiset"] != m["hand"] and m["classifier"] != m["hand"]]
    multi_only = [m for m in matched if m["multiset"] != m["hand"] and m["classifier"] == m["hand"]]
    cls_only = [m for m in matched if m["multiset"] == m["hand"] and m["classifier"] != m["hand"]]

    print(f"breakdown of {total} samples:")
    print(f"  both correct:                    {total - len(multi_wrong) - len(cls_wrong) + len(both_wrong)}")
    print(f"  multiset wrong, classifier right: {len(multi_only)}")
    print(f"  classifier wrong, multiset right: {len(cls_only)}")
    print(f"  both wrong:                       {len(both_wrong)}")
    print()

    if multi_only:
        print("## Cases where multiset label is WORSE than classifier (label noise)")
        print(f"   n  side face    hand        multiset     classifier   rgb")
        for m in sorted(multi_only, key=lambda x: x["n"]):
            print(
                f"  {m['n']:>2}   {m['side']}    {m['face']}    "
                f"{m['hand']:<10s}  {m['multiset']:<10s}  {m['classifier']:<10s}  {tuple(m['rgb'])}"
            )
        print()

    if cls_only:
        print("## Cases where classifier is wrong but multiset corrected it (multiset working)")
        print(f"   n  side face    hand        multiset     classifier   rgb")
        for m in sorted(cls_only, key=lambda x: x["n"]):
            print(
                f"  {m['n']:>2}   {m['side']}    {m['face']}    "
                f"{m['hand']:<10s}  {m['multiset']:<10s}  {m['classifier']:<10s}  {tuple(m['rgb'])}"
            )
        print()

    # Verdict
    multi_acc = multiset_agree / total
    print("# Verdict")
    if multi_acc >= 0.95:
        print(f"  Multiset labels are clean ({multi_acc:.1%} >= 95%). The eval can")
        print(f"  trust these labels. If a learned classifier doesn't beat baseline")
        print(f"  on this data, it's because the CLASSIFIER is near-optimal, not")
        print(f"  because labels are noisy. ==> LLM-labeling experiment can be SKIPPED.")
    elif multi_acc >= 0.85:
        print(f"  Multiset labels are mostly clean ({multi_acc:.1%}). Some noise but")
        print(f"  not dominant. LLM-labeling experiment would marginally help; not")
        print(f"  the highest leverage. Could also try only-corpus-success subset.")
    else:
        print(f"  Multiset labels are NOISY ({multi_acc:.1%} < 85%). The eval can't")
        print(f"  measure a real classifier improvement through this noise. LLM-")
        print(f"  labeling experiment IS worth running before any classifier work.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("set_id")
    args = ap.parse_args()
    return score(args.set_id)


if __name__ == "__main__":
    sys.exit(main())
