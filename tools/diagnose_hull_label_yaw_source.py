#!/usr/bin/env python3
"""Diagnose yaw inference from hull-label slot center colors.

The hull-label Tier 1 geometry produces stable image slots:
``upper`` / ``right`` / ``front`` for side A and side B. The remaining
production question is where capture yaw should come from before those
slots are mapped to WCA faces.

This diagnostic reuses the committed slot/yaw trace from
``tools/diagnose_slot_yaw_assignment.py`` and asks a production-shaped
question: can the six rectified center stickers choose yaw without legacy
joint face-ID?

The score is intentionally simple. For each yaw candidate 0..3, map each
slot to its expected WCA center face via ``corner_conventions`` and count
how many of the observed rectified center faces match. The winner is accepted
only with enough matches and margin over the runner-up.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnose_slot_yaw_assignment import _manifest_by_set, manifest_yaw_source  # noqa: E402
from tools.hull_label_yaw import (  # noqa: E402
    MIN_ACCEPTED_CENTER_MARGIN,
    MIN_ACCEPTED_CENTER_MATCHES,
    ObservedCenter,
    score_yaw_candidates,
)


DEFAULT_TRACE = REPO_ROOT / "tests" / "fixtures" / "hull_label_slot_yaw_assignment.json"
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "hull_label_center_yaw_source.json"
DEFAULT_REPORT = REPO_ROOT / "tools" / "HULL_LABEL_CENTER_YAW_SOURCE.md"


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _observed_slot_centers(row: Mapping[str, Any]) -> Tuple[List[ObservedCenter], Optional[str]]:
    """Extract ``(side, slot, observed_center_face)`` from the yaw=0 trace."""
    evaluation = row.get("evaluations", {}).get("assumed_zero")
    if not isinstance(evaluation, Mapping):
        return [], "missing_assumed_zero_evaluation"
    if evaluation.get("status") != "assembled":
        return [], str(evaluation.get("status") or "not_assembled")
    centers: List[ObservedCenter] = []
    for face in evaluation.get("perFace", []):
        if not isinstance(face, Mapping):
            continue
        side = face.get("side")
        slot = face.get("slot")
        convention_faces = face.get("conventionFaces")
        if side not in {"A", "B"} or slot not in {"upper", "right", "front"}:
            continue
        if not isinstance(convention_faces, str) or len(convention_faces) != 9:
            continue
        centers.append((str(side), str(slot), convention_faces[4]))
    if len(centers) != 6:
        return centers, "incomplete_center_observations"
    return centers, None


def _known_manifest_yaw(pair: Optional[Mapping[str, Any]]) -> Optional[int]:
    if not pair:
        return None
    source = manifest_yaw_source(pair)
    if source is None:
        return None
    yaw = source.get("yawQuarterTurns")
    return int(yaw) % 4 if isinstance(yaw, int) else None


def _legacy_detected_yaw(row: Mapping[str, Any]) -> Optional[int]:
    meta = row.get("yawSourceMeta")
    if not isinstance(meta, Mapping):
        return None
    detected = meta.get("detected")
    if not isinstance(detected, Mapping):
        return None
    yaw = detected.get("yawQuarterTurns")
    return int(yaw) % 4 if isinstance(yaw, int) else None


def evaluate_rows(
    trace_rows: Sequence[Mapping[str, Any]],
    manifest_by_set: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in trace_rows:
        set_id = str(row.get("setId"))
        observed, reject_reason = _observed_slot_centers(row)
        inference = (
            score_yaw_candidates(observed)
            if reject_reason is None
            else {
                "accepted": False,
                "yawQuarterTurns": None,
                "bestYawQuarterTurns": None,
                "bestScore": None,
                "secondScore": None,
                "margin": None,
                "candidates": [],
            }
        )
        manifest_yaw = _known_manifest_yaw(manifest_by_set.get(set_id))
        legacy_yaw = _legacy_detected_yaw(row)
        inferred_yaw = inference.get("yawQuarterTurns")
        out.append({
            "setId": set_id,
            "accepted": bool(inference["accepted"]),
            "rejectReason": reject_reason,
            "inferredYawQuarterTurns": inferred_yaw,
            "bestYawQuarterTurns": inference.get("bestYawQuarterTurns"),
            "bestScore": inference.get("bestScore"),
            "secondScore": inference.get("secondScore"),
            "margin": inference.get("margin"),
            "manifestYawQuarterTurns": manifest_yaw,
            "legacyDetectedYawQuarterTurns": legacy_yaw,
            "matchesManifest": (
                inferred_yaw == manifest_yaw
                if inferred_yaw is not None and manifest_yaw is not None
                else None
            ),
            "matchesLegacyDetected": (
                inferred_yaw == legacy_yaw
                if inferred_yaw is not None and legacy_yaw is not None
                else None
            ),
            "observedCenters": [
                {"side": side, "slot": slot, "centerFace": center}
                for side, slot, center in observed
            ],
            "candidates": inference["candidates"],
        })
    return out


def _count_true(rows: Iterable[Mapping[str, Any]], key: str) -> int:
    return sum(1 for row in rows if row.get(key) is True)


def build_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    accepted = [row for row in rows if row.get("accepted")]
    manifest_known = [row for row in accepted if row.get("manifestYawQuarterTurns") is not None]
    legacy_known = [row for row in accepted if row.get("legacyDetectedYawQuarterTurns") is not None]
    legacy_missing = [
        row for row in accepted
        if row.get("legacyDetectedYawQuarterTurns") is None
    ]
    rejected_reasons = Counter(str(row.get("rejectReason")) for row in rows if not row.get("accepted"))
    score_buckets = Counter(str(row.get("bestScore")) for row in accepted)
    margin_buckets = Counter(str(row.get("margin")) for row in accepted)
    return {
        "rows": len(rows),
        "accepted": len(accepted),
        "rejected": len(rows) - len(accepted),
        "rejectedReasons": dict(sorted(rejected_reasons.items())),
        "manifestKnown": len(manifest_known),
        "manifestAgreement": _count_true(manifest_known, "matchesManifest"),
        "legacyDetectedAvailable": len(legacy_known),
        "legacyDetectedAgreement": _count_true(legacy_known, "matchesLegacyDetected"),
        "legacyDetectedMissingButInferred": len(legacy_missing),
        "bestScoreBuckets": dict(sorted(score_buckets.items())),
        "marginBuckets": dict(sorted(margin_buckets.items())),
        "acceptedThresholds": {
            "minMatches": MIN_ACCEPTED_CENTER_MATCHES,
            "minMargin": MIN_ACCEPTED_CENTER_MARGIN,
        },
    }


def write_report(path: Path, *, trace_path: Path, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Hull-Label Center-Color Yaw Source Diagnostic",
        "",
        "## Purpose",
        "",
        "This diagnostic evaluates a production-shaped yaw source for the",
        "hull-label path: infer capture yaw from the six rectified slot center",
        "stickers, then map `upper` / `right` / `front` slots to WCA faces via",
        "`tools.corner_conventions.wca_face_by_slot()`.",
        "",
        "The inference accepts a yaw only when the best candidate has at least",
        f"{MIN_ACCEPTED_CENTER_MATCHES}/6 center matches and a margin of at least",
        f"{MIN_ACCEPTED_CENTER_MARGIN} over the runner-up.",
        "",
        f"Git head: `{_git_head_sha()}`",
        f"Generated: `{_dt.datetime.now(_dt.timezone.utc).isoformat()}`",
        f"Input trace: `{_rel(trace_path)}`",
        "",
        "## Summary",
        "",
        f"- Rows evaluated: {summary['rows']}",
        f"- Accepted yaw inference: {summary['accepted']} / {summary['rows']}",
        f"- Rejected: {summary['rejected']} (`{summary['rejectedReasons']}`)",
        f"- Agreement with manifest/human-known yaw: {summary['manifestAgreement']} / {summary['manifestKnown']}",
        f"- Agreement with legacy `captureYaw` when available: {summary['legacyDetectedAgreement']} / {summary['legacyDetectedAvailable']}",
        f"- Rows inferred where legacy `captureYaw` was missing: {summary['legacyDetectedMissingButInferred']}",
        f"- Best-score buckets: `{summary['bestScoreBuckets']}`",
        f"- Margin buckets: `{summary['marginBuckets']}`",
        "",
        "## Recommendation",
        "",
        "Use center-color yaw inference as the production hull-label yaw source.",
        "It is available from the hull-label rectified faces themselves, so it",
        "survives rows where the legacy recognizer rejects before emitting",
        "`captureYaw`. If CubeSnap later has explicit capture-yaw metadata, pass",
        "it as an optional hint/override and cross-check it against center-color",
        "inference; a conflict should fall back rather than force a yaw.",
        "",
        "Do not depend on the legacy `captureYaw` signal for the hull-label path.",
        "It is useful diagnostic metadata, but it is missing exactly in the class",
        "of reject rows where the hull-label path is supposed to help.",
        "",
        "## Per-Pair Snapshot",
        "",
        "| Set | Inferred yaw | Score | Margin | Manifest yaw | Legacy detected yaw | Result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        manifest = row.get("manifestYawQuarterTurns")
        legacy = row.get("legacyDetectedYawQuarterTurns")
        if row.get("accepted"):
            if row.get("matchesManifest") is False:
                result = "MANIFEST_MISMATCH"
            elif row.get("matchesLegacyDetected") is False:
                result = "LEGACY_MISMATCH"
            elif legacy is None:
                result = "inferred_no_legacy_yaw"
            else:
                result = "ok"
        else:
            result = str(row.get("rejectReason") or "rejected")
        lines.append(
            f"| {row['setId']} | {row.get('inferredYawQuarterTurns')} | "
            f"{row.get('bestScore')} | {row.get('margin')} | "
            f"{manifest} | {legacy} | {result} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    manifest_by_set = _manifest_by_set(args.manifest)
    rows = evaluate_rows(trace.get("rows", []), manifest_by_set)
    summary = build_summary(rows)
    payload = {
        "schema": "hull_label_center_yaw_source_v1",
        "metadata": {
            "tool": "tools/diagnose_hull_label_yaw_source.py",
            "inputTrace": _rel(args.trace),
            "manifest": _rel(args.manifest),
            "gitHead": _git_head_sha(),
            "generatedAt": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "summary": summary,
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.report, trace_path=args.trace, summary=summary, rows=rows)
    print(
        f"accepted={summary['accepted']}/{summary['rows']} "
        f"manifest={summary['manifestAgreement']}/{summary['manifestKnown']} "
        f"legacy={summary['legacyDetectedAgreement']}/{summary['legacyDetectedAvailable']} "
        f"legacy_missing_inferred={summary['legacyDetectedMissingButInferred']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
