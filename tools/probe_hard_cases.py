#!/usr/bin/env python3
"""Probe local recognizer robustness hard cases.

Unlike tools/probe_corpus.py, this harness does not require labelled ground
truth. It records input hashes, current recognizer status, failed checks, and
whether issue-specific target checks have been cleared.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_recognition_pair import file_sha256, parse_ground_truth, score_match  # noqa: E402
from rubik_recognizer.colors import rgb_to_hsv  # noqa: E402
from rubik_recognizer.recognizer import (  # noqa: E402
    WhiteUpRecognizer,
    _apply_pair_color_calibration,
    _assigned_grid_by_face,
    _public_repair_detail,
    _recognition_workset,
    _validation_failed_checks,
    _white_up_checks,
)
from rubik_recognizer.validation import validate_state  # noqa: E402


DEFAULT_MANIFEST = ROOT / "tests" / "fixtures" / "hard_case_manifest.json"


def load_manifest_document(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest {path} must be a JSON object.")
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError(f"Manifest {path} must contain a top-level 'pairs' array.")
    return data


def normalize_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def verify_hash(path: Path, expected: Optional[str]) -> Dict[str, Any]:
    actual = file_sha256(str(path))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "actual": actual,
        "expected": expected,
        "matches": expected in (None, "", actual),
    }


def missing_paths(paths: Sequence[Path]) -> List[str]:
    return [str(path) for path in paths if not path.exists()]


def linked_issue(row: Dict[str, Any]) -> Any:
    return row.get("linkedIssue", row.get("issue"))


def target_failures(row: Dict[str, Any], payload: Dict[str, Any], *, input_drift: bool) -> List[str]:
    failures: List[str] = []
    failed_checks = set(payload.get("failedChecks") or [])
    for check in row.get("targetFailedChecksAbsent") or []:
        if check in failed_checks:
            failures.append(f"target_check_still_present:{check}")
    for check in row.get("targetFailedChecksPresent") or []:
        if check not in failed_checks:
            failures.append(f"target_check_missing:{check}")
    expected_status = row.get("targetStatus")
    if expected_status and payload.get("status") != expected_status:
        failures.append(f"target_status_mismatch:{expected_status}")
    expected_category = row.get("targetCategory")
    if expected_category and payload.get("recognitionCategory") != expected_category:
        failures.append(f"target_category_mismatch:{expected_category}")
    expected_score = row.get("expectedScoreOnceFixed")
    if expected_score is not None and payload.get("score", -1) < int(expected_score):
        failures.append(f"score_below_expected_once_fixed:{expected_score}")
    if input_drift:
        failures.append("image_input_drift")
    return failures


def grid_cell_diagnostics(analysis: Any, anchor: str) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}
    for face, grid in _assigned_grid_by_face(analysis, anchor).items():
        diagnostics[face] = {
            "gridId": grid.id,
            "centerFace": grid.center_face,
            "matchedCount": grid.matched_count,
            "fitError": round(grid.fit_error, 3),
            "cells": [
                [
                    {
                        "id": sticker.id,
                        "source": sticker.source,
                        "rgb": list(sticker.rgb),
                        "hsv": [round(value, 4) for value in rgb_to_hsv(sticker.rgb)],
                        "color": sticker.match.color,
                        "face": sticker.match.face,
                        "confidence": round(sticker.match.confidence, 4),
                        "alternatives": [
                            {"color": color, "distance": round(float(distance), 4)}
                            for color, distance in sticker.match.alternatives[:4]
                        ],
                    }
                    for sticker in row
                ]
                for row in grid.stickers
            ],
        }
    return diagnostics


def image_diagnostics(result: Any, *, include_grid_cells: bool) -> Dict[str, Any]:
    images = (("imageA", result.image_a, "U"), ("imageB", result.image_b, "D"))
    diagnostics: Dict[str, Any] = {}
    for label, analysis, anchor in images:
        if analysis is None:
            continue
        image_summary = {
            "stickerColorCounts": dict(Counter(sticker.match.color for sticker in analysis.stickers)),
            "gridCenterFaceCounts": dict(Counter(grid.center_face for grid in analysis.grids)),
        }
        if include_grid_cells:
            image_summary["selectedGridCells"] = grid_cell_diagnostics(analysis, anchor)
        diagnostics[label] = image_summary
    return diagnostics


def repair_probe(
    result: Any,
    recognizer: WhiteUpRecognizer,
    *,
    expected_state: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if result.image_a is None or result.image_b is None:
        return None
    calibrated_a = copy.deepcopy(result.image_a)
    calibrated_b = copy.deepcopy(result.image_b)
    _apply_pair_color_calibration(calibrated_a, calibrated_b)
    return {
        "raw": repair_probe_for_analyses(
            result.image_a,
            result.image_b,
            recognizer,
            expected_state=expected_state,
        ),
        "calibrated": repair_probe_for_analyses(
            calibrated_a,
            calibrated_b,
            recognizer,
            expected_state=expected_state,
        ),
    }


def repair_probe_for_analyses(
    analysis_a: Any,
    analysis_b: Any,
    recognizer: WhiteUpRecognizer,
    *,
    expected_state: Optional[str] = None,
) -> Dict[str, Any]:
    start = time.perf_counter()
    checks = _white_up_checks(analysis_a, analysis_b)
    if checks:
        return {
            "status": "white_up_rejected",
            "whiteUpChecks": checks,
            "timings": {"totalSeconds": round(time.perf_counter() - start, 4)},
        }

    workset = _recognition_workset(analysis_a, analysis_b)
    merged_candidate_count = len(workset.merged_candidates)
    candidates = recognizer._state_candidates_from_workset(workset)
    invalid_reasons: List[str] = []
    direct_legal_count = 0
    for state, _, _ in candidates:
        validation = validate_state(state)
        if validation.valid:
            direct_legal_count += 1
        else:
            invalid_reasons.extend(validation.errors)
    failed_checks = _validation_failed_checks(invalid_reasons, analysis_a, analysis_b)

    repair_start = time.perf_counter()
    repair_details = recognizer._legal_repair_candidate_details_from_workset(workset, release_merged_candidates=True)
    return {
        "status": "probed",
        "optionsA": len(workset.options_a),
        "optionsB": len(workset.options_b),
        "mergedCandidateCount": merged_candidate_count,
        "directCandidateCount": len(candidates),
        "directLegalCount": direct_legal_count,
        "directFailedChecks": failed_checks,
        "repairCandidateCount": len(repair_details),
        "topRepairCandidates": repair_probe_public_details(repair_details[:3], expected_state=expected_state),
        "timings": {
            "repairSeconds": round(time.perf_counter() - repair_start, 4),
            "totalSeconds": round(time.perf_counter() - start, 4),
        },
    }


def repair_probe_public_details(
    repair_details: Sequence[Dict[str, Any]],
    *,
    expected_state: Optional[str] = None,
) -> List[Dict[str, Any]]:
    public = []
    for item in repair_details:
        detail = _public_repair_detail(item)
        if expected_state:
            detail["score"] = score_match(str(detail.get("state") or ""), expected_state)
        public.append(detail)
    return public


def probe_pair(
    row: Dict[str, Any],
    manifest_path: Path,
    recognizer: WhiteUpRecognizer,
    *,
    include_grid_cells: bool = False,
    include_repair_probe: bool = False,
) -> Dict[str, Any]:
    start = time.perf_counter()
    set_id = str(row.get("setId") or row.get("id") or "")
    image_a = normalize_path(str(row["imageAPath"]), manifest_path)
    image_b = normalize_path(str(row["imageBPath"]), manifest_path)
    truth_path = normalize_path(str(row["groundTruthPath"]), manifest_path) if row.get("groundTruthPath") else None
    missing = missing_paths(tuple(path for path in (image_a, image_b, truth_path) if path is not None))
    if missing:
        return {
            "setId": set_id,
            "linkedIssue": linked_issue(row),
            "failureClass": row.get("failureClass"),
            "status": "skipped",
            "category": "missing_files",
            "reason": "One or more manifest paths were not found.",
            "missingFiles": missing,
            "targetPassed": False,
            "targetFailures": ["missing_files"],
            "timings": {"totalSeconds": round(time.perf_counter() - start, 4)},
        }

    image_hashes = {
        "imageA": verify_hash(image_a, row.get("imageA_sha256_expected")),
        "imageB": verify_hash(image_b, row.get("imageB_sha256_expected")),
    }
    ground_truth_hash = verify_hash(truth_path, row.get("groundTruth_sha256_expected")) if truth_path is not None else None
    input_drift = not all(item["matches"] for item in [*image_hashes.values(), *([ground_truth_hash] if ground_truth_hash else [])])

    result = recognizer.recognize(image_a.read_bytes(), image_b.read_bytes())
    payload = result.to_api_dict(include_overlays=False)
    signals = payload.get("recognitionSignals") or {}
    score: Optional[int] = None
    canonical_state: Optional[str] = None
    if truth_path is not None:
        _, _, canonical_state, _ = parse_ground_truth(str(truth_path))
        score = score_match(payload.get("state") or "", canonical_state)
        payload["score"] = score
    failures = target_failures(row, payload, input_drift=input_drift)
    has_target = any(
        row.get(key) is not None
        for key in (
            "targetFailedChecksAbsent",
            "targetFailedChecksPresent",
            "targetStatus",
            "targetCategory",
            "expectedScoreOnceFixed",
        )
    )

    result_payload = {
        "setId": set_id,
        "linkedIssue": linked_issue(row),
        "failureClass": row.get("failureClass"),
        "status": payload.get("status"),
        "category": payload.get("recognitionCategory"),
        "categoryReason": payload.get("recognitionCategoryReason"),
        "reason": payload.get("reason"),
        "confidence": payload.get("confidence"),
        "score": score,
        "failedChecks": payload.get("failedChecks") or [],
        "candidateCount": payload.get("candidates"),
        "imageHashes": image_hashes,
        "groundTruthHash": ground_truth_hash,
        "inputDrift": input_drift,
        "currentStatus": row.get("currentStatus"),
        "currentCategory": row.get("currentCategory"),
        "currentFailedChecks": row.get("currentFailedChecks", row.get("baselineFailedChecks")) or [],
        "currentCandidates": row.get("currentCandidates"),
        "targetFailedChecksAbsent": row.get("targetFailedChecksAbsent") or [],
        "targetFailedChecksPresent": row.get("targetFailedChecksPresent") or [],
        "expectedScoreOnceFixed": row.get("expectedScoreOnceFixed"),
        "targetPassed": (not failures) if has_target else None,
        "targetFailures": failures,
        "pairColorCalibration": signals.get("pairColorCalibration"),
        "imageDiagnostics": image_diagnostics(result, include_grid_cells=include_grid_cells),
        "timings": {"totalSeconds": round(time.perf_counter() - start, 4)},
    }
    if include_repair_probe:
        result_payload["repairProbe"] = repair_probe(result, recognizer, expected_state=canonical_state)
    return result_payload


def probe_rows(
    rows: Sequence[Dict[str, Any]],
    manifest_path: Path,
    *,
    progress: bool = True,
    include_grid_cells: bool = False,
    include_repair_probe: bool = False,
) -> List[Dict[str, Any]]:
    recognizer = WhiteUpRecognizer()
    results: List[Dict[str, Any]] = []
    total = len(rows)
    for index, row in enumerate(rows, 1):
        if progress:
            set_id = row.get("setId") or row.get("id") or "?"
            print(f"[{index}/{total}] probing hard case set {set_id}", file=sys.stderr, flush=True)
        results.append(
            probe_pair(
                row,
                manifest_path,
                recognizer,
                include_grid_cells=include_grid_cells,
                include_repair_probe=include_repair_probe,
            )
        )
    return results


def write_json(path: Path, results: Sequence[Dict[str, Any]], manifest: Path) -> None:
    payload = {
        "schemaVersion": 1,
        "manifest": str(manifest),
        "results": list(results),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_table(results: Sequence[Dict[str, Any]]) -> None:
    print("Set Issue Class                         Status    Category            Score  Cands   Checks")
    print("--- ----- ----------------------------- --------- ------------------- ------ ------- ------------------------------")
    for result in results:
        checks = ",".join(result.get("failedChecks") or [])
        target = result.get("targetPassed")
        marker = " target=pass" if target is True else " target=FAIL" if target is False else ""
        score = "" if result.get("score") is None else f"{result.get('score')}/54"
        candidate_count = "" if result.get("candidateCount") is None else str(result.get("candidateCount"))
        print(
            f"{result.get('setId', ''):>3} "
            f"{str(result.get('linkedIssue') or ''):>5} "
            f"{str(result.get('failureClass') or ''):<29.29} "
            f"{str(result.get('status') or ''):<9.9} "
            f"{str(result.get('category') or ''):<19.19} "
            f"{score:>6} "
            f"{candidate_count:>7} "
            f"{checks}{marker}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Hard-case manifest JSON path.")
    parser.add_argument("--set-id", action="append", help="Only run one or more set ids from the manifest.")
    parser.add_argument("--json-output", help="Optional path for full JSON output.")
    parser.add_argument("--quiet", action="store_true", help="Do not print the readable table.")
    parser.add_argument("--no-progress", action="store_true", help="Do not print row-level probe progress to stderr.")
    parser.add_argument("--include-grid-cells", action="store_true", help="Include per-cell RGB/classification diagnostics for assigned grids.")
    parser.add_argument(
        "--include-repair-probe",
        action="store_true",
        help="Run the expensive direct/repair candidate probe for raw and calibrated analyses.",
    )
    parser.add_argument("--fail-on-target", action="store_true", help="Exit non-zero if any targeted row misses its target.")
    args = parser.parse_args()

    manifest = Path(args.manifest).expanduser()
    if not manifest.is_absolute():
        manifest = (Path.cwd() / manifest).resolve()
    selected = {str(item) for item in args.set_id or []}
    manifest_document = load_manifest_document(manifest)
    rows = [
        dict(item)
        for item in manifest_document["pairs"]
        if not selected or str(item.get("setId") or item.get("id")) in selected
    ]

    results = probe_rows(
        rows,
        manifest,
        progress=not args.no_progress,
        include_grid_cells=args.include_grid_cells,
        include_repair_probe=args.include_repair_probe,
    )
    if args.json_output:
        write_json(Path(args.json_output).expanduser(), results, manifest)
    if not args.quiet:
        print_table(results)

    if args.fail_on_target and any(result.get("targetPassed") is False for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
