#!/usr/bin/env python3
"""Probe a labelled local image corpus without changing recognizer behavior.

The probe is intentionally diagnostic. It runs the current recognizer against
each manifest pair, scores against canonical ground truth, checks image hashes,
and inspects the per-image oriented-option lists to separate likely failure
modes before any orientation-weight tuning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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
        "Missing probe runtime: Python >= 3.11 with NumPy and Pillow is required.\n"
        "Create the project environment with:\n"
        "  python3 -m venv .venv\n"
        "  .venv/bin/python -m pip install -r requirements.txt\n"
        "Then run either:\n"
        "  .venv/bin/python tools/probe_corpus.py ...\n"
        "or:\n"
        "  tools/probe_corpus.py ...\n"
        "The executable script will prefer CUBE_PYTHON, .venv/bin/python, then the Codex bundled runtime.",
        file=sys.stderr,
    )
    raise SystemExit(2)


_rerun_with_dependency_runtime()


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_recognition_pair import file_sha256, parse_ground_truth, score_match  # noqa: E402
from rubik_recognizer.colors import COLOR_TO_FACE  # noqa: E402
from rubik_recognizer.recognizer import (  # noqa: E402
    FACE_ORDER,
    WhiteUpRecognizer,
    _oriented_face_options,
)


DEFAULT_MANIFEST = ROOT / "tests" / "fixtures" / "corpus_manifest.json"
FAILURE_MODES = (
    "clean",
    "image_input_drift",
    "retake_or_low_confidence",
    "orientation_rank_failure",
    "color_or_merge_failure",
    "candidate_generation_failure",
    "unknown",
)


def normalize_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError(f"Manifest {path} must contain a top-level 'pairs' array.")
    return [dict(item) for item in pairs]


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for left, right in zip(a, b) if left != right)


def split_state(state: str) -> Dict[str, List[List[str]]]:
    matrices: Dict[str, List[List[str]]] = {}
    for face_index, face in enumerate(FACE_ORDER):
        chunk = state[face_index * 9 : (face_index + 1) * 9]
        matrices[face] = [list(chunk[row * 3 : (row + 1) * 3]) for row in range(3)]
    return matrices


def flatten_matrix(matrix: Sequence[Sequence[str]]) -> Tuple[str, ...]:
    return tuple(value for row in matrix for value in row)


def matrix_to_string(matrix: Optional[Sequence[Sequence[str]]]) -> Optional[str]:
    if matrix is None:
        return None
    return "".join(flatten_matrix(matrix))


def facelet_to_face(value: Any) -> str:
    if isinstance(value, str) and value in FACE_ORDER:
        return value
    match = getattr(value, "match", None)
    color = getattr(match, "color", None)
    if isinstance(color, str):
        return COLOR_TO_FACE.get(color, "?")
    return "?"


def option_face_matrix(option: Dict[str, Any], face: str) -> Optional[List[List[str]]]:
    matrix = option.get(face)
    if not matrix:
        return None
    return [[facelet_to_face(value) for value in row] for row in matrix]


def ranked_orientation_options(options: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(options, key=lambda item: float(item.get("_score", 0.0)), reverse=True)


def first_matching_rank(
    ranked: Sequence[Dict[str, Any]],
    face: str,
    target: Optional[Sequence[Sequence[str]]],
) -> Tuple[Optional[int], Optional[float]]:
    if target is None:
        return None, None
    target_string = matrix_to_string(target)
    for index, option in enumerate(ranked, start=1):
        candidate = option_face_matrix(option, face)
        if matrix_to_string(candidate) == target_string:
            return index, float(option.get("_score", 0.0))
    return None, None


def best_multiset_rank(
    ranked: Sequence[Dict[str, Any]],
    face: str,
    target: Sequence[Sequence[str]],
) -> Tuple[Optional[int], Optional[float]]:
    target_counter = Counter(flatten_matrix(target))
    for index, option in enumerate(ranked, start=1):
        candidate = option_face_matrix(option, face)
        if candidate is not None and Counter(flatten_matrix(candidate)) == target_counter:
            return index, float(option.get("_score", 0.0))
    return None, None


def classify_face_failure(
    *,
    status: str,
    input_drift: bool,
    selected_is_correct: bool,
    correct_matrix_generated: bool,
    correct_color_multiset_present: bool,
) -> str:
    if input_drift:
        return "image_input_drift"
    if status != "success":
        return "retake_or_low_confidence"
    if selected_is_correct:
        return "clean"
    if correct_matrix_generated:
        return "orientation_rank_failure"
    if correct_color_multiset_present:
        return "color_or_merge_failure"
    return "candidate_generation_failure"


def summarize_failure_modes(faces: Iterable[Dict[str, Any]], input_drift: bool, status: str) -> str:
    if input_drift:
        return "image_input_drift"
    if status != "success":
        return "retake_or_low_confidence"
    counts = Counter(face.get("failureMode", "unknown") for face in faces)
    for mode in (
        "orientation_rank_failure",
        "color_or_merge_failure",
        "candidate_generation_failure",
        "unknown",
    ):
        if counts[mode]:
            return mode
    return "clean"


def orientation_probe_for_image(
    *,
    image_label: str,
    analysis: Any,
    anchor: str,
    status: str,
    expected: Dict[str, List[List[str]]],
    recognized: Optional[Dict[str, List[List[str]]]],
    input_drift: bool,
) -> Dict[str, Any]:
    options = ranked_orientation_options(_oriented_face_options(analysis, anchor))
    faces = sorted({face for option in options for face in FACE_ORDER if option.get(face)})
    summaries = []
    for face in faces:
        expected_matrix = expected[face]
        selected_matrix = recognized.get(face) if recognized else None
        correct_rank, correct_score = first_matching_rank(options, face, expected_matrix)
        selected_rank, selected_score = first_matching_rank(options, face, selected_matrix)
        multiset_rank, multiset_score = best_multiset_rank(options, face, expected_matrix)
        selected_is_correct = selected_matrix == expected_matrix
        correct_generated = correct_rank is not None
        multiset_present = multiset_rank is not None
        gap = None
        if correct_score is not None and selected_score is not None:
            gap = round(selected_score - correct_score, 4)
        summaries.append(
            {
                "image": image_label,
                "face": face,
                "orientationOptionRank": correct_rank,
                "selectedOptionRank": selected_rank,
                "selectedVsCorrectScoreGap": gap,
                "selectedFaceScore": round(selected_score, 4) if selected_score is not None else None,
                "correctFaceScore": round(correct_score, 4) if correct_score is not None else None,
                "correctMatrixGenerated": correct_generated,
                "correctColorMultisetPresent": multiset_present,
                "correctColorMultisetRank": multiset_rank,
                "correctColorMultisetScore": round(multiset_score, 4) if multiset_score is not None else None,
                "selectedIsCorrect": selected_is_correct,
                "failureMode": classify_face_failure(
                    status=status,
                    input_drift=input_drift,
                    selected_is_correct=selected_is_correct,
                    correct_matrix_generated=correct_generated,
                    correct_color_multiset_present=multiset_present,
                ),
            }
        )
    return {
        "anchor": anchor,
        "orientationOptionCount": len(options),
        "faces": summaries,
    }


def verify_hash(path: Path, expected: Optional[str]) -> Dict[str, Any]:
    actual = file_sha256(str(path))
    return {
        "path": str(path),
        "actual": actual,
        "expected": expected,
        "matches": expected in (None, "", actual),
    }


def missing_paths(paths: Sequence[Path]) -> List[str]:
    return [str(path) for path in paths if not path.exists()]


def probe_pair(row: Dict[str, Any], manifest_path: Path) -> Dict[str, Any]:
    set_id = str(row.get("setId") or row.get("id") or "")
    image_a = normalize_path(str(row["imageAPath"]), manifest_path)
    image_b = normalize_path(str(row["imageBPath"]), manifest_path)
    truth_path = normalize_path(str(row["groundTruthPath"]), manifest_path)
    missing = missing_paths((image_a, image_b, truth_path))
    if missing:
        return {
            "setId": set_id,
            "status": "skipped",
            "category": "missing_files",
            "reason": "One or more manifest paths were not found.",
            "missingFiles": missing,
            "contractPassed": False,
        }

    image_hashes = {
        "imageA": verify_hash(image_a, row.get("imageA_sha256_expected")),
        "imageB": verify_hash(image_b, row.get("imageB_sha256_expected")),
        "groundTruth": verify_hash(truth_path, row.get("groundTruth_sha256_expected")),
    }
    input_drift = not all(item["matches"] for item in image_hashes.values())
    ground_truth_sha, raw_state, canonical_state, canonicalized = parse_ground_truth(str(truth_path))

    recognizer = WhiteUpRecognizer()
    result = recognizer.recognize(image_a.read_bytes(), image_b.read_bytes())
    payload = result.to_api_dict(include_overlays=False)
    recognized_state = payload.get("state") or ""
    score = score_match(recognized_state, canonical_state)
    score_vs_raw = score_match(recognized_state, raw_state)
    category = payload.get("recognitionCategory")
    signals = payload.get("recognitionSignals") or {}
    selected = signals.get("selectedRepairCandidate") or {}

    expected_matrix = split_state(canonical_state)
    recognized_matrix = split_state(recognized_state) if len(recognized_state) == 54 else None
    orientation = {
        "imageA": orientation_probe_for_image(
            image_label="imageA",
            analysis=result.image_a,
            anchor="U",
            status=payload.get("status"),
            expected=expected_matrix,
            recognized=recognized_matrix,
            input_drift=input_drift,
        )
        if result.image_a is not None
        else None,
        "imageB": orientation_probe_for_image(
            image_label="imageB",
            analysis=result.image_b,
            anchor="D",
            status=payload.get("status"),
            expected=expected_matrix,
            recognized=recognized_matrix,
            input_drift=input_drift,
        )
        if result.image_b is not None
        else None,
    }
    face_summaries = [
        face
        for image_probe in orientation.values()
        if isinstance(image_probe, dict)
        for face in image_probe.get("faces", [])
    ]
    primary_failure_mode = summarize_failure_modes(face_summaries, input_drift, payload.get("status"))

    expected_category = row.get("expectedCategory")
    expected_score_floor = row.get("expectedScoreFloor")
    category_ok = expected_category in (None, "", category)
    score_ok = expected_score_floor is None or score >= int(expected_score_floor)
    contract_passed = (not input_drift) and category_ok and score_ok

    return {
        "setId": set_id,
        "status": payload.get("status"),
        "category": category,
        "categoryReason": payload.get("recognitionCategoryReason"),
        "reason": payload.get("reason"),
        "confidence": payload.get("confidence"),
        "score": score,
        "scoreVsRaw": score_vs_raw,
        "hamming": hamming(recognized_state, canonical_state),
        "recognizedState": recognized_state,
        "rawCorrectedState": raw_state,
        "canonicalExpectedState": canonical_state,
        "canonicalizationApplied": canonicalized,
        "repairPathUsed": signals.get("repairPathUsed"),
        "candidateCount": payload.get("candidates"),
        "repairCandidateCount": signals.get("repairCandidateCount"),
        "repairRankingPenalty": selected.get("repairRankingPenalty"),
        "baseConfidence": selected.get("baseConfidence"),
        "repairChanges": selected.get("repairChanges"),
        "preRepairConflicts": selected.get("preRepairConflicts"),
        "imageHashes": image_hashes,
        "groundTruth_sha256": ground_truth_sha,
        "inputDrift": input_drift,
        "expectedCategory": expected_category,
        "expectedScoreFloor": expected_score_floor,
        "currentScoreObserved": row.get("currentScoreObserved"),
        "contractPassed": contract_passed,
        "contractFailures": [
            name
            for name, failed in (
                ("image_input_drift", input_drift),
                ("category_mismatch", not category_ok),
                ("score_below_floor", not score_ok),
            )
            if failed
        ],
        "primaryFailureMode": primary_failure_mode,
        "failureModes": dict(Counter(face.get("failureMode", "unknown") for face in face_summaries)),
        "orientationDiagnostics": orientation,
        "notes": row.get("notes"),
    }


def printable(value: Any, width: int) -> str:
    text = "" if value is None else str(value)
    if len(text) > width:
        text = text[: max(0, width - 1)] + "…"
    return text.ljust(width)


def smallest_rank_gaps(results: Sequence[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    rows = []
    for result in results:
        for image_key, image_probe in (result.get("orientationDiagnostics") or {}).items():
            if not isinstance(image_probe, dict):
                continue
            for face in image_probe.get("faces", []):
                gap = face.get("selectedVsCorrectScoreGap")
                if gap is None or face.get("selectedIsCorrect") or not face.get("correctMatrixGenerated"):
                    continue
                rows.append(
                    {
                        "setId": result.get("setId"),
                        "image": image_key,
                        "face": face.get("face"),
                        "gap": gap,
                        "mode": face.get("failureMode"),
                    }
                )
    return sorted(rows, key=lambda item: abs(float(item["gap"])))[:limit]


def print_table(results: Sequence[Dict[str, Any]]) -> None:
    headers = (
        ("Set", 5),
        ("Score", 8),
        ("Category", 33),
        ("Path", 7),
        ("Conf", 6),
        ("Pen", 6),
        ("Cands", 8),
        ("Mode", 30),
        ("Contract", 9),
    )
    print("".join(printable(label, width) for label, width in headers))
    print("-" * sum(width for _, width in headers))
    for result in results:
        if result.get("status") == "skipped":
            score = "skip"
            path = "-"
        else:
            score = f"{result.get('score')}/54"
            path = "repair" if result.get("repairPathUsed") else "direct"
        values = (
            result.get("setId"),
            score,
            result.get("category"),
            path,
            result.get("confidence"),
            result.get("repairRankingPenalty"),
            result.get("candidateCount"),
            result.get("primaryFailureMode"),
            "pass" if result.get("contractPassed") else "FAIL",
        )
        print("".join(printable(value, width) for value, (_, width) in zip(values, headers)))

    gaps = smallest_rank_gaps(results)
    if gaps:
        print("\nSmallest generated-correct orientation score gaps:")
        for row in gaps:
            print(
                f"  set {row['setId']} {row['image']} {row['face']}: "
                f"gap={row['gap']} mode={row['mode']}"
            )


def write_json(path: Path, results: Sequence[Dict[str, Any]], manifest: Path) -> None:
    payload = {
        "schemaVersion": 1,
        "manifest": str(manifest),
        "results": list(results),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(result, sort_keys=True) + "\n" for result in results),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Corpus manifest JSON path.")
    parser.add_argument("--set-id", action="append", help="Only run one or more set ids from the manifest.")
    parser.add_argument("--json-output", help="Optional path for full JSON output.")
    parser.add_argument("--jsonl-output", help="Optional path for JSONL output.")
    parser.add_argument("--quiet", action="store_true", help="Do not print the readable table.")
    parser.add_argument("--fail-on-contract", action="store_true", help="Exit non-zero if any non-skipped row fails its contract.")
    args = parser.parse_args()

    manifest = Path(args.manifest).expanduser()
    if not manifest.is_absolute():
        manifest = (Path.cwd() / manifest).resolve()
    selected = {str(item) for item in args.set_id or []}
    rows = [
        row
        for row in load_manifest(manifest)
        if not selected or str(row.get("setId") or row.get("id")) in selected
    ]
    results = [probe_pair(row, manifest) for row in rows]

    if args.json_output:
        write_json(Path(args.json_output).expanduser(), results, manifest)
    if args.jsonl_output:
        write_jsonl(Path(args.jsonl_output).expanduser(), results)
    if not args.quiet:
        print_table(results)

    if args.fail_on_contract and any(not result.get("contractPassed") and result.get("status") != "skipped" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
