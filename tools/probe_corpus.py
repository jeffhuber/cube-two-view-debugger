#!/usr/bin/env python3
"""Probe a labelled local image corpus without changing recognizer behavior.

The probe is intentionally diagnostic. It runs the current recognizer against
each manifest pair, scores against canonical ground truth, checks image hashes,
and inspects the per-image oriented-option lists to separate likely failure
modes before any orientation-weight tuning.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
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
from rubik_recognizer.image_pipeline import analyze_image  # noqa: E402
from rubik_recognizer.recognizer import (  # noqa: E402
    FACE_ORDER,
    WhiteUpRecognizer,
    _grid_quality_score,
    _oriented_face_options,
    recognition_diagnostics,
)


DEFAULT_MANIFEST = ROOT / "tests" / "fixtures" / "corpus_manifest.json"
FINGERPRINT_PATH_ALIASES = {
    "numpy.version": "packages.numpy.version",
    "pillow.version": "packages.pillow.version",
}
ENVIRONMENT_POLICY_METADATA_KEYS = {"label", "notes"}
FAILURE_MODES = (
    "clean",
    "image_input_drift",
    "retake_or_low_confidence",
    "orientation_rank_failure",
    "color_or_merge_failure",
    "candidate_generation_failure",
    "unknown",
)


def _capture_stdout(callback: Any) -> str:
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            callback()
    except Exception as exc:  # pragma: no cover - defensive runtime fingerprinting
        return f"{type(exc).__name__}: {exc}"
    return buffer.getvalue()


def _git_sha(cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def runtime_fingerprint() -> Dict[str, Any]:
    import numpy  # type: ignore
    import PIL  # type: ignore
    from PIL import features  # type: ignore

    return {
        "python": {
            "version": sys.version,
            "versionInfo": list(sys.version_info[:5]),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "build": list(platform.python_build()),
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {
            "numpy": {
                "version": numpy.__version__,
                "showConfig": _capture_stdout(numpy.show_config),
            },
            "pillow": {
                "version": PIL.__version__,
                "pilInfo": _capture_stdout(features.pilinfo),
            },
        },
        "git": {
            "sha": _git_sha(ROOT),
            "cwd": str(ROOT),
        },
    }


def runtime_summary(fingerprint: Dict[str, Any]) -> str:
    python_info = fingerprint.get("python") or {}
    platform_info = fingerprint.get("platform") or {}
    packages = fingerprint.get("packages") or {}
    numpy_info = packages.get("numpy") or {}
    pillow_info = packages.get("pillow") or {}
    version_info = python_info.get("versionInfo") or []
    python_version = ".".join(str(part) for part in version_info[:3]) or "unknown"
    return (
        f"Python {python_version} ({python_info.get('executable', 'unknown')}); "
        f"Pillow {pillow_info.get('version', 'unknown')}; "
        f"NumPy {numpy_info.get('version', 'unknown')}; "
        f"{platform_info.get('platform', 'unknown')}; "
        f"git {(fingerprint.get('git') or {}).get('sha', 'unknown')}"
    )


def _elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def normalize_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


def load_manifest_document(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Manifest {path} must be a JSON object.")
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError(f"Manifest {path} must contain a top-level 'pairs' array.")
    return data


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    data = load_manifest_document(path)
    pairs = data["pairs"]
    return [dict(item) for item in pairs]


def _fingerprint_value(fingerprint: Dict[str, Any], dotted_key: str) -> Any:
    path = FINGERPRINT_PATH_ALIASES.get(dotted_key, dotted_key)
    value: Any = fingerprint
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def environment_policy_warnings(
    manifest_document: Dict[str, Any],
    fingerprint: Dict[str, Any],
) -> List[Dict[str, Any]]:
    supported = manifest_document.get("supportedArchitectures")
    if not isinstance(supported, dict):
        return []
    primary = supported.get("primary")
    if not isinstance(primary, dict):
        return []

    mismatches = []
    for key, expected in primary.items():
        if key in ENVIRONMENT_POLICY_METADATA_KEYS:
            continue
        actual = _fingerprint_value(fingerprint, key)
        if actual != expected:
            mismatches.append({"key": key, "expected": expected, "actual": actual})
    if not mismatches:
        return []

    label = primary.get("label") or "primary"
    return [
        {
            "architecture": "primary",
            "label": label,
            "message": (
                f"Corpus manifest baseline expects {label}; current runtime differs. "
                "Scores/categories may be architecture-dependent."
            ),
            "mismatches": mismatches,
            "notes": primary.get("notes"),
        }
    ]


def print_environment_warnings(warnings: Sequence[Dict[str, Any]]) -> None:
    for warning in warnings:
        mismatch_text = "; ".join(
            f"{item['key']} expected {item['expected']!r} got {item['actual']!r}"
            for item in warning.get("mismatches", [])
        )
        detail = f" ({mismatch_text})" if mismatch_text else ""
        print(f"Warning: {warning.get('message')}{detail}", file=sys.stderr)


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for left, right in zip(a, b) if left != right)


def score_direct_legal_candidates(candidates: Any, expected_state: str) -> List[Dict[str, Any]]:
    scored = []
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        state = str(candidate.get("state") or "")
        item = dict(candidate)
        item["score"] = score_match(state, expected_state)
        item["hamming"] = hamming(state, expected_state)
        scored.append(item)
    return scored


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


def grid_weak_reasons(grid: Dict[str, Any]) -> List[str]:
    reasons = []
    matched = grid.get("matchedCount")
    fit_error = grid.get("fitError")
    quality = grid.get("quality")
    grid_samples = grid.get("gridSamples")
    bad_samples = grid.get("badSamples")
    suspect_samples = grid.get("suspectSamples")
    if isinstance(matched, (int, float)) and matched < 6:
        reasons.append("matchedCount_below_6")
    if isinstance(fit_error, (int, float)) and fit_error > 12.0:
        reasons.append("fitError_above_12")
    if isinstance(quality, (int, float)) and quality < 80.0:
        reasons.append("quality_below_80")
    if isinstance(grid_samples, (int, float)) and grid_samples >= 3:
        reasons.append("grid_sample_heavy")
    if isinstance(bad_samples, (int, float)) and bad_samples > 0:
        reasons.append("bad_samples_present")
    if isinstance(suspect_samples, (int, float)) and suspect_samples > 2.5:
        reasons.append("suspect_samples_present")
    span = grid.get("gridSpanContamination") if isinstance(grid.get("gridSpanContamination"), dict) else {}
    span_score = span.get("score")
    shape_spread = span.get("componentShapeSpread")
    extrapolated_cells = span.get("extrapolatedCellCount")
    outside_cells = span.get("sampleCellsOutsideGridComponentHull")
    if isinstance(span_score, (int, float)) and span_score >= 8.0:
        reasons.append("grid_span_contamination_score_ge_8")
    if isinstance(shape_spread, (int, float)) and shape_spread >= 32.0:
        reasons.append("component_shape_spread_ge_32")
    if isinstance(extrapolated_cells, (int, float)) and extrapolated_cells >= 3:
        reasons.append("extrapolated_cells_ge_3")
    if isinstance(outside_cells, (int, float)) and outside_cells >= 3:
        reasons.append("sample_cells_outside_grid_component_hull_ge_3")
    return reasons


def selected_grid_span_summary(signals: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for image_key, grids in (signals.get("selectedGridQuality") or {}).items():
        if not isinstance(grids, dict):
            continue
        for face, grid in grids.items():
            if not isinstance(grid, dict):
                continue
            span = grid.get("gridSpanContamination") if isinstance(grid.get("gridSpanContamination"), dict) else {}
            row = {
                "image": image_key,
                "face": face,
                "gridId": grid.get("gridId"),
                "score": span.get("score"),
                "componentShapeSpread": span.get("componentShapeSpread"),
                "componentShapeAngleCount": span.get("componentShapeAngleCount"),
                "sampledCellCount": span.get("sampledCellCount"),
                "extrapolatedCellCount": span.get("extrapolatedCellCount"),
                "unsupportedCellCount": span.get("unsupportedCellCount"),
                "badSampleCellCount": span.get("badSampleCellCount"),
                "cubeHullOutsideCount": span.get("cubeHullOutsideCount"),
                "maxOutsideGridComponentHullRatio": span.get("maxOutsideGridComponentHullRatio"),
                "maxNearestGridComponentRatio": span.get("maxNearestGridComponentRatio"),
                "sampleCellsOutsideGridComponentHull": span.get("sampleCellsOutsideGridComponentHull"),
                "sampleCellsFarFromGridComponents": span.get("sampleCellsFarFromGridComponents"),
            }
            rows.append(row)
    scores = [float(row["score"]) for row in rows if isinstance(row.get("score"), (int, float))]
    shape_spreads = [
        float(row["componentShapeSpread"])
        for row in rows
        if isinstance(row.get("componentShapeSpread"), (int, float))
    ]
    outside_ratios = [
        float(row["maxOutsideGridComponentHullRatio"])
        for row in rows
        if isinstance(row.get("maxOutsideGridComponentHullRatio"), (int, float))
    ]
    nearest_ratios = [
        float(row["maxNearestGridComponentRatio"])
        for row in rows
        if isinstance(row.get("maxNearestGridComponentRatio"), (int, float))
    ]
    return {
        "rows": rows,
        "maxScore": round(max(scores, default=0.0), 3),
        "maxComponentShapeSpread": round(max(shape_spreads, default=0.0), 3),
        "maxOutsideGridComponentHullRatio": round(max(outside_ratios, default=0.0), 3),
        "maxNearestGridComponentRatio": round(max(nearest_ratios, default=0.0), 3),
        "totalSampledCells": sum(
            int(row.get("sampledCellCount") or 0)
            for row in rows
            if isinstance(row.get("sampledCellCount"), (int, float))
        ),
        "totalExtrapolatedCells": sum(
            int(row.get("extrapolatedCellCount") or 0)
            for row in rows
            if isinstance(row.get("extrapolatedCellCount"), (int, float))
        ),
    }


def selected_grid_health(signals: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    weak = []
    for image_key, grids in (signals.get("selectedGridQuality") or {}).items():
        if not isinstance(grids, dict):
            continue
        for face, grid in grids.items():
            if not isinstance(grid, dict):
                continue
            reasons = grid_weak_reasons(grid)
            row = {
                "image": image_key,
                "face": face,
                "gridId": grid.get("gridId"),
                "matchedCount": grid.get("matchedCount"),
                "fitError": grid.get("fitError"),
                "quality": grid.get("quality"),
                "gridSamples": grid.get("gridSamples"),
                "badSamples": grid.get("badSamples"),
                "suspectSamples": grid.get("suspectSamples"),
                "gridSpanContamination": grid.get("gridSpanContamination"),
                "weakReasons": reasons,
            }
            rows.append(row)
            if reasons:
                weak.append(row)
    return {
        "selectedGrids": rows,
        "weakSelectedGrids": weak,
        "weakSelectedGridCount": len(weak),
    }


def grid_group_breakdown(diagnostics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(diagnostics, dict):
        return {}
    by_image = {}
    for image_key in ("imageA", "imageB"):
        image_diag = diagnostics.get(image_key)
        if not isinstance(image_diag, dict):
            continue
        groups = image_diag.get("gridGroups") or {}
        by_image[image_key] = {
            face: {
                "count": len(grids) if isinstance(grids, list) else 0,
                "matchedCounts": [grid.get("matchedCount") for grid in grids[:8] if isinstance(grid, dict)],
                "fitErrors": [grid.get("fitError") for grid in grids[:8] if isinstance(grid, dict)],
                "qualities": [grid.get("quality") for grid in grids[:8] if isinstance(grid, dict)],
            }
            for face, grids in groups.items()
            if isinstance(grids, list)
        }
    return by_image


def count_deviation_summary(count_rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not count_rows:
        return None
    first = count_rows[0]
    counts = first.get("counts") or {}
    if not isinstance(counts, dict):
        return None
    deviations = {
        face: int(counts.get(face, 0)) - 9
        for face in FACE_ORDER
        if int(counts.get(face, 0)) != 9
    }
    return {
        "mostCommonCounts": counts,
        "mostCommonCountFrequency": first.get("n"),
        "deviationsFromNine": deviations,
    }


def localization_hypotheses(
    *,
    payload: Dict[str, Any],
    health: Dict[str, Any],
    diagnostics: Optional[Dict[str, Any]],
) -> List[str]:
    hypotheses = []
    failed = set(payload.get("failedChecks") or [])
    weak = health.get("weakSelectedGrids") or []
    if any(item.get("image") == "imageB" and item.get("face") == "D" for item in weak):
        hypotheses.append("weak_imageB_down_anchor")
    if any(str(check).endswith("_count_not_9") for check in failed):
        hypotheses.append("pre_repair_face_count_imbalance")
    groups = grid_group_breakdown(diagnostics)
    for image_key, image_groups in groups.items():
        if not isinstance(image_groups, dict):
            continue
        for face in FACE_ORDER:
            face_count = (image_groups.get(face) or {}).get("count", 0)
            if face_count >= 6:
                hypotheses.append(f"{image_key}_{face}_grid_overgeneration")
    merged = (diagnostics or {}).get("mergedCandidates") if isinstance(diagnostics, dict) else None
    if isinstance(merged, dict) and not merged.get("legalInSample"):
        hypotheses.append("no_legal_state_in_sampled_merged_candidates")
    return hypotheses


def rejection_localization_probe(result: Any, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if payload.get("status") == "success":
        return None
    signals = payload.get("recognitionSignals") or {}
    health = selected_grid_health(signals)
    diagnostics = None
    if result.image_a is not None and result.image_b is not None:
        diagnostics = recognition_diagnostics(result.image_a, result.image_b)
    merged = (diagnostics or {}).get("mergedCandidates") if isinstance(diagnostics, dict) else {}
    return {
        "selectedGridHealth": health,
        "gridGroupBreakdown": grid_group_breakdown(diagnostics),
        "mergedCandidateSummary": {
            "optionsA": merged.get("optionsA") if isinstance(merged, dict) else None,
            "optionsB": merged.get("optionsB") if isinstance(merged, dict) else None,
            "merged": merged.get("merged") if isinstance(merged, dict) else None,
            "sampled": merged.get("sampled") if isinstance(merged, dict) else None,
            "legalInSample": merged.get("legalInSample") if isinstance(merged, dict) else None,
            "validationErrors": (merged.get("validationErrors") or [])[:8] if isinstance(merged, dict) else [],
            "faceCountDeviation": count_deviation_summary(merged.get("faceCounts") or []) if isinstance(merged, dict) else None,
            "examples": (merged.get("examples") or [])[:3] if isinstance(merged, dict) else [],
        },
        "hypotheses": localization_hypotheses(payload=payload, health=health, diagnostics=diagnostics),
    }


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


def _check_expected_yaw(
    expected: Optional[Dict[str, Any]],
    signals: Dict[str, Any],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Validate the recognizer's `recognitionSignals.captureYaw` against
    the manifest's `expectedYaw` block.

    Returns (passed, diagnostic). `passed` is True when:
      - the manifest row has no `expectedYaw` (legacy entries: pass-through), OR
      - every specified key in `expectedYaw` matches the corresponding
        field of `recognitionSignals.captureYaw`. Keys checked:
          status, quarterTurns, normalizationApplied

    `diagnostic` describes the comparison when `expectedYaw` is present,
    so the probe table / JSON output can surface "expected status=
    nonstandard, got standard" without the reader chasing the raw
    signals block. Returns None when there's no expectation to check.

    Three reasonable use cases for `expectedYaw`:
      - Pin "Set 12 must remain a nonstandard/3 yaw" so a future
        recognizer regression that silently re-canonicalizes the cube
        (yaw -> 0) gets caught by the probe harness.
      - Pin "Set 15 must remain standard/0" so an `_oriented_face_options`
        change that picks a different visible-face triple producing a
        non-zero yaw inference is also caught.
      - Document the expected yaw in the manifest so a reviewer can see
        at a glance whether each corpus pair exercises canonical-yaw or
        non-canonical-yaw code paths.
    """
    if not isinstance(expected, dict):
        return True, None
    capture_yaw = signals.get("captureYaw") if isinstance(signals, dict) else None
    if not isinstance(capture_yaw, dict):
        capture_yaw = {}
    actual = {
        "status": capture_yaw.get("status"),
        "quarterTurns": capture_yaw.get("quarterTurns"),
        "normalizationApplied": capture_yaw.get("normalizationApplied"),
    }
    mismatches: List[str] = []
    for key in ("status", "quarterTurns", "normalizationApplied"):
        if key not in expected:
            continue
        if actual.get(key) != expected[key]:
            mismatches.append(key)
    return (not mismatches), {
        "expected": {key: expected[key] for key in ("status", "quarterTurns", "normalizationApplied") if key in expected},
        "actual": actual,
        "mismatchedKeys": mismatches,
    }


def decoded_image_fingerprint(path: Path) -> Dict[str, Any]:
    from PIL import Image, ImageOps  # type: ignore

    try:
        with Image.open(path) as image:
            decoded = ImageOps.exif_transpose(image).convert("RGB")
            return {
                "format": image.format,
                "width": decoded.width,
                "height": decoded.height,
                "mode": decoded.mode,
                "rgbSha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
            }
    except Exception as exc:  # pragma: no cover - diagnostics must not fail probes
        return {"error": f"{type(exc).__name__}: {exc}"}


def _round_float(value: Any, digits: int = 4) -> Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _round_point(point: Sequence[Any]) -> List[Any]:
    return [_round_float(value) for value in point]


def _sticker_dump(sticker: Any) -> Dict[str, Any]:
    match = getattr(sticker, "match", None)
    return {
        "id": getattr(sticker, "id", None),
        "source": getattr(sticker, "source", None),
        "center": _round_point(getattr(sticker, "center", ()) or ()),
        "bbox": [_round_float(value) for value in (getattr(sticker, "bbox", ()) or ())],
        "rgb": list(getattr(sticker, "rgb", ()) or ()),
        "face": getattr(match, "face", None),
        "color": getattr(match, "color", None),
        "distance": _round_float(getattr(match, "distance", None)),
        "confidence": _round_float(getattr(match, "confidence", None)),
        "alternatives": [
            {"color": color, "distance": _round_float(distance)}
            for color, distance in (getattr(match, "alternatives", []) or [])[:6]
        ],
        "area": getattr(sticker, "area", None),
        "shapeAngle": _round_float(getattr(sticker, "shape_angle", None)),
    }


def _grid_dump(grid: Any) -> Dict[str, Any]:
    center_sticker = getattr(grid, "center_sticker", None)
    center_match = getattr(center_sticker, "match", None)
    return {
        "id": getattr(grid, "id", None),
        "centerFace": getattr(grid, "center_face", None),
        "centerColor": getattr(center_match, "color", None),
        "centerRgb": list(getattr(center_sticker, "rgb", ()) or ()),
        "matchedCount": getattr(grid, "matched_count", None),
        "fitError": _round_float(getattr(grid, "fit_error", None)),
        "quality": _round_float(_grid_quality_score(grid)),
        "points": [[_round_point(point) for point in row] for row in getattr(grid, "points", [])],
        "stickers": [[_sticker_dump(sticker) for sticker in row] for row in getattr(grid, "stickers", [])],
    }


def analysis_dump_for_image(path: Path, label: str) -> Dict[str, Any]:
    start = time.perf_counter()
    image_bytes = path.read_bytes()
    analysis = analyze_image(image_bytes)
    return {
        "label": label,
        "path": str(path),
        "byteSha256": hashlib.sha256(image_bytes).hexdigest(),
        "decodedImage": decoded_image_fingerprint(path),
        "timings": {"analyzeSeconds": _elapsed(start)},
        "analysis": {
            "width": analysis.width,
            "height": analysis.height,
            "roi": list(analysis.roi),
            "warnings": list(analysis.warnings),
            "stickerCount": len(analysis.stickers),
            "stickers": [_sticker_dump(sticker) for sticker in analysis.stickers],
            "gridCount": len(analysis.grids),
            "grids": [_grid_dump(grid) for grid in analysis.grids],
        },
    }


def analysis_dump_for_rows(
    rows: Sequence[Dict[str, Any]],
    manifest_path: Path,
    *,
    image_selection: str,
    fingerprint: Dict[str, Any],
) -> Dict[str, Any]:
    images = []
    for row in rows:
        set_id = str(row.get("setId") or row.get("id") or "")
        selected_images = ("imageA", "imageB") if image_selection == "both" else (image_selection,)
        for image_key in selected_images:
            path_key = f"{image_key}Path"
            if path_key not in row:
                continue
            image_path = normalize_path(str(row[path_key]), manifest_path)
            images.append(
                {
                    "setId": set_id,
                    **analysis_dump_for_image(image_path, image_key),
                }
            )
    return {
        "schemaVersion": 1,
        "manifest": str(manifest_path),
        "runtimeFingerprint": fingerprint,
        "images": images,
    }


def write_analysis_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_hash(path: Path, expected: Optional[str], *, decode_image: bool = False) -> Dict[str, Any]:
    actual = file_sha256(str(path))
    result = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "actual": actual,
        "expected": expected,
        "matches": expected in (None, "", actual),
    }
    if decode_image:
        result["decodedImage"] = decoded_image_fingerprint(path)
    return result


def missing_paths(paths: Sequence[Path]) -> List[str]:
    return [str(path) for path in paths if not path.exists()]


def probe_pair(row: Dict[str, Any], manifest_path: Path) -> Dict[str, Any]:
    total_start = time.perf_counter()
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
            "timings": {"totalSeconds": _elapsed(total_start)},
        }

    hash_start = time.perf_counter()
    image_hashes = {
        "imageA": verify_hash(image_a, row.get("imageA_sha256_expected"), decode_image=True),
        "imageB": verify_hash(image_b, row.get("imageB_sha256_expected"), decode_image=True),
        "groundTruth": verify_hash(truth_path, row.get("groundTruth_sha256_expected")),
    }
    hash_seconds = _elapsed(hash_start)
    input_drift = not all(item["matches"] for item in image_hashes.values())

    ground_truth_start = time.perf_counter()
    ground_truth_sha, raw_state, canonical_state, canonicalized = parse_ground_truth(str(truth_path))
    ground_truth_seconds = _elapsed(ground_truth_start)

    recognition_start = time.perf_counter()
    recognizer = WhiteUpRecognizer()
    result = recognizer.recognize(image_a.read_bytes(), image_b.read_bytes())
    recognition_seconds = _elapsed(recognition_start)

    diagnostics_start = time.perf_counter()
    payload = result.to_api_dict(include_overlays=False)
    recognized_state = payload.get("state") or ""
    score = score_match(recognized_state, canonical_state)
    score_vs_raw = score_match(recognized_state, raw_state)
    category = payload.get("recognitionCategory")
    signals = payload.get("recognitionSignals") or {}
    selected = signals.get("selectedRepairCandidate") or {}
    direct_legal = signals.get("directLegalCandidates") or {}

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
    rejection_probe = rejection_localization_probe(result, payload)
    diagnostics_seconds = _elapsed(diagnostics_start)

    expected_category = row.get("expectedCategory")
    expected_score_floor = row.get("expectedScoreFloor")
    expected_yaw = row.get("expectedYaw")
    category_ok = expected_category in (None, "", category)
    score_ok = expected_score_floor is None or score >= int(expected_score_floor)
    yaw_ok, yaw_diagnostic = _check_expected_yaw(expected_yaw, signals)
    contract_passed = (not input_drift) and category_ok and score_ok and yaw_ok

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
        "directLegalCandidateStatus": direct_legal.get("status"),
        "directLegalStateCount": direct_legal.get("stateCount"),
        "directLegalTopConfidence": direct_legal.get("topConfidence"),
        "directLegalSecondConfidence": direct_legal.get("secondConfidence"),
        "directLegalConfidenceGap": direct_legal.get("confidenceGap"),
        "directLegalTopTieCount": direct_legal.get("topTieCount"),
        "directLegalTopRawMergedScore": direct_legal.get("topRawMergedScore"),
        "directLegalSecondRawMergedScore": direct_legal.get("secondRawMergedScore"),
        "directLegalRawMergedScoreGap": direct_legal.get("rawMergedScoreGap"),
        "directLegalTopVariantCost": direct_legal.get("topVariantCost"),
        "directLegalSecondVariantCost": direct_legal.get("secondVariantCost"),
        "directLegalVariantCostGap": direct_legal.get("variantCostGap"),
        "topDirectLegalCandidates": score_direct_legal_candidates(direct_legal.get("topCandidates"), canonical_state),
        "selectedGridSpanSummary": selected_grid_span_summary(signals),
        "selectedGridQuality": signals.get("selectedGridQuality"),
        "topVisibleTripleQuality": signals.get("topVisibleTripleQuality"),
        "imageHashes": image_hashes,
        "groundTruth_sha256": ground_truth_sha,
        "inputDrift": input_drift,
        "expectedCategory": expected_category,
        "expectedScoreFloor": expected_score_floor,
        "expectedYaw": expected_yaw,
        "yawDiagnostic": yaw_diagnostic,
        "currentScoreObserved": row.get("currentScoreObserved"),
        "contractPassed": contract_passed,
        "contractFailures": [
            name
            for name, failed in (
                ("image_input_drift", input_drift),
                ("category_mismatch", not category_ok),
                ("score_below_floor", not score_ok),
                ("yaw_mismatch", not yaw_ok),
            )
            if failed
        ],
        "primaryFailureMode": primary_failure_mode,
        "failureModes": dict(Counter(face.get("failureMode", "unknown") for face in face_summaries)),
        "orientationDiagnostics": orientation,
        "rejectionLocalization": rejection_probe,
        "timings": {
            "hashSeconds": hash_seconds,
            "groundTruthSeconds": ground_truth_seconds,
            "recognizeSeconds": recognition_seconds,
            "diagnosticsSeconds": diagnostics_seconds,
            "totalSeconds": _elapsed(total_start),
        },
        "notes": row.get("notes"),
    }


def _format_progress_seconds(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes}m{remainder:02d}s"


def _probe_progress(message: str) -> None:
    print(f"[probe] {message}", file=sys.stderr, flush=True)


def probe_rows(
    rows: Sequence[Dict[str, Any]],
    manifest_path: Path,
    *,
    progress: bool = False,
) -> List[Dict[str, Any]]:
    results = []
    run_start = time.perf_counter()
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        set_id = str(row.get("setId") or row.get("id") or "")
        if progress:
            elapsed = time.perf_counter() - run_start
            _probe_progress(f"{index}/{total} set {set_id} start elapsed={_format_progress_seconds(elapsed)}")

        result = probe_pair(row, manifest_path)
        results.append(result)

        if progress:
            elapsed = time.perf_counter() - run_start
            average = elapsed / index if index else 0.0
            eta = average * (total - index)
            timings = result.get("timings") or {}
            row_seconds = float(timings.get("totalSeconds") or 0.0)
            if result.get("status") == "skipped":
                score = "skip"
            else:
                score = f"{result.get('score')}/54"
            contract = "pass" if result.get("contractPassed") else "FAIL"
            _probe_progress(
                f"{index}/{total} set {set_id} done "
                f"row={_format_progress_seconds(row_seconds)} "
                f"elapsed={_format_progress_seconds(elapsed)} "
                f"eta={_format_progress_seconds(eta)} "
                f"score={score} "
                f"category={result.get('category')} "
                f"contract={contract}"
            )
    return results


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
        ("Sec", 10),
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
            result.get("timings", {}).get("totalSeconds"),
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


def timing_summary(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    row_times = [
        float((result.get("timings") or {}).get("totalSeconds", 0.0))
        for result in results
        if result.get("status") != "skipped"
    ]
    sorted_times = sorted(row_times)
    total_seconds = round(sum(row_times), 4)
    row_count = len(row_times)
    return {
        "totalSeconds": total_seconds,
        "rowCount": row_count,
        "meanSeconds": round(total_seconds / row_count, 4) if row_count else 0.0,
        "p50Seconds": _median(sorted_times),
        "p95Seconds": _nearest_rank_percentile(sorted_times, 0.95),
        "slowestRows": [
            {
                "setId": result.get("setId"),
                "totalSeconds": (result.get("timings") or {}).get("totalSeconds"),
                "recognizeSeconds": (result.get("timings") or {}).get("recognizeSeconds"),
            }
            for result in sorted(
                (item for item in results if item.get("status") != "skipped"),
                key=lambda item: float((item.get("timings") or {}).get("totalSeconds", 0.0)),
                reverse=True,
            )[:5]
        ],
    }


def _median(sorted_values: Sequence[float]) -> float:
    if not sorted_values:
        return 0.0
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return round(sorted_values[midpoint], 4)
    return round((sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0, 4)


def _nearest_rank_percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, math.ceil(percentile * len(sorted_values)) - 1))
    return round(sorted_values[index], 4)


def write_json(
    path: Path,
    results: Sequence[Dict[str, Any]],
    manifest: Path,
    fingerprint: Dict[str, Any],
    environment_warnings: Sequence[Dict[str, Any]],
) -> None:
    payload = {
        "schemaVersion": 1,
        "manifest": str(manifest),
        "runtimeFingerprint": fingerprint,
        "environmentPolicyWarnings": list(environment_warnings),
        "timingSummary": timing_summary(results),
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
    parser.add_argument("--analysis-output", help="Optional path for analysis-grid dump JSON.")
    parser.add_argument(
        "--analysis-image",
        choices=("imageA", "imageB", "both"),
        default="both",
        help="Which manifest image(s) to include in --analysis-output.",
    )
    parser.add_argument("--analysis-only", action="store_true", help="Write analysis dump without running recognition.")
    parser.add_argument("--quiet", action="store_true", help="Do not print the readable table.")
    parser.add_argument("--no-progress", action="store_true", help="Do not print row-level probe progress to stderr.")
    parser.add_argument("--fail-on-contract", action="store_true", help="Exit non-zero if any non-skipped row fails its contract.")
    args = parser.parse_args()

    manifest = Path(args.manifest).expanduser()
    if not manifest.is_absolute():
        manifest = (Path.cwd() / manifest).resolve()
    selected = {str(item) for item in args.set_id or []}
    manifest_document = load_manifest_document(manifest)
    rows = [
        row
        for row in (dict(item) for item in manifest_document["pairs"])
        if not selected or str(row.get("setId") or row.get("id")) in selected
    ]
    fingerprint = runtime_fingerprint()
    environment_warnings = environment_policy_warnings(manifest_document, fingerprint)
    print_environment_warnings(environment_warnings)
    if args.analysis_output:
        write_analysis_json(
            Path(args.analysis_output).expanduser(),
            analysis_dump_for_rows(rows, manifest, image_selection=args.analysis_image, fingerprint=fingerprint),
        )
    if args.analysis_only:
        # Analysis-only intentionally stops before recognition, so it also
        # skips any requested --json-output probe result.
        if not args.quiet:
            print(f"Runtime: {runtime_summary(fingerprint)}")
            print(f"Analysis dump rows={len(rows)} image={args.analysis_image}")
        return 0

    results = probe_rows(rows, manifest, progress=not args.no_progress)

    if args.json_output:
        write_json(Path(args.json_output).expanduser(), results, manifest, fingerprint, environment_warnings)
    if args.jsonl_output:
        write_jsonl(Path(args.jsonl_output).expanduser(), results)
    if not args.quiet:
        print(f"Runtime: {runtime_summary(fingerprint)}")
        summary = timing_summary(results)
        print(
            f"Timing: total={summary['totalSeconds']}s rows={summary['rowCount']} "
            f"mean={summary['meanSeconds']}s p50={summary['p50Seconds']}s p95={summary['p95Seconds']}s"
        )
        print()
        print_table(results)

    if args.fail_on_contract and any(not result.get("contractPassed") and result.get("status") != "skipped" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
