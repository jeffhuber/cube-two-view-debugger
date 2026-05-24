#!/usr/bin/env python3
"""Build a diagnostic color-evidence report from oracle-rectified faces.

This tool consumes the oracle output from `tools/build_oracle_rectified_faces.py`
and joins every sampled WCA facelet position (`U1`, `R7`, etc.) to the
canonical ground-truth cube state in `tests/fixtures/corpus_manifest.json`.

Important: expected colors come from the scrambled ground-truth state, not from
the solved-face name. `U1` does not mean "white"; it means "the sticker currently
at U1", whose expected color is `state[U1]`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import (  # noqa: E402
    CLASSIFIER_CANONICAL,
    CLASSIFIER_KNN5_LAB,
    CLASSIFIER_KNN5_LAB_FULL,
    COLOR_ORDER,
    FACE_TO_COLOR,
    classify_rgb_with_mode,
)
from rubik_recognizer.validation import FACE_ORDER, validate_state  # noqa: E402
from tools.build_oracle_rectified_faces import (  # noqa: E402
    DEFAULT_FACE_SIZE,
    DEFAULT_MANIFEST,
    DEFAULT_OUT as DEFAULT_ORACLE_ROOT,
    DEFAULT_PATCH_FRACTION,
    DEFAULT_TRUTH,
    build_all,
)


DEFAULT_JSON_OUTPUT = REPO_ROOT / "runs" / "oracle_color_evidence_report.json"
DEFAULT_MD_OUTPUT = REPO_ROOT / "runs" / "oracle_color_evidence_report.md"
MODE_ORDER = (
    CLASSIFIER_CANONICAL,
    CLASSIFIER_KNN5_LAB,
    CLASSIFIER_KNN5_LAB_FULL,
)


@dataclass(frozen=True)
class GroundTruthEntry:
    path: Path
    sha256_expected: Optional[str] = None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_mismatch_reason(path: Path, expected: Optional[str]) -> Optional[str]:
    if not expected:
        return None
    got = _sha256_file(path)
    if got.lower() == expected.lower():
        return None
    return (
        f"ground truth hash mismatch for {path}: expected "
        f"{expected.lower()}, got {got}"
    )


def load_manifest_ground_truths(path: Path) -> Dict[str, GroundTruthEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, GroundTruthEntry] = {}
    for pair in raw.get("pairs", []):
        set_id = pair.get("setId")
        gt_path = pair.get("groundTruthPath")
        if not set_id or not gt_path:
            continue
        out[str(set_id)] = GroundTruthEntry(
            path=Path(gt_path),
            sha256_expected=pair.get("groundTruth_sha256_expected"),
        )
    return out


def _iter_payload_dicts(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            if isinstance(value, (dict, list)):
                yield from _iter_payload_dicts(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_payload_dicts(item)


def load_ground_truth_state(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in _iter_payload_dicts(payload):
        for key in (
            "corrected",
            "correctedState",
            "corrected_state",
            "expected",
            "expectedState",
            "expected_state",
            "state",
            "urfdlb",
        ):
            value = item.get(key)
            if isinstance(value, str):
                state = "".join(value.split()).upper()
                if len(state) == 54 and not (set(state) - set(FACE_ORDER)):
                    validation = validate_state(state)
                    if not validation.valid:
                        raise ValueError(
                            f"ground truth state at {path} is not a valid "
                            f"WCA URFDLB cube: {validation.errors}"
                        )
                    return state
    raise ValueError(f"no 54-character ground truth state found at {path}")


def expected_color_for_facelet(state: str, facelet_id: str) -> Tuple[str, str]:
    if len(facelet_id) < 2 or facelet_id[0] not in FACE_ORDER:
        raise ValueError(f"invalid facelet_id: {facelet_id!r}")
    try:
        sticker_id = int(facelet_id[1:])
    except ValueError as exc:
        raise ValueError(f"invalid facelet_id: {facelet_id!r}") from exc
    if not 1 <= sticker_id <= 9:
        raise ValueError(f"invalid facelet_id: {facelet_id!r}")
    index = FACE_ORDER.index(facelet_id[0]) * 9 + sticker_id - 1
    expected_face = state[index]
    return expected_face, FACE_TO_COLOR[expected_face]


def _prediction(rgb: Sequence[int], mode: str) -> Dict[str, Any]:
    match = classify_rgb_with_mode(tuple(int(v) for v in rgb), mode)
    second = match.alternatives[1][1] if len(match.alternatives) > 1 else match.distance
    return {
        "color": match.color,
        "face": match.face,
        "confidence": round(float(match.confidence), 4),
        "distance": round(float(match.distance), 4),
        "margin": round(float(second - match.distance), 4),
        "alternatives": [
            {"color": color, "distance": round(float(distance), 4)}
            for color, distance in match.alternatives[:3]
        ],
    }


def _load_or_build_oracle_index(
    *,
    oracle_root: Path,
    truth_path: Path,
    manifest_path: Path,
    face_size: int,
    patch_fraction: float,
    rows_glob: str,
    build_oracle: bool,
) -> Dict[str, Any]:
    index_path = oracle_root / "index.json"
    if build_oracle or not index_path.exists():
        return build_all(
            truth_path=truth_path,
            manifest_path=manifest_path,
            out_root=oracle_root,
            face_size=face_size,
            patch_fraction=patch_fraction,
            yaw_overrides={},
            save_patches=True,
            rows_glob=rows_glob,
        )
    return json.loads(index_path.read_text(encoding="utf-8"))


def flatten_oracle_observations(
    oracle_index: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    truth_entries = load_manifest_ground_truths(manifest_path)
    state_cache: Dict[str, str] = {}
    state_errors: Dict[str, str] = {}
    observations: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = [
        {"key": str(row.get("key", "")), "reason": str(row.get("reason", ""))}
        for row in oracle_index.get("skipped", [])
    ]

    def state_for_set(set_id: str) -> Optional[str]:
        if set_id in state_cache:
            return state_cache[set_id]
        if set_id in state_errors:
            return None
        entry = truth_entries.get(set_id)
        if entry is None:
            state_errors[set_id] = f"no groundTruthPath in manifest for set {set_id}"
            return None
        if not entry.path.exists():
            state_errors[set_id] = f"ground truth not found: {entry.path}"
            return None
        mismatch = _hash_mismatch_reason(entry.path, entry.sha256_expected)
        if mismatch:
            state_errors[set_id] = mismatch
            return None
        try:
            state_cache[set_id] = load_ground_truth_state(entry.path)
        except Exception as exc:  # noqa: BLE001
            state_errors[set_id] = f"error loading {entry.path}: {exc!r}"
            return None
        return state_cache[set_id]

    for row in oracle_index.get("rows", []):
        set_id = str(row["set_id"])
        state = state_for_set(set_id)
        if state is None:
            skipped.append({"key": str(row["key"]), "reason": state_errors[set_id]})
            continue
        for face in row.get("faces", []):
            for sticker in face.get("stickers", []):
                expected_face, expected_color = expected_color_for_facelet(
                    state, sticker["facelet_id"]
                )
                rgb = [int(v) for v in sticker["rgb"]]
                predictions = {
                    mode: _prediction(rgb, mode)
                    for mode in MODE_ORDER
                }
                for mode in MODE_ORDER:
                    predictions[mode]["correct"] = (
                        predictions[mode]["color"] == expected_color
                    )
                observations.append({
                    "set_id": set_id,
                    "key": row["key"],
                    "side": row["side"],
                    "yaw_quarter_turns": row["yaw_quarter_turns"],
                    "slot": face["slot"],
                    "wca_face": face["wca_face"],
                    "facelet_id": sticker["facelet_id"],
                    "sticker_id": sticker["sticker_id"],
                    "row": sticker["row"],
                    "col": sticker["col"],
                    "expected_face": expected_face,
                    "expected_color": expected_color,
                    "rgb": rgb,
                    "hsv": sticker["hsv"],
                    "lab": sticker["lab"],
                    "sticker_png": sticker["sticker_png"],
                    "patch_png": sticker["patch_png"],
                    "predictions": predictions,
                })
    return observations, skipped


def _confusion_matrix(observations: Sequence[Dict[str, Any]], mode: str) -> List[List[int]]:
    index = {color: idx for idx, color in enumerate(COLOR_ORDER)}
    matrix = [[0 for _ in COLOR_ORDER] for _ in COLOR_ORDER]
    for obs in observations:
        truth = obs["expected_color"]
        pred = obs["predictions"][mode]["color"]
        matrix[index[truth]][index[pred]] += 1
    return matrix


def _accuracy(observations: Sequence[Dict[str, Any]], mode: str) -> float:
    if not observations:
        return 0.0
    correct = sum(1 for obs in observations if obs["predictions"][mode]["correct"])
    return correct / len(observations)


def _group_accuracy(
    observations: Sequence[Dict[str, Any]],
    mode: str,
    key: str,
) -> Dict[str, float]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        grouped[str(obs[key])].append(obs)
    return {
        group: round(_accuracy(rows, mode), 4)
        for group, rows in sorted(grouped.items())
    }


def _distribution_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        median = ordered[n // 2]
    else:
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    return {
        "min": round(float(ordered[0]), 4),
        "median": round(float(median), 4),
        "mean": round(float(sum(ordered) / n), 4),
        "max": round(float(ordered[-1]), 4),
    }


def _group_prediction_stats(
    observations: Sequence[Dict[str, Any]],
    mode: str,
    key: str,
    field: str,
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for obs in observations:
        grouped[str(obs[key])].append(float(obs["predictions"][mode][field]))
    return {
        group: _distribution_stats(values)
        for group, values in sorted(grouped.items())
    }


def _mode_summary(observations: Sequence[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    correct = sum(1 for obs in observations if obs["predictions"][mode]["correct"])
    return {
        "correct": correct,
        "total": len(observations),
        "accuracy": round(_accuracy(observations, mode), 4),
        "confusion_labels": list(COLOR_ORDER),
        "confusion_matrix": _confusion_matrix(observations, mode),
        "per_expected_color_accuracy": _group_accuracy(observations, mode, "expected_color"),
        "per_set_accuracy": _group_accuracy(observations, mode, "set_id"),
        "per_side_accuracy": _group_accuracy(observations, mode, "side"),
        "per_yaw_accuracy": _group_accuracy(observations, mode, "yaw_quarter_turns"),
        "confidence_stats": _distribution_stats([
            float(obs["predictions"][mode]["confidence"])
            for obs in observations
        ]),
        "margin_stats": _distribution_stats([
            float(obs["predictions"][mode]["margin"])
            for obs in observations
        ]),
        "confidence_stats_by_expected_color": _group_prediction_stats(
            observations, mode, "expected_color", "confidence"
        ),
        "margin_stats_by_expected_color": _group_prediction_stats(
            observations, mode, "expected_color", "margin"
        ),
    }


def _mismatches(
    observations: Sequence[Dict[str, Any]],
    mode: str,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    rows = []
    for obs in observations:
        pred = obs["predictions"][mode]
        if pred["correct"]:
            continue
        rows.append({
            "key": obs["key"],
            "facelet_id": obs["facelet_id"],
            "slot": obs["slot"],
            "wca_face": obs["wca_face"],
            "expected_color": obs["expected_color"],
            "predicted_color": pred["color"],
            "confidence": pred["confidence"],
            "margin": pred["margin"],
            "rgb": obs["rgb"],
            "sticker_png": obs["sticker_png"],
            "patch_png": obs["patch_png"],
        })
    rows.sort(
        key=lambda row: (
            -float(row["confidence"]),
            -float(row["margin"]),
            str(row["key"]),
            str(row["facelet_id"]),
        )
    )
    return rows[:limit]


def _weakest_observations(
    observations: Sequence[Dict[str, Any]],
    mode: str,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    rows = []
    for obs in observations:
        pred = obs["predictions"][mode]
        rows.append({
            "key": obs["key"],
            "facelet_id": obs["facelet_id"],
            "slot": obs["slot"],
            "wca_face": obs["wca_face"],
            "expected_color": obs["expected_color"],
            "predicted_color": pred["color"],
            "correct": pred["correct"],
            "confidence": pred["confidence"],
            "margin": pred["margin"],
            "rgb": obs["rgb"],
            "sticker_png": obs["sticker_png"],
            "patch_png": obs["patch_png"],
        })
    rows.sort(
        key=lambda row: (
            float(row["confidence"]),
            float(row["margin"]),
            str(row["key"]),
            str(row["facelet_id"]),
        )
    )
    return rows[:limit]


def analyze_oracle_color_evidence(
    oracle_index: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> Dict[str, Any]:
    observations, skipped = flatten_oracle_observations(
        oracle_index, manifest_path=manifest_path
    )
    modes = {mode: _mode_summary(observations, mode) for mode in MODE_ORDER}
    return {
        "schema": "oracle_color_evidence_report_v1",
        "source": {
            "oracle_schema": oracle_index.get("schema"),
            "oracle_source": oracle_index.get("source", {}),
            "manifest_path": str(manifest_path),
        },
        "summary": {
            "row_count": len({obs["key"] for obs in observations}),
            "set_count": len({obs["set_id"] for obs in observations}),
            "observation_count": len(observations),
            "expected_color_counts": dict(Counter(obs["expected_color"] for obs in observations)),
            "skipped_count": len(skipped),
        },
        "modes": modes,
        "mismatches": {
            mode: _mismatches(observations, mode)
            for mode in MODE_ORDER
        },
        "weakest_observations": {
            mode: _weakest_observations(observations, mode)
            for mode in MODE_ORDER
        },
        "observations": observations,
        "skipped": skipped,
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return out


def render_markdown_report(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    summary = report["summary"]
    lines.append("# Oracle color evidence report")
    lines.append("")
    lines.append(
        f"{summary['observation_count']} oracle sticker observations "
        f"across {summary['row_count']} rows / {summary['set_count']} sets."
    )
    if summary["skipped_count"]:
        lines.append(f"{summary['skipped_count']} row(s) skipped; see JSON for details.")
    lines.append("")
    lines.append("## Mode accuracy")
    mode_rows = []
    for mode in MODE_ORDER:
        mode_summary = report["modes"][mode]
        mode_rows.append([
            mode,
            f"{mode_summary['correct']}/{mode_summary['total']}",
            _pct(mode_summary["accuracy"]),
        ])
    lines.extend(_markdown_table(["Mode", "Correct", "Accuracy"], mode_rows))
    lines.append("")
    lines.append("## Accuracy by expected color")
    color_rows = []
    for color in COLOR_ORDER:
        color_rows.append([
            color,
            *[
                _pct(report["modes"][mode]["per_expected_color_accuracy"].get(color, 0.0))
                for mode in MODE_ORDER
            ],
        ])
    lines.extend(_markdown_table(["Expected", *MODE_ORDER], color_rows))
    lines.append("")
    lines.append("## Accuracy by set")
    set_ids = sorted({
        set_id
        for mode in MODE_ORDER
        for set_id in report["modes"][mode]["per_set_accuracy"].keys()
    }, key=lambda value: int(value) if value.isdigit() else value)
    set_rows = []
    for set_id in set_ids:
        set_rows.append([
            set_id,
            *[
                _pct(report["modes"][mode]["per_set_accuracy"].get(set_id, 0.0))
                for mode in MODE_ORDER
            ],
        ])
    lines.extend(_markdown_table(["Set", *MODE_ORDER], set_rows))
    lines.append("")
    lines.append("## Canonical confidence by expected color")
    conf_rows = []
    canonical = report["modes"][CLASSIFIER_CANONICAL]
    for color in COLOR_ORDER:
        conf = canonical["confidence_stats_by_expected_color"].get(color, {})
        margin = canonical["margin_stats_by_expected_color"].get(color, {})
        conf_rows.append([
            color,
            conf.get("min", 0.0),
            conf.get("median", 0.0),
            margin.get("min", 0.0),
            margin.get("median", 0.0),
        ])
    lines.extend(_markdown_table(
        ["Expected", "Min conf", "Median conf", "Min margin", "Median margin"],
        conf_rows,
    ))
    lines.append("")
    lines.append("## Canonical confusion matrix")
    matrix_rows = []
    for label, row in zip(canonical["confusion_labels"], canonical["confusion_matrix"]):
        matrix_rows.append([label, *row])
    lines.extend(_markdown_table(["Expected \\ Pred", *canonical["confusion_labels"]], matrix_rows))
    lines.append("")
    lines.append("## Lowest-confidence canonical observations")
    weak_rows = []
    for row in report["weakest_observations"][CLASSIFIER_CANONICAL][:20]:
        weak_rows.append([
            f"`{row['key']}`",
            row["facelet_id"],
            row["expected_color"],
            row["predicted_color"],
            "yes" if row["correct"] else "no",
            row["confidence"],
            row["margin"],
            row["rgb"],
        ])
    lines.extend(_markdown_table(
        ["Row", "Facelet", "Expected", "Pred", "Correct", "Conf", "Margin", "RGB"],
        weak_rows,
    ))
    lines.append("")
    lines.append("## Highest-confidence canonical mismatches")
    mismatch_rows = []
    for row in report["mismatches"][CLASSIFIER_CANONICAL][:20]:
        mismatch_rows.append([
            f"`{row['key']}`",
            row["facelet_id"],
            row["expected_color"],
            row["predicted_color"],
            row["confidence"],
            row["margin"],
            row["rgb"],
        ])
    if mismatch_rows:
        lines.extend(_markdown_table(
            ["Row", "Facelet", "Expected", "Pred", "Conf", "Margin", "RGB"],
            mismatch_rows,
        ))
    else:
        lines.append("No canonical mismatches.")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Expected colors are read from canonical ground truth at each WCA facelet position.")
    lines.append("- Geometry comes from human-reviewed full-corner labels and the oracle rectifier.")
    lines.append("- This report diagnoses color evidence only; it does not run the recognizer fit.")
    lines.append("- Canonical margins can be negative when the HSV hint overrides the nearest Lab prototype; treat that as an ambiguity cue, not automatically as a wrong classification.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--face-size", type=int, default=DEFAULT_FACE_SIZE)
    parser.add_argument("--sticker-patch", type=float, default=DEFAULT_PATCH_FRACTION)
    parser.add_argument("--rows-glob", type=str, default="*")
    parser.add_argument("--build-oracle", action="store_true")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    oracle_index = _load_or_build_oracle_index(
        oracle_root=args.oracle_root,
        truth_path=args.truth,
        manifest_path=args.manifest,
        face_size=args.face_size,
        patch_fraction=args.sticker_patch,
        rows_glob=args.rows_glob,
        build_oracle=args.build_oracle,
    )
    report = analyze_oracle_color_evidence(
        oracle_index,
        manifest_path=args.manifest,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.md_output.write_text(render_markdown_report(report), encoding="utf-8")

    print(
        "wrote oracle color evidence report: "
        f"{args.json_output} and {args.md_output}",
        file=sys.stderr,
    )
    for mode in MODE_ORDER:
        mode_summary = report["modes"][mode]
        print(
            f"{mode}: {mode_summary['correct']}/{mode_summary['total']} "
            f"({_pct(mode_summary['accuracy'])})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
