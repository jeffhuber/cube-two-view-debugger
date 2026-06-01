"""Replay imported CubeSnap iOS repro bundles against the local recognizer.

Use this after ``tools/ios_repro_bundle.py`` has unpacked one or more
``cubesnap-repro`` JSON bundles into a durable local corpus. The evaluator
does not commit or copy images; it reads local manifests, runs the exact
uploaded crops through constrained recognition, and writes a compact JSON/MD
report that makes rejection-quality regressions easy to spot.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.dataset import evaluate_state  # noqa: E402
from tools.ios_repro_bundle import DEFAULT_CORPUS_ROOT  # noqa: E402


DEFAULT_REPORT_JSON = Path("runs") / "ios_repro_bundle_eval.json"
DEFAULT_REPORT_MD = Path("runs") / "ios_repro_bundle_eval.md"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string(value: Any) -> Optional[str]:
    return value if isinstance(value, str) else None


def find_manifests(paths: Sequence[Path]) -> list[Path]:
    manifests: list[Path] = []
    for path in paths:
        if path.is_file() and path.name == "manifest.json":
            manifests.append(path)
        elif path.is_dir() and (path / "manifest.json").exists():
            manifests.append(path / "manifest.json")
        elif path.is_dir():
            manifests.extend(path.glob("**/manifest.json"))
    return sorted(dict.fromkeys(manifests))


def load_manifest(path: Path) -> Mapping[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    if manifest.get("manifestSchema") != "ctvd.iosReproBundleManifest.v1":
        raise ValueError(f"{path} is not an iOS repro manifest")
    return manifest


def image_paths(manifest_path: Path, manifest: Mapping[str, Any], edge_px: Optional[int]) -> tuple[Path, Path]:
    root = manifest_path.parent
    images = manifest.get("images")
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        raise ValueError(f"{manifest_path} manifest.images must be an array")

    sizes = [edge_px] if edge_px is not None else sorted(
        {
            image.get("edgePx")
            for image in images
            if isinstance(image, Mapping) and isinstance(image.get("edgePx"), int)
        },
        reverse=True,
    )
    for size in sizes:
        image_a = next(
            (
                image
                for image in images
                if isinstance(image, Mapping)
                and image.get("role") == "imageA"
                and image.get("edgePx") == size
            ),
            None,
        )
        image_b = next(
            (
                image
                for image in images
                if isinstance(image, Mapping)
                and image.get("role") == "imageB"
                and image.get("edgePx") == size
            ),
            None,
        )
        if isinstance(image_a, Mapping) and isinstance(image_b, Mapping):
            path_a = root / str(image_a.get("path") or "")
            path_b = root / str(image_b.get("path") or "")
            if not path_a.exists() or not path_b.exists():
                raise FileNotFoundError(f"{manifest_path} references missing decoded image(s)")
            return path_a, path_b
    requested = f" at {edge_px}px" if edge_px is not None else ""
    raise ValueError(f"{manifest_path} does not contain an imageA/imageB pair{requested}")


def expected_state_from_manifest(manifest: Mapping[str, Any]) -> Optional[str]:
    recognized = _string(manifest.get("recognizedState"))
    if recognized and len(recognized) == 54:
        return recognized
    expected = _string(manifest.get("expectedState"))
    if expected and len(expected) == 54:
        return expected
    return None


def _constrained_signal(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    signals = _mapping(payload.get("recognitionSignals"))
    return _mapping(signals.get("constrainedInference"))


def summarize_payload(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
    elapsed_ms: float,
    expected_state: Optional[str],
) -> dict[str, Any]:
    signal = _constrained_signal(payload)
    fast_reject = _mapping(signal.get("fastReject"))
    promotion_gate = _mapping(signal.get("promotionGate"))
    evaluation = evaluate_state(_string(payload.get("state")), expected_state)
    return {
        "manifest": str(manifest_path),
        "generatedAt": manifest.get("generatedAt"),
        "originalStatus": manifest.get("status"),
        "originalRecognizedState": manifest.get("recognizedState"),
        "replayStatus": payload.get("status"),
        "replayRecognitionCategory": payload.get("recognitionCategory"),
        "replayRecognitionCategoryReason": payload.get("recognitionCategoryReason"),
        "replayReason": payload.get("reason"),
        "failedChecks": payload.get("failedChecks") if isinstance(payload.get("failedChecks"), list) else [],
        "recommendedMethod": signal.get("recommendedMethod"),
        "promotionAccepted": promotion_gate.get("accepted"),
        "fastRejectSource": fast_reject.get("source"),
        "fastRejectReason": fast_reject.get("reason"),
        "fastRejectQualityIssue": fast_reject.get("qualityIssue"),
        "elapsedMs": round(elapsed_ms, 2),
        "expectedAvailable": evaluation.get("available"),
        "exact": evaluation.get("exact"),
        "hamming": evaluation.get("hamming"),
    }


def replay_manifest(manifest_path: Path, *, mode: str, edge_px: Optional[int]) -> dict[str, Any]:
    from app import _recognize_with_constrained_inference_mode
    from rubik_recognizer.recognizer import WhiteUpRecognizer

    manifest = load_manifest(manifest_path)
    path_a, path_b = image_paths(manifest_path, manifest, edge_px)
    expected_state = expected_state_from_manifest(manifest)
    started = time.perf_counter()
    result = _recognize_with_constrained_inference_mode(
        WhiteUpRecognizer(),
        path_a.read_bytes(),
        path_b.read_bytes(),
        mode,
        expected_state=expected_state,
    )
    payload = result.to_api_dict()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return summarize_payload(
        manifest_path=manifest_path,
        manifest=manifest,
        payload=payload,
        elapsed_ms=elapsed_ms,
        expected_state=expected_state,
    )


def build_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    check_counts: dict[str, int] = {}
    exact_available = 0
    exact_count = 0
    for row in rows:
        status = str(row.get("replayStatus") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if row.get("expectedAvailable") is True:
            exact_available += 1
            if row.get("exact") is True:
                exact_count += 1
        for check in row.get("failedChecks") or []:
            check_counts[str(check)] = check_counts.get(str(check), 0) + 1
    return {
        "schema": "ctvd.iosReproBundleEval.v1",
        "count": len(rows),
        "statusCounts": dict(sorted(status_counts.items())),
        "failedCheckCounts": dict(sorted(check_counts.items())),
        "expectedStateExact": {
            "available": exact_available,
            "exact": exact_count,
        },
        "rows": list(rows),
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    rows = summary.get("rows") if isinstance(summary.get("rows"), list) else []
    lines = [
        "# iOS Repro Bundle Evaluation",
        "",
        f"- Bundles: `{summary.get('count')}`",
        f"- Replay status counts: `{json.dumps(summary.get('statusCounts'), sort_keys=True)}`",
        f"- Failed check counts: `{json.dumps(summary.get('failedCheckCounts'), sort_keys=True)}`",
    ]
    exact = _mapping(summary.get("expectedStateExact"))
    if exact.get("available"):
        lines.append(f"- Expected-state exact: `{exact.get('exact')}/{exact.get('available')}`")
    lines.extend(
        [
            "",
            "| Manifest | Original | Replay | Method | Checks | Exact | Hamming |",
            "|---|---:|---:|---|---|---:|---:|",
        ]
    )
    for row in rows:
        manifest = Path(str(row.get("manifest") or "")).parent.name
        checks = ",".join(str(check) for check in row.get("failedChecks") or [])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{manifest}`",
                    f"`{row.get('originalStatus')}`",
                    f"`{row.get('replayStatus')}`",
                    f"`{row.get('recommendedMethod') or row.get('fastRejectReason') or ''}`",
                    f"`{checks}`",
                    f"`{row.get('exact')}`",
                    f"`{row.get('hamming')}`",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_CORPUS_ROOT],
        help="manifest.json files, bundle directories, or roots containing imported bundles",
    )
    parser.add_argument("--mode", choices=("prefer", "shadow"), default="prefer")
    parser.add_argument("--edge-px", type=int, help="specific decoded crop size to replay")
    parser.add_argument("--limit", type=int, help="maximum manifests to replay")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    manifests = find_manifests(args.paths)
    if args.limit is not None:
        manifests = manifests[: max(0, args.limit)]
    if not manifests:
        print("evaluate_ios_repro_bundles: no manifests found", file=sys.stderr)
        return 1

    rows = [replay_manifest(path, mode=args.mode, edge_px=args.edge_px) for path in manifests]
    summary = build_summary(rows)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(render_markdown(summary), encoding="utf-8")
    print(f"evaluated {len(rows)} iOS repro bundle(s)")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
