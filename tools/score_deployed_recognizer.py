#!/usr/bin/env python3
"""Score a deployed CTVD recognizer endpoint against the local manifest.

This is the production-path counterpart to the in-process constrained
recognizer diagnostics. It sends the manifest's A/B image files to a live
`/api/recognize` endpoint, scores returned states against local ground truth,
and writes JSON plus an optional Markdown summary.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib import error, parse, request

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_recognition_pair import parse_ground_truth, score_match  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_ENDPOINT = "https://api.cubesnap.app/api/recognize"


def _load_manifest(path: Path, only_sets: Optional[set[str]]) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError(f"{path} must contain a top-level pairs array")
    rows: List[Dict[str, Any]] = []
    for raw in pairs:
        if not isinstance(raw, Mapping):
            continue
        set_id = str(raw.get("setId") or "")
        if only_sets and set_id not in only_sets:
            continue
        image_a = Path(str(raw.get("imageAPath") or "")).expanduser()
        image_b = Path(str(raw.get("imageBPath") or "")).expanduser()
        gt = Path(str(raw.get("groundTruthPath") or "")).expanduser()
        if not image_a.exists() or not image_b.exists() or not gt.exists():
            rows.append({
                "setId": set_id,
                "status": "skipped_missing_local_file",
                "paths": {
                    "imageA": str(image_a),
                    "imageB": str(image_b),
                    "groundTruth": str(gt),
                },
            })
            continue
        rows.append({
            "setId": set_id,
            "imageA": image_a,
            "imageB": image_b,
            "groundTruth": gt,
        })
    return rows


def _recognize_url(endpoint: str, mode: str) -> str:
    url = parse.urlparse(endpoint)
    query = dict(parse.parse_qsl(url.query, keep_blank_values=True))
    query.setdefault("slim", "1")
    query.setdefault("hullLabelTier1", mode)
    return parse.urlunparse(url._replace(query=parse.urlencode(query)))


def _multipart_body(fields: Sequence[Tuple[str, str, bytes, str]]) -> Tuple[bytes, str]:
    boundary = f"----cubesnap-score-{uuid.uuid4().hex}"
    chunks: List[bytes] = []
    for name, filename, data, content_type in fields:
        chunks.append(f"--{boundary}\r\n".encode("ascii"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), boundary


def _post_pair(endpoint: str, image_a: Path, image_b: Path, timeout: float) -> Tuple[int, Dict[str, Any]]:
    body, boundary = _multipart_body([
        ("imageA", image_a.name, image_a.read_bytes(), "image/jpeg"),
        ("imageB", image_b.name, image_b.read_bytes(), "image/jpeg"),
    ])
    req = request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return int(resp.status), payload
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"status": "http_error", "reason": raw[:1000]}
        return int(exc.code), payload


def _hamming(left: str, right: str) -> Optional[int]:
    if len(left) != 54 or len(right) != 54:
        return None
    return 54 - score_match(left, right)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _constrained_performance(constrained: Mapping[str, Any]) -> Dict[str, Any]:
    performance = constrained.get("performance")
    if not isinstance(performance, Mapping):
        return {}
    timings = performance.get("stageTimingsMs")
    stage_timings = dict(timings) if isinstance(timings, Mapping) else {}
    return {
        "performanceSchema": performance.get("schema"),
        "rectifiedInputPerformanceSchema": performance.get("rectifiedInputPerformanceSchema"),
        "contactSheetsIncluded": performance.get("contactSheetsIncluded"),
        "stageTimingsMs": stage_timings,
        "recognizeTotalMs": _number(stage_timings.get("recognizeTotal")),
        "prepareTotalMs": _number(stage_timings.get("prepareTotal")),
        "prepareConstrainedInputMs": _number(stage_timings.get("prepareConstrainedInput")),
        "importsMs": _number(stage_timings.get("imports")),
        "rembgSessionMs": _number(stage_timings.get("rembgSession")),
        "loadImagesMs": _number(stage_timings.get("loadImages")),
        "rembgAMs": _number(stage_timings.get("rembgA")),
        "rembgBMs": _number(stage_timings.get("rembgB")),
        "hullFitAMs": _number(stage_timings.get("hullFitA")),
        "hullFitBMs": _number(stage_timings.get("hullFitB")),
        "selectGuardedPairMs": _number(stage_timings.get("selectGuardedPair")),
        "legacyFallbackMs": _number(stage_timings.get("legacyFallback")),
    }


def _score_row(row: Mapping[str, Any], endpoint: str, timeout: float) -> Dict[str, Any]:
    if row.get("status") == "skipped_missing_local_file":
        return dict(row)
    set_id = str(row["setId"])
    image_a = row["imageA"]
    image_b = row["imageB"]
    ground_truth = row["groundTruth"]
    started = time.perf_counter()
    _sha, _raw, expected, canonicalized = parse_ground_truth(str(ground_truth))
    try:
        http_status, payload = _post_pair(endpoint, image_a, image_b, timeout)
    except Exception as exc:  # noqa: BLE001 - production smoke should keep going.
        return {
            "setId": set_id,
            "status": "request_exception",
            "error": f"{type(exc).__name__}: {exc}",
            "latencyMs": round((time.perf_counter() - started) * 1000),
        }
    state = payload.get("state") if isinstance(payload, Mapping) else None
    hamming = _hamming(state, expected) if isinstance(state, str) else None
    signals = payload.get("recognitionSignals") if isinstance(payload, Mapping) else {}
    constrained = signals.get("constrainedInference") if isinstance(signals, Mapping) else {}
    constrained_performance = (
        _constrained_performance(constrained)
        if isinstance(constrained, Mapping)
        else {}
    )
    return {
        "setId": set_id,
        "status": payload.get("status") if isinstance(payload, Mapping) else "malformed_response",
        "httpStatus": http_status,
        "latencyMs": round((time.perf_counter() - started) * 1000),
        "state": state,
        "expectedState": expected,
        "groundTruthCanonicalized": canonicalized,
        "exactMatch": hamming == 0 if hamming is not None else False,
        "hamming": hamming,
        "stickersCorrect": (54 - hamming) if hamming is not None else 0,
        "reason": payload.get("reason") if isinstance(payload, Mapping) else None,
        "recognitionCategory": payload.get("recognitionCategory") if isinstance(payload, Mapping) else None,
        "constrainedSelected": constrained.get("selected") if isinstance(constrained, Mapping) else None,
        "constrainedFallbackToLegacy": constrained.get("fallbackToLegacy") if isinstance(constrained, Mapping) else None,
        "recommendedMethod": constrained.get("recommendedMethod") if isinstance(constrained, Mapping) else None,
        "twoViewStatus": (
            (constrained.get("twoViewConsistencyRepair") or {}).get("status")
            if isinstance(constrained, Mapping) and isinstance(constrained.get("twoViewConsistencyRepair"), Mapping)
            else None
        ),
        **constrained_performance,
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    values = sorted(
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
    )
    if not values:
        return {"count": 0}

    def pct(p: float) -> float:
        index = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
        return round(values[index], 2)

    return {
        "count": len(values),
        "min": round(values[0], 2),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": round(values[-1], 2),
        "avg": round(sum(values) / len(values), 2),
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scored = [row for row in rows if row.get("status") != "skipped_missing_local_file"]
    exact = [row for row in scored if row.get("exactMatch") is True]
    within3 = [row for row in scored if isinstance(row.get("hamming"), int) and int(row["hamming"]) <= 3]
    rejected = [row for row in scored if row.get("status") != "success"]
    return {
        "rowCount": len(rows),
        "scoredCount": len(scored),
        "exactCount": len(exact),
        "within3Count": len(within3),
        "rejectedCount": len(rejected),
        "missingLocalFileCount": len(rows) - len(scored),
        "recommendedMethodCounts": _counts(row.get("recommendedMethod") for row in scored),
        "recognitionCategoryCounts": _counts(row.get("recognitionCategory") for row in scored),
        "twoViewStatusCounts": _counts(row.get("twoViewStatus") for row in scored),
        "performanceSchemaCounts": _counts(row.get("performanceSchema") for row in scored),
        "contactSheetsIncludedCounts": _counts(row.get("contactSheetsIncluded") for row in scored),
        "timings": {
            "latencyMs": _metric_summary(scored, "latencyMs"),
            "recognizeTotalMs": _metric_summary(scored, "recognizeTotalMs"),
            "prepareTotalMs": _metric_summary(scored, "prepareTotalMs"),
            "prepareConstrainedInputMs": _metric_summary(scored, "prepareConstrainedInputMs"),
            "importsMs": _metric_summary(scored, "importsMs"),
            "rembgSessionMs": _metric_summary(scored, "rembgSessionMs"),
            "loadImagesMs": _metric_summary(scored, "loadImagesMs"),
            "rembgAMs": _metric_summary(scored, "rembgAMs"),
            "rembgBMs": _metric_summary(scored, "rembgBMs"),
            "hullFitAMs": _metric_summary(scored, "hullFitAMs"),
            "hullFitBMs": _metric_summary(scored, "hullFitBMs"),
            "selectGuardedPairMs": _metric_summary(scored, "selectGuardedPairMs"),
            "legacyFallbackMs": _metric_summary(scored, "legacyFallbackMs"),
        },
    }


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = "none" if value is None or value == "" else str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _render_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    rows = payload["rows"]
    non_exact = [
        row for row in rows
        if row.get("status") != "skipped_missing_local_file" and row.get("exactMatch") is not True
    ]
    lines = [
        "# Deployed constrained recognizer scoreboard",
        "",
        f"Generated: `{payload['generatedAtUtc']}`",
        f"Endpoint: `{payload['endpoint']}`",
        f"Manifest: `{payload['manifest']}`",
        "",
        "## Summary",
        "",
        "| Rows | Scored | Exact | Within 3 | Rejected | Missing local files |",
        "|---:|---:|---:|---:|---:|---:|",
        (
            f"| {summary['rowCount']} | {summary['scoredCount']} | "
            f"{summary['exactCount']} | {summary['within3Count']} | "
            f"{summary['rejectedCount']} | {summary['missingLocalFileCount']} |"
        ),
        "",
        "Recommended methods:",
        "",
    ]
    for key, value in summary["recommendedMethodCounts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Two-view repair statuses:", ""])
    for key, value in summary["twoViewStatusCounts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Performance schemas:", ""])
    for key, value in summary["performanceSchemaCounts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Timing summary:", ""])
    lines.extend([
        "| Metric | Count | Min | P50 | P90 | Max | Avg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key, metric in summary["timings"].items():
        if metric.get("count", 0) == 0:
            continue
        lines.append(
            f"| `{key}` | {metric['count']} | {metric['min']} | {metric['p50']} | "
            f"{metric['p90']} | {metric['max']} | {metric['avg']} |"
        )
    lines.extend(["", "## Non-exact Rows", ""])
    if not non_exact:
        lines.append("_None._")
    else:
        lines.extend([
            "| Set | Status | Hamming | Category | Method | Reason |",
            "|---|---|---:|---|---|---|",
        ])
        for row in non_exact:
            reason = str(row.get("reason") or row.get("error") or "").replace("|", "\\|")
            lines.append(
                f"| {row.get('setId')} | {row.get('status')} | "
                f"{row.get('hamming')} | {row.get('recognitionCategory')} | "
                f"{row.get('recommendedMethod')} | {reason[:160]} |"
            )
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--mode", default="constrained")
    parser.add_argument("--only-sets", nargs="*", default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out-json", type=Path, default=Path("/tmp/deployed_constrained_scoreboard.json"))
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    endpoint = _recognize_url(args.endpoint, args.mode)
    rows = _load_manifest(args.manifest, set(args.only_sets or []) or None)
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        future_to_set = {
            executor.submit(_score_row, row, endpoint, args.timeout): row.get("setId")
            for row in rows
        }
        for future in concurrent.futures.as_completed(future_to_set):
            result = future.result()
            results.append(result)
            hamming = result.get("hamming")
            score = f"{54 - hamming}/54" if isinstance(hamming, int) else "0/54"
            print(
                f"[deployed-score] {result.get('setId')}: "
                f"{result.get('status')} {score} {result.get('latencyMs')}ms",
                file=sys.stderr,
            )
    results.sort(key=lambda row: str(row.get("setId")))
    payload = {
        "schema": "deployed_constrained_recognizer_scoreboard_v1",
        "generatedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "endpoint": endpoint,
        "manifest": str(args.manifest),
        "summary": _summary(results),
        "rows": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[deployed-score] wrote {args.out_json}", file=sys.stderr)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(_render_report(payload), encoding="utf-8")
        print(f"[deployed-score] wrote {args.report}", file=sys.stderr)
    s = payload["summary"]
    print(
        f"exact={s['exactCount']}/{s['scoredCount']} "
        f"within3={s['within3Count']}/{s['scoredCount']} "
        f"rejected={s['rejectedCount']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
