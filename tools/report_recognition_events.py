#!/usr/bin/env python3
"""Summarize the durable CTVD recognition event SQLite log.

The recognizer stores metadata-only events in ``recognition_events``. This tool
turns that log into product feedback: success/reject rates, constrained status,
failure reasons, latency distributions, client source breakdowns, and recent
attempts. It never reads or writes image bytes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib import parse, request

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "runs" / "recognition_events.sqlite3"
DEFAULT_ENDPOINT = "https://api.cubesnap.app/api/recognition-events/report"
EVENT_DB_ENV = "CUBE_RECOGNITION_EVENT_DB_PATH"
DEFAULT_STAGES = (
    "recognizeTotal",
    "prepareConstrainedInput",
    "prepareTotal",
    "rembgA",
    "rembgB",
    "hullFitWall",
    "hullFitA",
    "hullFitB",
    "selectGuardedPair",
    "selectGuardedPair.thresholdPairEnumeration",
    "selectGuardedPair.currentPairEvaluatePair",
    "selectGuardedPair.currentGuardSelection",
    "selectGuardedPair.lightPairEvaluatePair",
    "selectGuardedPair.lightPairSearch",
    "selectGuardedPair.canonicalLightFilter",
    "selectGuardedPair.lightGuardSelection",
    "selectGuardedPair.fullPairEvaluatePair",
    "selectGuardedPair.fullPairSearch",
    "selectGuardedPair.fullGuardSelection",
    "selectGuardedPair.selectedFullPairEvaluatePair",
    "selectGuardedPair.yawInference",
    "selectGuardedPair.repairCanonicalOnly",
    "selectGuardedPair.repairWithLegal",
    "selectGuardedPair.rankAndScore",
    "promotionGate",
    "buildCandidate",
    "legacyFallback",
)


def default_db_path() -> Path:
    raw = os.environ.get(EVENT_DB_ENV)
    if raw and raw.strip().lower() not in {"", "0", "false", "off", "none"}:
        return Path(raw).expanduser()
    return DEFAULT_DB


def _json_loads(value: Any, fallback: Any) -> Any:
    if not isinstance(value, str) or not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    counts: Counter[str] = Counter("none" if value in (None, "") else str(value) for value in values)
    return dict(sorted(counts.items()))


def _metric(values: Sequence[float]) -> Dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}

    def pct(p: float) -> float:
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return round(ordered[index], 2)

    return {
        "count": len(ordered),
        "min": round(ordered[0], 2),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": round(ordered[-1], 2),
        "avg": round(sum(ordered) / len(ordered), 2),
    }


def _since_iso(hours: Optional[float]) -> Optional[str]:
    if hours is None:
        return None
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=float(hours))
    return since.isoformat()


def _query_rows(db: sqlite3.Connection, *, since_hours: Optional[float]) -> List[sqlite3.Row]:
    where = ""
    params: Dict[str, Any] = {}
    since = _since_iso(since_hours)
    if since:
        where = "WHERE created_at >= :since"
        params["since"] = since
    return db.execute(
        f"""
        SELECT *
        FROM recognition_events
        {where}
        ORDER BY created_at ASC, id ASC
        """,
        params,
    ).fetchall()


def _event_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _json_loads(row.get("event_json"), {})
    return payload if isinstance(payload, Mapping) else {}


def _stage_timings(row: Mapping[str, Any]) -> Mapping[str, Any]:
    event = _event_payload(row)
    performance = event.get("performance") if isinstance(event.get("performance"), Mapping) else {}
    timings = performance.get("stageTimingsMs") if isinstance(performance.get("stageTimingsMs"), Mapping) else {}
    return timings if isinstance(timings, Mapping) else {}


def _failure_reason(row: Mapping[str, Any]) -> str:
    event = _event_payload(row)
    result = event.get("result") if isinstance(event.get("result"), Mapping) else {}
    reason = result.get("reason")
    if isinstance(reason, str) and reason:
        return reason
    failed = _json_loads(row.get("failed_checks_json"), [])
    if isinstance(failed, list) and failed:
        return ",".join(str(item) for item in failed)
    return "none"


def build_summary(rows: Sequence[Mapping[str, Any]], *, recent_limit: int = 20) -> Dict[str, Any]:
    stage_values: Dict[str, List[float]] = {stage: [] for stage in DEFAULT_STAGES}
    for row in rows:
        timings = _stage_timings(row)
        for stage in DEFAULT_STAGES:
            value = timings.get(stage)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                stage_values[stage].append(float(value))

    recent = list(reversed(rows))[:recent_limit]
    return {
        "totalEvents": len(rows),
        "statusCounts": _counts(row.get("status") for row in rows),
        "recognitionCategoryCounts": _counts(row.get("recognition_category") for row in rows),
        "constrainedStatusCounts": _counts(row.get("constrained_status") for row in rows),
        "recommendedMethodCounts": _counts(row.get("recommended_method") for row in rows),
        "clientSourceCounts": _counts(row.get("client_source") for row in rows),
        "appVersionCounts": _counts(row.get("app_version") for row in rows),
        "failureReasonCounts": _counts(
            _failure_reason(row)
            for row in rows
            if row.get("status") != "success"
        ),
        "latencyMs": _metric([
            float(row["latency_ms"])
            for row in rows
            if isinstance(row.get("latency_ms"), (int, float))
        ]),
        "recognizeTotalMs": _metric([
            float(row["recognize_total_ms"])
            for row in rows
            if isinstance(row.get("recognize_total_ms"), (int, float))
        ]),
        "prepareConstrainedInputMs": _metric([
            float(row["prepare_constrained_input_ms"])
            for row in rows
            if isinstance(row.get("prepare_constrained_input_ms"), (int, float))
        ]),
        "stageTimingsMs": {
            stage: _metric(values)
            for stage, values in stage_values.items()
        },
        "recentAttempts": [
            {
                "createdAt": row.get("created_at"),
                "setId": row.get("set_id"),
                "status": row.get("status"),
                "category": row.get("recognition_category"),
                "constrainedStatus": row.get("constrained_status"),
                "recommendedMethod": row.get("recommended_method"),
                "latencyMs": row.get("latency_ms"),
                "clientSource": row.get("client_source"),
                "appVersion": row.get("app_version"),
                "failureReason": _failure_reason(row) if row.get("status") != "success" else None,
            }
            for row in recent
        ],
    }


def render_report(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Recognition Event Report",
        "",
        f"Generated: `{payload['generatedAtUtc']}`",
        f"Database: `{payload['database']}`",
        f"Since hours: `{payload.get('sinceHours') if payload.get('sinceHours') is not None else 'all'}`",
        "",
        "## Summary",
        "",
        f"- Total events: `{summary['totalEvents']}`",
        "",
        "Status counts:",
        "",
    ]
    for key, value in summary["statusCounts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Recognition categories:", ""])
    for key, value in summary["recognitionCategoryCounts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Client sources:", ""])
    for key, value in summary["clientSourceCounts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Failure reasons:", ""])
    if not summary["failureReasonCounts"]:
        lines.append("- _None._")
    else:
        for key, value in summary["failureReasonCounts"].items():
            lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Latency",
        "",
        "| Metric | Count | P50 | P90 | Max | Avg |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    latency_metrics = {
        "latencyMs": summary["latencyMs"],
        "recognizeTotalMs": summary["recognizeTotalMs"],
        "prepareConstrainedInputMs": summary["prepareConstrainedInputMs"],
        **{
            stage: metric
            for stage, metric in summary["stageTimingsMs"].items()
            if metric.get("count", 0)
        },
    }
    for key, metric in latency_metrics.items():
        if metric.get("count", 0) == 0:
            continue
        lines.append(
            f"| `{key}` | {metric['count']} | {metric['p50']} | "
            f"{metric['p90']} | {metric['max']} | {metric['avg']} |"
        )
    lines.extend(["", "## Recent Attempts", ""])
    if not summary["recentAttempts"]:
        lines.append("_None._")
    else:
        lines.extend([
            "| Time | Status | Category | Method | Latency ms | Source | Failure |",
            "|---|---|---|---|---:|---|---|",
        ])
        for row in summary["recentAttempts"]:
            failure = str(row.get("failureReason") or "").replace("|", "\\|")
            lines.append(
                f"| `{row.get('createdAt')}` | `{row.get('status')}` | "
                f"`{row.get('category')}` | `{row.get('recommendedMethod')}` | "
                f"{row.get('latencyMs')} | `{row.get('clientSource')}` | {failure[:120]} |"
            )
    lines.append("")
    return "\n".join(lines)


def _rows_as_dicts(rows: Sequence[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=default_db_path())
    parser.add_argument(
        "--endpoint",
        default=None,
        help=(
            "Fetch a production report from an HTTPS report endpoint instead "
            "of reading a local SQLite DB. Example: "
            f"{DEFAULT_ENDPOINT}"
        ),
    )
    parser.add_argument("--since-hours", type=float, default=None)
    parser.add_argument("--recent-limit", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    return parser.parse_args(argv)


def load_report_payload(
    db_path: Path,
    *,
    since_hours: Optional[float] = None,
    recent_limit: int = 20,
) -> Dict[str, Any]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5) as db:
        db.row_factory = sqlite3.Row
        rows = _rows_as_dicts(_query_rows(db, since_hours=since_hours))
    return {
        "schema": "recognition_event_report_v1",
        "generatedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "database": str(db_path),
        "sinceHours": since_hours,
        "summary": build_summary(rows, recent_limit=max(0, recent_limit)),
    }


def _endpoint_report_url(
    endpoint: str,
    *,
    since_hours: Optional[float],
    recent_limit: int,
) -> str:
    url = parse.urlparse(endpoint)
    query = dict(parse.parse_qsl(url.query, keep_blank_values=True))
    if since_hours is not None:
        query["sinceHours"] = str(since_hours)
    query["recentLimit"] = str(max(0, recent_limit))
    return parse.urlunparse(url._replace(query=parse.urlencode(query)))


def load_endpoint_report_payload(
    endpoint: str,
    *,
    since_hours: Optional[float] = None,
    recent_limit: int = 20,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    url = _endpoint_report_url(
        endpoint,
        since_hours=since_hours,
        recent_limit=recent_limit,
    )
    req = request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ctvd-recognition-event-report/1",
        },
    )
    with request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report endpoint returned {type(payload).__name__}, expected object")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.endpoint:
        payload = load_endpoint_report_payload(
            args.endpoint,
            since_hours=args.since_hours,
            recent_limit=args.recent_limit,
            timeout=args.timeout,
        )
    else:
        if not args.db.exists():
            print(f"recognition event DB not found: {args.db}", file=sys.stderr)
            return 2
        payload = load_report_payload(
            args.db,
            since_hours=args.since_hours,
            recent_limit=args.recent_limit,
        )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(payload), encoding="utf-8")

    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
