"""Tests for tools/analyze_shadow_traces.py.

The analyzer is mostly aggregation logic over pre-computed per-row
records — rembg/image-loading is the slow part and not what we want
to exercise in unit tests. So the tests focus on:

  - _summarize: counts roll up correctly across status/acceptance/side
  - render_report: the markdown surfaces accept/reject totals, gate
    histograms, rejected-row punch list, and warning list without
    crashing on empty/edge inputs
  - main: end-to-end smoke with a synthetic axis_truth + manifest +
    monkeypatched analyze_row (no rembg, no real images)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_shadow_traces as ast  # noqa: E402


def _row(
    key: str,
    *,
    accepted: bool | None = True,
    hard_failures: list[str] | None = None,
    warnings: list[str] | None = None,
    sticker: float | None = 500.0,
    spread_norm: float | None = 0.20,
    residual: float | None = 0.010,
    vertex_source: str | None = "affine",
    status: str = "accepted",
) -> Dict[str, Any]:
    """Build a per-row record matching what analyze_row returns."""
    rec: Dict[str, Any] = {
        "key": key,
        "side": key.rsplit("_", 1)[-1],
        "trace_status": status,
        "accepted": accepted,
        "selected": False,
        "mode": "shadow",
        "hard_failures": hard_failures or [],
        "warnings": warnings or [],
        "sticker_score_total": sticker,
        "mean_sticker_distance": (sticker / 27.0) if sticker is not None else None,
        "vertex_source": vertex_source,
        "vertex_cloud_spread_px": 180.0 if spread_norm is not None else None,
        "vertex_cloud_spread_norm": spread_norm,
        "projective_residual_norm": residual,
        "hexagon_diameter_px": 903.0,
        "projective_degeneracy": "none",
        "full_trace": {"side": rec_side_for(key)},
    }
    return rec


def rec_side_for(key: str) -> str:
    return key.rsplit("_", 1)[-1]


# --- _summarize ------------------------------------------------------


def test_summarize_counts_acceptance_split() -> None:
    rows = [
        _row("12_A", accepted=True),
        _row("12_B", accepted=True),
        _row("14_A", accepted=False, hard_failures=["sticker_score_total_above_hard"]),
        _row("14_B", accepted=None, status="harness_error"),
    ]
    s = ast._summarize(rows)
    assert s["total_rows"] == 4
    assert s["by_acceptance"] == {
        "accepted": 2,
        "rejected": 1,
        "no_decision": 1,
    }


def test_summarize_hard_failure_histogram_aggregates_by_gate() -> None:
    """Codex P2 on PR #292: messages like 'vertex_cloud_spread_px=
    252.4; warning 240.0' carry the measured value, so the raw
    string is unique per row. The histogram should aggregate by
    GATE NAME (prefix), not by full message — otherwise 3 hits of
    the same gate fragment into 3 count-1 buckets.
    """
    rows = [
        _row("a_A", accepted=False, hard_failures=[
            "sticker_score_total=950; hard 900",
        ]),
        _row("b_A", accepted=False, hard_failures=[
            "sticker_score_total=1020; hard 900",  # SAME gate, diff value
        ]),
        _row("c_B", accepted=False, hard_failures=[
            "vertex_cloud_spread_px=350; hard 320",
            "projective_residual_norm=0.030; hard 0.025",
        ]),
        _row("d_A", accepted=True),
    ]
    s = ast._summarize(rows)
    # By-gate aggregation: 2 hits of sticker_score, 1 each of others
    assert s["hard_failure_gates"] == {
        "sticker_score_total": 2,
        "vertex_cloud_spread_px": 1,
        "projective_residual_norm": 1,
    }
    # By-message also preserved so the report can show specific values
    assert s["hard_failure_messages"]["sticker_score_total=950; hard 900"] == 1
    assert s["hard_failure_messages"]["sticker_score_total=1020; hard 900"] == 1


def test_summarize_warning_histogram_aggregates_by_gate() -> None:
    """Same shape as hard_failure but on warnings — three hits of
    the vertex_cloud_spread warning at three different measured
    values should report '3' once, not '1' three times.
    """
    rows = [
        _row("a_A", warnings=["vertex_cloud_spread_px=252.4; warning 240.0"]),
        _row("b_A", warnings=["vertex_cloud_spread_px=265.7; warning 240.0"]),
        _row("c_A", warnings=["vertex_cloud_spread_px=267.6; warning 240.0"]),
        _row("d_B"),  # no warnings
    ]
    s = ast._summarize(rows)
    assert s["warning_gates"] == {"vertex_cloud_spread_px": 3}
    # All 3 individual messages also recorded (count 1 each — they're
    # genuinely distinct strings even though the gate is the same)
    assert len(s["warning_messages"]) == 3
    assert all(v == 1 for v in s["warning_messages"].values())


def test_summarize_vertex_source_breakdown_accepted_only() -> None:
    rows = [
        _row("a_A", accepted=True, vertex_source="affine"),
        _row("b_A", accepted=True, vertex_source="affine"),
        _row("c_A", accepted=True, vertex_source="projective"),
        # Rejected row's vertex_source must NOT show up in the accepted
        # tally even though the field is populated.
        _row("d_B", accepted=False, vertex_source="projective"),
    ]
    s = ast._summarize(rows)
    assert s["vertex_source_accepted"] == {"affine": 2, "projective": 1}


def test_summarize_per_side_acceptance_split() -> None:
    rows = [
        _row("a_A", accepted=True),
        _row("b_A", accepted=False, hard_failures=["x"]),
        _row("c_B", accepted=True),
        _row("d_B", accepted=True),
    ]
    s = ast._summarize(rows)
    assert s["by_side_acceptance"] == {
        "A_accepted": 1,
        "A_rejected": 1,
        "B_accepted": 2,
    }


def test_summarize_stats_excludes_rejected_and_missing() -> None:
    rows = [
        _row("a_A", accepted=True, spread_norm=0.10, sticker=300.0, residual=0.005),
        _row("b_A", accepted=True, spread_norm=0.30, sticker=500.0, residual=0.015),
        # Rejected row's signals are intentionally extreme — must NOT
        # appear in accepted-only stats.
        _row("c_A", accepted=False, spread_norm=0.99, sticker=9999.0, residual=0.999),
        # No-decision row with None signals — also must not contaminate.
        _row("d_B", accepted=None, spread_norm=None, sticker=None, residual=None),
    ]
    s = ast._summarize(rows)
    assert s["spread_norm_accepted_stats"]["n"] == 2
    assert s["spread_norm_accepted_stats"]["max"] == 0.30
    assert s["sticker_score_accepted_stats"]["max"] == 500.0
    assert s["projective_residual_accepted_stats"]["max"] == 0.015


def test_summarize_empty_corpus_does_not_crash() -> None:
    s = ast._summarize([])
    assert s["total_rows"] == 0
    assert s["spread_norm_accepted_stats"] == {"n": 0}


def test_summarize_by_status_includes_skip_and_error_reasons() -> None:
    """Codex P2 on PR #292: rows that error before hitting the gate
    only have `status` (no `trace_status`). The by_status histogram
    must surface those reasons individually, not collapse them into
    a None bucket.
    """
    rows = [
        {"key": "a_A", "side": "A", "trace_status": "accepted",
         "accepted": True, "hard_failures": [], "warnings": []},
        {"key": "b_A", "side": "A", "status": "skipped_no_image_path",
         "error": "...", "hard_failures": [], "warnings": []},
        {"key": "c_A", "side": "A", "status": "skipped_unresolved_image",
         "error": "...", "hard_failures": [], "warnings": []},
        {"key": "d_B", "side": "B", "status": "harness_error",
         "error": "...", "hard_failures": [], "warnings": []},
    ]
    s = ast._summarize(rows)
    # All 4 distinct kinds of row surface separately — no None bucket
    assert s["by_status"] == {
        "accepted": 1,
        "skipped_no_image_path": 1,
        "skipped_unresolved_image": 1,
        "harness_error": 1,
    }
    assert None not in s["by_status"]


def test_normalize_gate_token_extracts_prefix() -> None:
    """Direct test of the message → gate-name reducer."""
    assert ast._normalize_gate_token(
        "vertex_cloud_spread_px=252.4; warning 240.0"
    ) == "vertex_cloud_spread_px"
    assert ast._normalize_gate_token(
        "sticker_score_total=950; hard 900"
    ) == "sticker_score_total"
    # Falls back to the raw message if no recognizable prefix
    assert ast._normalize_gate_token(
        "=== unrecognizable ==="
    ) == "=== unrecognizable ==="


# --- render_report ---------------------------------------------------


def test_render_report_includes_headline_counts() -> None:
    rows = [
        _row("a_A", accepted=True),
        _row("b_A", accepted=False, hard_failures=["sticker_score_total=950; hard 900"]),
    ]
    s = ast._summarize(rows)
    md = ast.render_report(
        s, rows,
        head_sha="abc123",
        axis_truth_path=Path("axis.json"),
        trace_path=Path("trace.json"),
    )
    assert "**1/2 (50.0%)" in md  # accept headline
    assert "**1/2 (50.0%)" in md  # reject headline
    assert "abc123" in md
    assert "sticker_score_total" in md


def test_render_report_rejected_punch_list() -> None:
    rows = [
        _row("a_A", accepted=False, hard_failures=["sticker_score_total=950; hard 900"]),
        _row("b_B", accepted=True),
    ]
    md = ast.render_report(
        ast._summarize(rows), rows,
        head_sha=None,
        axis_truth_path=Path("axis.json"),
        trace_path=Path("trace.json"),
    )
    assert "## Rejected-row punch list" in md
    assert "| a_A | A |" in md
    assert "| b_B | B |" not in md  # accepted rows excluded from punch list


def test_render_report_empty_corpus() -> None:
    s = ast._summarize([])
    # Should produce a valid markdown document even with zero rows.
    md = ast.render_report(
        s, [],
        head_sha=None,
        axis_truth_path=Path("axis.json"),
        trace_path=Path("trace.json"),
    )
    assert "0/0" in md
    assert "_None._" in md  # rejected punch list shows None
    assert "_No hard failures observed" in md


def test_render_report_no_warnings_shows_explicit_none() -> None:
    rows = [_row("a_A", accepted=True)]  # default has no warnings
    md = ast.render_report(
        ast._summarize(rows), rows,
        head_sha=None,
        axis_truth_path=Path("axis.json"),
        trace_path=Path("trace.json"),
    )
    assert "_No warnings observed._" in md


# --- main (end-to-end with monkeypatched analyze_row) ----------------


def test_main_end_to_end_with_synthetic_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end smoke: synthetic axis_truth + manifest + stubbed
    analyze_row. Exercises CLI arg parsing, row iteration, manifest
    join, and the JSON+markdown output side-effects.
    """
    axis_truth = {
        "12_A": {"vertex": [100, 200], "axis_x": [10, 0], "axis_y": [0, 10],
                  "axis_z": [10, 10], "approved": True},
        "12_B": {"vertex": [100, 200], "axis_x": [10, 0], "axis_y": [0, 10],
                  "axis_z": [10, 10], "approved": True},
        # Unapproved row — must be filtered out before iteration.
        "99_A": {"vertex": [0, 0], "axis_x": [0, 0], "axis_y": [0, 0],
                  "axis_z": [0, 0], "approved": False},
        # Row missing from manifest — must surface as skipped, not crash.
        "77_A": {"vertex": [0, 0], "axis_x": [0, 0], "axis_y": [0, 0],
                  "axis_z": [0, 0], "approved": True},
    }
    manifest = {
        "schemaVersion": 1,
        "name": "test",
        "description": "test",
        "supportedArchitectures": [],
        "pairs": [
            {
                "setId": "12",
                "imageAPath": "/tmp/fake_A.jpg",
                "imageBPath": "/tmp/fake_B.jpg",
            },
        ],
    }
    axis_truth_path = tmp_path / "axis.json"
    manifest_path = tmp_path / "manifest.json"
    axis_truth_path.write_text(json.dumps(axis_truth))
    manifest_path.write_text(json.dumps(manifest))
    trace_out = tmp_path / "trace.json"
    report_out = tmp_path / "report.md"

    # Stub analyze_row to return a canned per-key record without
    # touching rembg or images. ALSO stub _resolve_image_path so the
    # 12_* rows resolve to a fake path.
    def fake_resolve(raw_path, set_id, side, image_roots,
                     expected_sha256=None):
        return Path(raw_path)

    def fake_analyze_row(sess, key, image_path, *, max_image_dim):
        return {
            "key": key, "side": key.rsplit("_", 1)[-1],
            "trace_status": "accepted", "accepted": True, "selected": False,
            "mode": "shadow", "hard_failures": [], "warnings": [],
            "sticker_score_total": 400.0,
            "mean_sticker_distance": 14.8,
            "vertex_source": "affine",
            "vertex_cloud_spread_px": 150.0,
            "vertex_cloud_spread_norm": 0.18,
            "projective_residual_norm": 0.008,
            "hexagon_diameter_px": 900.0,
            "projective_degeneracy": "none",
            "full_trace": {"key": key},
        }

    monkeypatch.setattr(ast, "_resolve_image_path", fake_resolve)
    monkeypatch.setattr(ast, "analyze_row", fake_analyze_row)
    # Stub the rembg session init so we don't try to download a model
    # in a unit test even though analyze_row never uses it.
    monkeypatch.setattr(ast, "_get_rembg_session", lambda: object())

    rc = ast.main([
        "--axis-truth", str(axis_truth_path),
        "--manifest", str(manifest_path),
        # Force an empty hard-case manifest so the test exercises
        # only the primary manifest path; the manifest-miss row (77_A)
        # then takes the sentinel-path → fake_resolve route.
        "--hard-case-manifest", str(tmp_path / "no_hard.json"),
        "--trace-out", str(trace_out),
        "--report-out", str(report_out),
    ])
    assert rc == 0
    assert trace_out.exists()
    assert report_out.exists()

    artifact = json.loads(trace_out.read_text())
    keys = [r["key"] for r in artifact["per_row"]]
    # Unapproved 99_A must be filtered out at iteration time.
    assert "99_A" not in keys
    # Codex P2 #1 on PR #292: 77_A is NOT in either manifest, but
    # the analyzer now hands a sentinel path to _resolve_image_path
    # so the pattern-search fallback runs. fake_resolve returns
    # Path(raw_path) unconditionally, so 77_A now gets "analyzed"
    # rather than skipped at the manifest stage — exactly the
    # behavior measure_hull_labels_corpus uses.
    assert sorted(keys) == ["12_A", "12_B", "77_A"]
    by_status = {r["key"]: (r.get("trace_status") or r.get("status"))
                 for r in artifact["per_row"]}
    assert by_status["12_A"] == "accepted"
    assert by_status["12_B"] == "accepted"
    # 77_A now reaches the gate via sentinel path → fake_resolve →
    # fake_analyze_row → accepted (since the stub always accepts)
    assert by_status["77_A"] == "accepted"

    md = report_out.read_text()
    assert "12_A" in md or "## Headline" in md  # rendered report
    assert "**3/3" in md  # all 3 accept (77_A no longer skipped)


def test_main_with_limit_truncates_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--limit N should process only the first N approved rows."""
    axis_truth = {
        f"{i:02d}_A": {"vertex": [0, 0], "axis_x": [0, 0], "axis_y": [0, 0],
                       "axis_z": [0, 0], "approved": True}
        for i in range(1, 11)
    }
    manifest = {"pairs": [
        {"setId": f"{i:02d}", "imageAPath": f"/tmp/{i}.jpg",
         "imageBPath": f"/tmp/{i}b.jpg"}
        for i in range(1, 11)
    ]}
    axis_truth_path = tmp_path / "axis.json"
    manifest_path = tmp_path / "manifest.json"
    axis_truth_path.write_text(json.dumps(axis_truth))
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(
        ast, "_resolve_image_path",
        lambda raw, sid, side, roots, expected_sha256=None: Path(raw),
    )
    monkeypatch.setattr(
        ast, "analyze_row",
        lambda s, k, p, *, max_image_dim: {
            "key": k, "side": "A", "trace_status": "accepted",
            "accepted": True, "selected": False, "mode": "shadow",
            "hard_failures": [], "warnings": [],
            "sticker_score_total": 400.0, "mean_sticker_distance": 14.8,
            "vertex_source": "affine",
            "vertex_cloud_spread_px": 150.0,
            "vertex_cloud_spread_norm": 0.18,
            "projective_residual_norm": 0.008,
            "hexagon_diameter_px": 900.0,
            "projective_degeneracy": "none",
            "full_trace": {},
        },
    )
    monkeypatch.setattr(ast, "_get_rembg_session", lambda: object())

    rc = ast.main([
        "--axis-truth", str(axis_truth_path),
        "--manifest", str(manifest_path),
        "--trace-out", str(tmp_path / "trace.json"),
        "--report-out", str(tmp_path / "report.md"),
        "--limit", "3",
    ])
    assert rc == 0
    artifact = json.loads((tmp_path / "trace.json").read_text())
    assert len(artifact["per_row"]) == 3
