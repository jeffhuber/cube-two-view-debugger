"""Unit tests for `tools/diagnose_chirality_failures.py`.

Diagnostic-only tool; tests verify the classifier logic and the
end-to-end pipeline against the existing matrix fixture. No production
behavior is touched."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.diagnose_chirality_failures import (
    DEFAULT_MATRIX,
    _classify_failure,
    analyze,
    render_report,
)


# ----- _classify_failure -----


def test_classify_ambiguous_no_correction():
    row = {
        "phase_check": "ambiguous_no_correction",
        "phase_darkness_separation": 4.2,
        "err_near_deg": 50.0,
    }
    mode, rationale = _classify_failure(row)
    assert mode == "DETECTOR_AMBIGUOUS"
    assert "sep=4.2" in rationale


def test_classify_correct_but_wrong():
    """phase_check says 'correct' but err_near is high → detector mis-called
    the polarity on this row."""
    row = {
        "phase_check": "correct",
        "phase_darkness_separation": -25.0,
        "err_near_deg": 55.8,
    }
    mode, rationale = _classify_failure(row)
    assert mode == "DETECTOR_WRONG_CALL"
    assert "correct" in rationale
    assert "55.8" in rationale


def test_classify_corrected_but_still_wrong():
    row = {
        "phase_check": "corrected_60deg_flip",
        "phase_darkness_separation": 20.0,
        "err_near_deg": 58.0,
    }
    mode, rationale = _classify_failure(row)
    assert mode == "DETECTOR_WRONG_CALL"
    assert "flipped" in rationale
    assert "58" in rationale


def test_classify_flip_suggested_not_applied():
    row = {
        "phase_check": "flip_suggested_diagnostic_only",
        "phase_darkness_separation": 15.0,
        "err_near_deg": 50.0,
    }
    mode, _rationale = _classify_failure(row)
    assert mode == "FLIP_SUGGESTED_NOT_APPLIED"


def test_classify_unexpected_phase_check_is_pipeline_bug():
    row = {"phase_check": "this_value_does_not_exist", "err_near_deg": 50.0}
    mode, rationale = _classify_failure(row)
    assert mode == "PIPELINE_BUG"
    assert "this_value_does_not_exist" in rationale


# ----- analyze() end-to-end on the real fixture -----


@pytest.fixture(scope="module")
def analysis():
    """Run analyze() once per test module — small fixture, fast.
    (Greptile P2 on PR #250: explicit `scope="module"` matches the
    documented intent; default `scope="function"` would re-invoke
    analyze() once per consuming test.)"""
    return analyze(DEFAULT_MATRIX)


def test_analyze_returns_expected_keys(analysis):
    expected = {
        "matrix_path",
        "total_rows",
        "total_cases",
        "per_category_counts",
        "chirality_rows",
        "failure_mode_counts",
        "sep_stats_by_category_and_phase_check",
        "meta_signal_stats",
    }
    assert set(analysis.keys()) == expected


def test_analyze_total_rows_matches_fixture(analysis):
    # 70 cases × 2 runs = 140 rows expected on the current matrix.
    assert analysis["total_cases"] == 70
    assert analysis["total_rows"] == 140


def test_analyze_finds_chirality_failures(analysis):
    # The matrix has known chirality-failure rows — assert at least some
    # were enumerated. Tight number isn't pinned (matrix can grow).
    assert len(analysis["chirality_rows"]) >= 10


def test_analyze_failure_modes_only_classified_when_categorized(analysis):
    """Every row in `chirality_rows` should have been categorized."""
    for r in analysis["chirality_rows"]:
        assert "_failure_mode" in r
        assert r["_failure_mode"] in {
            "DETECTOR_AMBIGUOUS",
            "DETECTOR_WRONG_CALL",
            "FLIP_SUGGESTED_NOT_APPLIED",
            "PIPELINE_BUG",
        }
        assert r.get("category") in {
            "CHIRALITY_MISS",
            "CHIRALITY_FALSE_FLIP",
        }


def test_analyze_failure_mode_counts_sum_correctly(analysis):
    total = sum(analysis["failure_mode_counts"].values())
    assert total == len(analysis["chirality_rows"])


# ----- render_report basic shape -----


def test_render_report_contains_required_sections(analysis):
    report = render_report(analysis)
    for section in [
        "# Chirality detector failure analysis",
        "## Per-category row counts",
        "## Chirality-row failure modes",
        "## Key findings",
        "## Per-row detail",
        "## Recommended next experiments",
    ]:
        assert section in report, f"missing section header: {section}"


def test_render_report_no_chirality_rows_returns_short_message():
    empty_analysis = {
        "matrix_path": "/tmp/fake",
        "total_rows": 0,
        "total_cases": 0,
        "per_category_counts": {},
        "chirality_rows": [],
        "failure_mode_counts": {},
        "sep_stats_by_category_and_phase_check": {},
        "meta_signal_stats": {"right_n": 0, "wrong_n": 0, "features": {}},
    }
    report = render_report(empty_analysis)
    # No chirality rows → render_report returns early before the
    # meta-signal section, so the report should be short and just say
    # "No chirality-failure rows in this matrix."
    assert "No chirality-failure rows" in report


def test_analyze_meta_signal_stats_shape(analysis):
    """The meta_signal_stats dict has expected shape + identifies at
    least one feature analysis."""
    meta = analysis["meta_signal_stats"]
    assert set(meta.keys()) == {"right_n", "wrong_n", "features"}
    assert meta["right_n"] > 0
    assert meta["wrong_n"] > 0
    assert "junction_score_at_ensemble" in meta["features"]
    js = meta["features"]["junction_score_at_ensemble"]
    assert "right" in js and "wrong" in js and "iqr_overlap" in js


def test_render_report_sep_stats_table_escapes_pipes(analysis):
    """Codex P2 round-2 regression: keys in
    `sep_stats_by_category_and_phase_check` contain a literal `|`
    (`category | phase_check`). Markdown renderer would interpret an
    unescaped pipe as a column boundary, shifting column values.
    Verify the rendered table escapes the pipe so the table is valid.

    Scope: only inspect rows in the sep-stats table specifically (bounded
    by the section header and the next `## ` header)."""
    report = render_report(analysis)
    in_section = False
    found_any_row = False
    for line in report.split("\n"):
        if "Separation distribution" in line:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break  # left the sep-stats section
        if not in_section:
            continue
        if line.startswith("| ") and not line.startswith("| Category"):
            if "---" in line:
                continue
            # Effective pipes = total pipes minus escaped pipes.
            n_pipes = line.count("|") - line.count(r"\|")
            assert n_pipes == 6, (
                f"sep-stats table row has unescaped pipe in key, "
                f"got {n_pipes} effective pipes in {line!r}"
            )
            found_any_row = True
    assert found_any_row, "expected at least one sep-stats table row"
