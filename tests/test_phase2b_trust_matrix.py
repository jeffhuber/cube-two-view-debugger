"""Tests for tools/phase2b_trust_matrix.py.

Covers the pure-function half of the tool: outcome categorization,
matrix join, candidate-rule evaluation. The CLI / file-IO side is
exercised by the script itself when run against the committed
fixtures.
"""

from __future__ import annotations

from typing import Any, Dict

from tools import phase2b_trust_matrix as p2b


def test_categorize_outcome_maps_all_categories():
    assert p2b.categorize_outcome("GOOD") == "GOOD"
    assert p2b.categorize_outcome("MARGINAL") == "MARGINAL"
    # All three catastrophic categories map to CATASTROPHIC
    assert p2b.categorize_outcome("CHIRALITY_MISS") == "CATASTROPHIC"
    assert p2b.categorize_outcome("CHIRALITY_FALSE_FLIP") == "CATASTROPHIC"
    assert p2b.categorize_outcome("TRUE_GEOMETRY_FAIL") == "CATASTROPHIC"
    # Anything unrecognized maps to UNKNOWN (won't be counted against bar)
    assert p2b.categorize_outcome("?") == "UNKNOWN"
    assert p2b.categorize_outcome("SOMETHING_NEW") == "UNKNOWN"


def _stub_post_218(rows):
    """rows: list of (case, run_idx, category, phase_sep) — assemble post_218
    in the schema expected by build_matrix."""
    by_case: Dict[str, Any] = {}
    for case, run_idx, category, phase_sep in rows:
        by_case.setdefault(case, []).append({
            "run": run_idx,
            "category": category,
            "phase_sep": phase_sep,
            "phase_check": "stub",
        })
    return {"by_case": by_case, "summary": {}}


def _stub_cv_local(per_case_status):
    """per_case_status: dict[case] -> status string."""
    by_case = {case: [{"status": status, "run": 0}] for case, status in per_case_status.items()}
    return {"by_case": by_case, "summary": {}}


def test_build_matrix_joins_per_run_with_per_case_cv_local():
    post = _stub_post_218([
        ("12_A", 0, "GOOD", 5.2),
        ("12_A", 1, "GOOD", 4.8),
        ("17_A", 0, "CHIRALITY_MISS", 18.1),
    ])
    cv = _stub_cv_local({"12_A": "ok", "17_A": "cluster_pattern_mismatch"})
    rows = p2b.build_matrix(post, cv)

    assert len(rows) == 3
    # 12_A's two runs both get cv_status="ok" / cv_consistent=True
    r12_runs = [r for r in rows if r.case == "12_A"]
    assert len(r12_runs) == 2
    assert all(r.cv_status == "ok" and r.cv_consistent for r in r12_runs)
    assert all(r.outcome == "GOOD" for r in r12_runs)
    # 17_A's catastrophic
    r17 = next(r for r in rows if r.case == "17_A")
    assert r17.outcome == "CATASTROPHIC"
    assert r17.cv_status == "cluster_pattern_mismatch"
    assert not r17.cv_consistent
    assert r17.category == "CHIRALITY_MISS"
    assert r17.phase_sep == 18.1


def test_build_matrix_handles_missing_cv_status():
    """If a case is in post_218 but not in cv_local, cv_status='missing'."""
    post = _stub_post_218([("99_A", 0, "GOOD", 3.0)])
    cv = _stub_cv_local({})  # empty
    rows = p2b.build_matrix(post, cv)
    assert len(rows) == 1
    assert rows[0].cv_status == "missing"
    assert rows[0].cv_consistent is False


def test_evaluate_rule_counts_correctly():
    # Construct a synthetic mini-corpus: 2 GOOD, 2 CATASTROPHIC, 1 MARGINAL.
    rows = [
        p2b.TrustRow(case="a", run=0, outcome="GOOD", category="GOOD",
                     phase_sep=2.0, phase_check="?", cv_status="ok", cv_consistent=True),
        p2b.TrustRow(case="b", run=0, outcome="GOOD", category="GOOD",
                     phase_sep=15.0, phase_check="?", cv_status="ok", cv_consistent=True),
        p2b.TrustRow(case="c", run=0, outcome="CATASTROPHIC", category="CHIRALITY_MISS",
                     phase_sep=20.0, phase_check="?", cv_status="cluster_pattern_mismatch",
                     cv_consistent=False),
        p2b.TrustRow(case="d", run=0, outcome="CATASTROPHIC", category="TRUE_GEOMETRY_FAIL",
                     phase_sep=5.0, phase_check="?", cv_status="cluster_pattern_mismatch",
                     cv_consistent=False),
        p2b.TrustRow(case="e", run=0, outcome="MARGINAL", category="MARGINAL",
                     phase_sep=8.0, phase_check="?", cv_status="ok", cv_consistent=True),
    ]

    # Rule: phase_sep >= 10 → retake.
    # Catches case b (GOOD, false) and case c (CATASTROPHIC, true).
    # Misses case d (CATASTROPHIC, phase_sep=5).
    result = p2b.evaluate_rule(
        rows,
        lambda r: r.phase_sep is not None and r.phase_sep >= 10,
        "phase_ge_10",
        "phase >= 10",
    )
    assert result.catastrophic_caught == 1
    assert result.catastrophic_total == 2
    assert result.catastrophic_recall == 0.5
    assert result.good_retaken == 1
    assert result.good_total == 2
    assert result.good_false_retake_rate == 0.5
    assert result.marginal_retaken == 0
    assert result.marginal_total == 1
    # 50% recall is below the 80% bar; 50% FPR is above the 10% bar — fail
    assert not result.meets_bar


def test_evaluate_rule_meets_bar_when_perfect():
    rows = [
        p2b.TrustRow(case="g", run=0, outcome="GOOD", category="GOOD",
                     phase_sep=2.0, phase_check="?", cv_status="ok", cv_consistent=True),
        p2b.TrustRow(case="c1", run=0, outcome="CATASTROPHIC", category="CHIRALITY_MISS",
                     phase_sep=99.0, phase_check="?", cv_status="bad", cv_consistent=False),
        p2b.TrustRow(case="c2", run=0, outcome="CATASTROPHIC", category="CHIRALITY_MISS",
                     phase_sep=99.0, phase_check="?", cv_status="bad", cv_consistent=False),
    ]
    # Perfect predicate: retake catastrophic, leave GOOD alone.
    result = p2b.evaluate_rule(
        rows,
        lambda r: r.outcome == "CATASTROPHIC",
        "oracle",
        "oracle predicate (won't work in production, only here as a sanity check)",
    )
    assert result.catastrophic_recall == 1.0
    assert result.good_false_retake_rate == 0.0
    assert result.meets_bar


def test_candidate_rules_covers_expected_compositions():
    """Smoke check: the rule list includes solo + OR + AND combinations
    over the available signals."""
    rules = p2b.candidate_rules()
    names = [name for name, _, _ in rules]
    # Must include phase-solo, cv-local-solo, and at least one each of
    # phase_or_cv and phase_and_cv composites.
    assert any("phase_sep_alone" in n for n in names)
    assert "cv_local_alone" in names
    assert any("phase_or_cv" in n for n in names)
    assert any("phase_and_cv" in n for n in names)
    assert "cv_severe_alone" in names


def test_summarize_counts_outcomes_and_cases():
    rows = [
        p2b.TrustRow(case="x", run=0, outcome="GOOD", category="GOOD",
                     phase_sep=1.0, phase_check="?", cv_status="ok", cv_consistent=True),
        p2b.TrustRow(case="x", run=1, outcome="GOOD", category="GOOD",
                     phase_sep=1.0, phase_check="?", cv_status="ok", cv_consistent=True),
        p2b.TrustRow(case="y", run=0, outcome="CATASTROPHIC", category="CHIRALITY_MISS",
                     phase_sep=20.0, phase_check="?", cv_status="bad", cv_consistent=False),
        p2b.TrustRow(case="z", run=0, outcome="MARGINAL", category="MARGINAL",
                     phase_sep=5.0, phase_check="?", cv_status="ok", cv_consistent=True),
    ]
    s = p2b.summarize(rows)
    assert s["n_rows"] == 4
    assert s["n_cases"] == 3  # x is counted once even with 2 runs
    assert s["n_good"] == 2
    assert s["n_catastrophic"] == 1
    assert s["n_marginal"] == 1


def test_full_run_against_committed_fixtures_produces_expected_shape():
    """Smoke test: build the matrix from the actual committed fixtures and
    confirm the shape matches what the report consumes. This locks in the
    expected outcome counts against the post_218 baseline (76 GOOD, 16
    MARGINAL, 24 CATASTROPHIC; 116 runs across 58 cases)."""
    import json
    post = json.loads(p2b.POST_218_PATH.read_text())
    cv = json.loads(p2b.CV_LOCAL_PATH.read_text())
    rows = p2b.build_matrix(post, cv)
    s = p2b.summarize(rows)
    assert s["n_rows"] == 116
    assert s["n_cases"] == 58
    assert s["n_good"] == 76
    assert s["n_marginal"] == 16
    assert s["n_catastrophic"] == 24
    # Every row should have a known outcome (no UNKNOWN leakage).
    assert all(r.outcome in {"GOOD", "MARGINAL", "CATASTROPHIC"} for r in rows)
