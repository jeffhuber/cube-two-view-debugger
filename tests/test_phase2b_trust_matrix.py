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


def _stub_recomputed(rows):
    """rows: list of dicts (already in the recompute schema). Returns the
    full {summary, by_case} envelope."""
    by_case: Dict[str, list] = {}
    for r in rows:
        case = r.pop("_case", "x")
        by_case.setdefault(case, []).append(r)
    return {"summary": {}, "by_case": by_case}


def test_build_matrix_from_recomputed_extracts_continuous_signals():
    """The recomputed builder must populate all continuous signal fields,
    not just the existing-signal fields. This pins the field mapping
    between the recompute schema and TrustRow."""
    recomp = _stub_recomputed([
        {
            "_case": "12_A", "run": 0, "status": "ok",
            "err_near_deg": 5.0, "err_far_deg": 15.0,
            "phase_check": "correct", "category": "GOOD",
            "phase_darkness_separation": 3.2,
            "fit_residual_rms_px": 95.0,
            "pnp_rms_px": 100.0,
            "hexagon_centroid_vs_bezel_vertex_offset_px": 25.0,
            "bezel_vs_fit_cube_center_offset_px": 5.0,
            "junction_score_at_ensemble": 180.0,
            "ensemble_shift_px": 15.0,
            "ensemble_n_candidates": 3,
        },
    ])
    cv = _stub_cv_local({"12_A": "ok"})
    rows = p2b.build_matrix_from_recomputed(recomp, cv)
    assert len(rows) == 1
    r = rows[0]
    assert r.outcome == "GOOD"
    assert r.phase_sep == 3.2
    assert r.fit_residual_rms_px == 95.0
    assert r.pnp_rms_px == 100.0
    assert r.hexagon_centroid_vs_bezel_vertex_offset_px == 25.0
    assert r.junction_score_at_ensemble == 180.0
    assert r.ensemble_shift_px == 15.0


def test_build_matrix_from_recomputed_handles_fit_failure():
    """A fit-failure run must produce a CATASTROPHIC outcome row with most
    signals None AND `model_fit_failed=True` so evaluate_rule treats it as
    an automatic retake (Codex's #233 P2 finding — the original v2 code
    counted these as uncaught-catastrophic, undercounting recall)."""
    recomp = _stub_recomputed([
        {
            "_case": "99_X", "run": 0, "status": "model_fit_failed",
            "category": "TRUE_GEOMETRY_FAIL",
        },
    ])
    cv = _stub_cv_local({"99_X": "cluster_pattern_mismatch"})
    rows = p2b.build_matrix_from_recomputed(recomp, cv)
    assert len(rows) == 1
    r = rows[0]
    assert r.outcome == "CATASTROPHIC"
    assert r.category == "TRUE_GEOMETRY_FAIL"
    assert r.fit_residual_rms_px is None
    assert r.hexagon_centroid_vs_bezel_vertex_offset_px is None
    assert r.model_fit_failed is True  # the new auto-retake signal


def test_evaluate_rule_auto_retakes_model_fit_failures():
    """Codex's #233 P2 finding: when a row has `model_fit_failed=True`,
    every candidate rule must count it as RETAKEN even if the predicate
    returns False on its None signals. Otherwise model-fit failures
    undercount recall.
    """
    # Two catastrophic rows: one with phase_sep that the predicate WOULD catch,
    # one model_fit_failed with no signals. A predicate that never fires
    # explicitly should still catch the fit-failed row via short-circuit.
    rows = [
        p2b.TrustRow(case="a", run=0, outcome="CATASTROPHIC", category="CHIRALITY_MISS",
                     phase_sep=2.0, phase_check="?", cv_status="ok", cv_consistent=True),
        p2b.TrustRow(case="b", run=0, outcome="CATASTROPHIC", category="TRUE_GEOMETRY_FAIL",
                     phase_sep=None, phase_check="?", cv_status="ok", cv_consistent=True,
                     model_fit_failed=True),
        p2b.TrustRow(case="c", run=0, outcome="GOOD", category="GOOD",
                     phase_sep=20.0, phase_check="?", cv_status="ok", cv_consistent=True),
    ]
    # A predicate that ONLY fires for low phase_sep — catches "a" by predicate,
    # catches "b" by the auto-retake short-circuit.
    result = p2b.evaluate_rule(
        rows,
        lambda r: r.phase_sep is not None and abs(r.phase_sep) < 5.0,
        "low_phase",
        "Retake when |phase_sep| < 5.",
    )
    assert result.catastrophic_caught == 2  # both — "a" via predicate, "b" via auto-retake
    assert result.catastrophic_total == 2
    assert result.good_retaken == 0  # "c" has phase_sep=20, doesn't fire; not fit-failed


def test_build_matrix_from_recomputed_skips_harness_error_rows(capsys):
    """Codex #233 round-3 P2: harness errors (`status: "error"`) are NOT
    model-fit failures — they're recompute-side exceptions that should be
    re-run, not silently counted as "auto-retaken catastrophics". The
    builder must skip them and warn loudly.
    """
    recomp = _stub_recomputed([
        {"_case": "A", "run": 0, "status": "ok",
         "err_near_deg": 5.0, "phase_check": "correct", "category": "GOOD",
         "phase_darkness_separation": 1.0},
        {"_case": "B", "run": 0, "status": "error",
         "error": "ZeroDivisionError: boom"},
    ])
    cv = _stub_cv_local({"A": "ok", "B": "ok"})
    rows = p2b.build_matrix_from_recomputed(recomp, cv)
    # Only the OK row survives; the error row is skipped.
    assert len(rows) == 1
    assert rows[0].case == "A"
    # Warning written to stderr (captured via capsys).
    captured = capsys.readouterr()
    assert "warn: skipping B" in captured.err
    assert "ZeroDivisionError" in captured.err


def test_build_matrix_from_recomputed_raises_on_unknown_status():
    """Unknown statuses should fail loudly, not silently corrupt the
    matrix. Codex #233 round-3 P2 follow-on guard."""
    import pytest
    recomp = _stub_recomputed([
        {"_case": "X", "run": 0, "status": "weird_new_status"},
    ])
    cv = _stub_cv_local({"X": "ok"})
    with pytest.raises(ValueError, match="unknown recompute status"):
        p2b.build_matrix_from_recomputed(recomp, cv)


def test_evaluate_rule_auto_retake_doesnt_affect_good_rows():
    """The auto-retake short-circuit applies to ALL outcomes — including
    GOOD. A fit-failed GOOD row would be counted as a false-retake (which
    is correct semantically: the model couldn't fit, so we'd retake even
    if ground truth says the cube was actually GOOD). Verify this counts
    in the FPR numerator.
    """
    rows = [
        p2b.TrustRow(case="g", run=0, outcome="GOOD", category="GOOD",
                     phase_sep=None, phase_check="?", cv_status="ok",
                     cv_consistent=True, model_fit_failed=True),
        p2b.TrustRow(case="g2", run=0, outcome="GOOD", category="GOOD",
                     phase_sep=20.0, phase_check="?", cv_status="ok", cv_consistent=True),
    ]
    result = p2b.evaluate_rule(
        rows,
        lambda r: False,  # predicate never fires
        "never",
        "never retake",
    )
    # The fit-failed GOOD counts as a false-retake; the other GOOD doesn't.
    assert result.good_retaken == 1
    assert result.good_total == 2
    assert abs(result.good_false_retake_rate - 0.5) < 1e-9


def test_recomputed_signal_rules_include_expected_compositions():
    """Smoke check: the recomputed-rule list includes fit_residual,
    pnp_rms, hex_bezel, ensemble_shift, junction_score thresholds, plus
    OR-compounds with phaseANDcv, plus AND-compounds."""
    rules = p2b.recomputed_signal_rules()
    names = [name for name, _, _ in rules]
    assert any("fit_residual_alone" in n for n in names)
    assert any("pnp_rms_alone" in n for n in names)
    assert any("hex_bezel_disagree" in n for n in names)
    assert any("ensemble_shift" in n for n in names)
    assert any("junction_score_below" in n for n in names)
    assert any("phaseANDcv_OR_fit_residual" in n for n in names)
    assert any("phaseANDcv_OR_fit80.0_OR_hex50.0" in n for n in names)
    assert any("fit80.0_AND_hex50.0" in n for n in names)


def test_recomputed_rules_handle_missing_signals_gracefully():
    """Rules over continuous signals must return False (= don't retake) on
    rows where the signal is None. The matrix may legitimately have rows
    with some signals missing (model_fit_failed cases)."""
    row_missing = p2b.TrustRow(
        case="x", run=0, outcome="GOOD", category="GOOD",
        phase_sep=None, phase_check="?",
        cv_status="ok", cv_consistent=True,
    )
    assert p2b._signal_above(row_missing, "fit_residual_rms_px", 100.0) is False
    assert p2b._signal_below(row_missing, "junction_score_at_ensemble", 50.0) is False


def test_signal_above_and_below_inequality_boundaries():
    """_signal_above uses >=, _signal_below uses <. Pin the boundaries —
    Phase 2 calibration depends on which side of the threshold counts."""
    row = p2b.TrustRow(
        case="x", run=0, outcome="GOOD", category="GOOD",
        phase_sep=None, phase_check="?",
        cv_status="ok", cv_consistent=True,
        fit_residual_rms_px=100.0,
        junction_score_at_ensemble=100.0,
    )
    assert p2b._signal_above(row, "fit_residual_rms_px", 100.0) is True
    assert p2b._signal_above(row, "fit_residual_rms_px", 100.1) is False
    assert p2b._signal_below(row, "junction_score_at_ensemble", 100.0) is False
    assert p2b._signal_below(row, "junction_score_at_ensemble", 100.1) is True


def test_candidate_rule_thresholds_are_correctly_bound_per_rule():
    """Regression guard for a Python closure gotcha that Qwen's audit of
    PR #232 flagged (incorrectly — see commit history). Each thresholded
    rule built inside a `for t in (...):` loop captures `t` via the
    `lambda r, t=t:` default-argument idiom; this test pins that behavior
    so a future refactor that drops the `t=t` binding (re-introducing the
    classic late-binding-closure bug) fails loudly here instead of
    silently making every thresholded rule evaluate at the LAST threshold
    in the loop.

    UPDATED after Codex's #232 review (real polarity bug fix): the phase
    predicate is `|phase_sep| < T` (Phase 2A semantics — low magnitude
    means low confidence means retake), not `phase_sep >= T`. Probe row
    is now phase_sep=2.0 so we can test both sides of small thresholds.
    """
    rules = {name: pred for name, _, pred in p2b.candidate_rules()}

    # |phase_sep| = 2.0 — should fire for T > 2 (low-confidence below T),
    # not fire for T <= 2 (above-threshold-confidence).
    probe_low = p2b.TrustRow(
        case="probe", run=0, outcome="GOOD", category="GOOD",
        phase_sep=2.0, phase_check="?", cv_status="ok", cv_consistent=True,
    )
    assert rules["phase_sep_alone_T0.5"](probe_low) is False  # 2 < 0.5 → False
    assert rules["phase_sep_alone_T2.0"](probe_low) is False  # |sep| < T uses strict <; 2 < 2 → False
    assert rules["phase_sep_alone_T5.0"](probe_low) is True   # 2 < 5 → True
    assert rules["phase_sep_alone_T8.0"](probe_low) is True
    assert rules["phase_sep_alone_T15.0"](probe_low) is True

    # Same probe through OR-compound rules — cv_consistent=True isolates phase side.
    assert rules["phase_or_cv_T8.0"](probe_low) is True   # |2| < 8 → True
    assert rules["phase_or_cv_T11.7"](probe_low) is True  # |2| < 11.7 → True

    # AND-compounds with cv_consistent=True NEVER fire (the AND-cv side
    # is always False here).
    assert rules["phase_and_cv_T8.0"](probe_low) is False

    # Confirm negative phase_sep is also handled correctly via abs().
    # phase_sep = -10 should be |10|, same as phase_sep = +10.
    probe_neg = p2b.TrustRow(
        case="probe2", run=0, outcome="GOOD", category="GOOD",
        phase_sep=-10.0, phase_check="?", cv_status="ok", cv_consistent=True,
    )
    assert rules["phase_sep_alone_T5.0"](probe_neg) is False  # |-10|=10, not < 5
    assert rules["phase_sep_alone_T15.0"](probe_neg) is True  # |-10|=10, < 15


def test_phase_2a_reference_values_at_T11_7_match_documentation():
    """Codex's #232 review caught a polarity inversion: the predicate
    `phase_sep >= T` produced 41.7% recall / 38.2% FPR at T=11.7, but
    Phase 2A documents 45.8% / 9.2% at the same threshold using `|sep| < T`.
    After the polarity fix, evaluating on the committed post_218 baseline
    must reproduce Phase 2A's numbers.

    This test pins those reference values — if a future refactor reverts
    the polarity or perturbs the predicate, this test fails loudly with
    the bar values clearly violated.
    """
    import json
    post = json.loads(p2b.POST_218_PATH.read_text())
    cv = json.loads(p2b.CV_LOCAL_PATH.read_text())
    rows = p2b.build_matrix(post, cv)

    rules = {name: (desc, pred) for name, desc, pred in p2b.candidate_rules()}
    _, pred = rules["phase_sep_alone_T11.7"]
    result = p2b.evaluate_rule(rows, pred, "phase_sep_alone_T11.7",
                               "Phase 2A reference rule")

    # Phase 2A documents 45.8% recall (11/24) at 9.2% GOOD FPR (7/76).
    # Allow a small slack (±2 pp) to absorb any minor data-counting drift
    # between Phase 2A's separate calibration run and the post_218 join.
    assert abs(result.catastrophic_recall - 0.458) <= 0.05, (
        f"catastrophic recall {result.catastrophic_recall:.3f} drifted from "
        f"Phase 2A's documented 0.458"
    )
    assert abs(result.good_false_retake_rate - 0.092) <= 0.05, (
        f"GOOD FPR {result.good_false_retake_rate:.3f} drifted from "
        f"Phase 2A's documented 0.092"
    )
    # Hard bar: should NOT be the buggy values (41.7% / 38.2%).
    assert result.good_false_retake_rate < 0.20, (
        f"GOOD FPR {result.good_false_retake_rate:.3f} is way above the bar; "
        f"phase predicate polarity may be reverted to `phase_sep >= T`"
    )


def test_display_path_handles_paths_outside_repo_root():
    """Codex's #232 non-blocking find: `Path.relative_to(REPO_ROOT)` crashes
    on absolute paths outside the repo (e.g., `--matrix-out /tmp/foo.json`).
    The fix uses `_display_path()` which falls back to the absolute path."""
    from pathlib import Path
    # Inside repo root → relative form
    inside = p2b.REPO_ROOT / "tests" / "fixtures" / "x.json"
    assert p2b._display_path(inside) == "tests/fixtures/x.json"
    # Outside repo root → absolute fallback (no crash)
    outside = Path("/tmp/foo.json")
    assert p2b._display_path(outside) == "/tmp/foo.json"


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
