#!/usr/bin/env python3
"""Phase 2B: trust-signal matrix.

Joins per-case / per-run signals from existing baselines, evaluates candidate
trust rules against the Phase 2 bar:

  - catastrophic_recall  >= 0.80
  - good_false_retake    <= 0.10

Diagnostics-only. No production behavior change. Emits:

  tests/fixtures/phase2b_trust_signal_matrix.json   — per-run signal+outcome rows
  tools/PHASE_2B_TRUST_SIGNAL_MATRIX.md             — human-readable report

Per Codex's spec, joins these signals per labeled case/run:

  outcome      — GOOD / MARGINAL / CATASTROPHIC (from post_218_baseline)
  phase_sep    — Phase 2A's signal (continuous; per-run from post_218)
  cv_status    — cv-local face-quad structural status (per-case from
                 cv_local_baseline; one of ok / cluster_pattern_mismatch /
                 fewer_than_3_face_quads)
  cv_consistent — derived bool (cv_status == "ok")

Future extensions gated behind `--recompute-global-model`:

  fit_residual_rms_px        — global model affine/PnP residual
  vertex_ensemble_stddev_px  — disagreement across the 3-vertex ensemble
  two_view_consistency       — pair-wise A↔B orientation agreement per set

Phase 2A already established phase_sep alone hits only 45.8% catastrophic
recall at the 9.2% GOOD false-retake operating point — below the bar.
Phase 2B's job is to find a multi-signal rule that DOES meet the bar, or
produce clean evidence that none of these signals (alone or combined) are
sufficient — informing a pivot to learned geometry / capture UX.

Usage:

    python3 tools/phase2b_trust_matrix.py
    python3 tools/phase2b_trust_matrix.py --recompute-global-model  # future
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
POST_218_PATH = REPO_ROOT / "tests" / "fixtures" / "post_218_baseline.json"
CV_LOCAL_PATH = REPO_ROOT / "tests" / "fixtures" / "cv_local_baseline.json"
RECOMPUTED_PATH = REPO_ROOT / "tests" / "fixtures" / "phase2b_recomputed_signals.json"
MATRIX_OUT_PATH = REPO_ROOT / "tests" / "fixtures" / "phase2b_trust_signal_matrix.json"
RECOMPUTED_MATRIX_OUT_PATH = REPO_ROOT / "tests" / "fixtures" / "phase2b_trust_signal_matrix_recomputed.json"
REPORT_OUT_PATH = REPO_ROOT / "tools" / "PHASE_2B_TRUST_SIGNAL_MATRIX.md"
RECOMPUTED_REPORT_OUT_PATH = REPO_ROOT / "tools" / "PHASE_2B_TRUST_SIGNAL_MATRIX_RECOMPUTED.md"

# Phase 2 success criteria (per CLAUDE.md / Codex / Phase 2A doc)
CATASTROPHIC_RECALL_BAR = 0.80
GOOD_FALSE_RETAKE_BAR = 0.10

# Phase 2A's chosen operating point on phase_sep alone
PHASE_2A_THRESHOLD = 11.7
PHASE_2A_REPORTED_RECALL = 0.458
PHASE_2A_REPORTED_FPR = 0.092


def _display_path(p: Path) -> str:
    """Render `p` as a repo-relative string when possible, falling back to
    the absolute path for arbitrary `--matrix-out /tmp/foo.json` invocations
    (Codex caught the `relative_to()` crash in #232 review)."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


@dataclass
class TrustRow:
    """One row of the trust matrix — per case/run, joined signals + outcome."""
    case: str
    run: int
    outcome: str  # "GOOD" | "MARGINAL" | "CATASTROPHIC"
    category: str  # raw post_218 category (e.g. "CHIRALITY_MISS")
    phase_sep: Optional[float]
    phase_check: str
    cv_status: str
    cv_consistent: bool
    # Continuous signals from `tools/phase2b_recompute.py` (populated only
    # when --recompute-global-model is set; None for join-only matrix).
    fit_residual_rms_px: Optional[float] = None
    pnp_rms_px: Optional[float] = None
    hexagon_centroid_vs_bezel_vertex_offset_px: Optional[float] = None
    bezel_vs_fit_cube_center_offset_px: Optional[float] = None
    junction_score_at_ensemble: Optional[float] = None
    ensemble_shift_px: Optional[float] = None
    ensemble_n_candidates: Optional[float] = None
    # Reserved for a future iteration that pairs A↔B views per set.
    two_view_consistency_deg: Optional[float] = None
    # True when the global model couldn't fit this run at all (status was
    # "model_fit_failed" or some other harness error). Codex caught the
    # original v2 behavior treating these as "uncaught catastrophic"
    # because every continuous-signal predicate returns False on None.
    # The correct semantics: a model-fit failure is an AUTOMATIC retake,
    # not something a trust rule needs to "catch". `evaluate_rule` short-
    # circuits this flag to True regardless of predicate.
    model_fit_failed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case,
            "run": self.run,
            "outcome": self.outcome,
            "category": self.category,
            "phase_sep": self.phase_sep,
            "phase_check": self.phase_check,
            "cv_status": self.cv_status,
            "cv_consistent": self.cv_consistent,
            "fit_residual_rms_px": self.fit_residual_rms_px,
            "pnp_rms_px": self.pnp_rms_px,
            "hexagon_centroid_vs_bezel_vertex_offset_px": self.hexagon_centroid_vs_bezel_vertex_offset_px,
            "bezel_vs_fit_cube_center_offset_px": self.bezel_vs_fit_cube_center_offset_px,
            "junction_score_at_ensemble": self.junction_score_at_ensemble,
            "ensemble_shift_px": self.ensemble_shift_px,
            "ensemble_n_candidates": self.ensemble_n_candidates,
            "two_view_consistency_deg": self.two_view_consistency_deg,
            "model_fit_failed": self.model_fit_failed,
        }


@dataclass
class RuleEvalResult:
    name: str
    description: str
    catastrophic_caught: int
    catastrophic_total: int
    good_retaken: int
    good_total: int
    marginal_retaken: int
    marginal_total: int

    @property
    def catastrophic_recall(self) -> float:
        return self.catastrophic_caught / self.catastrophic_total if self.catastrophic_total else 0.0

    @property
    def good_false_retake_rate(self) -> float:
        return self.good_retaken / self.good_total if self.good_total else 0.0

    @property
    def marginal_routed_rate(self) -> float:
        return self.marginal_retaken / self.marginal_total if self.marginal_total else 0.0

    @property
    def meets_bar(self) -> bool:
        return (
            self.catastrophic_recall >= CATASTROPHIC_RECALL_BAR
            and self.good_false_retake_rate <= GOOD_FALSE_RETAKE_BAR
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "catastrophic_recall": round(self.catastrophic_recall, 3),
            "good_false_retake_rate": round(self.good_false_retake_rate, 3),
            "marginal_routed_rate": round(self.marginal_routed_rate, 3),
            "catastrophic_caught": self.catastrophic_caught,
            "catastrophic_total": self.catastrophic_total,
            "good_retaken": self.good_retaken,
            "good_total": self.good_total,
            "marginal_retaken": self.marginal_retaken,
            "marginal_total": self.marginal_total,
            "meets_bar": self.meets_bar,
        }


# ----- Outcome / category mapping -----


CATASTROPHIC_CATEGORIES = {"CHIRALITY_MISS", "CHIRALITY_FALSE_FLIP", "TRUE_GEOMETRY_FAIL"}


def categorize_outcome(category: str) -> str:
    if category == "GOOD":
        return "GOOD"
    if category == "MARGINAL":
        return "MARGINAL"
    if category in CATASTROPHIC_CATEGORIES:
        return "CATASTROPHIC"
    return "UNKNOWN"


# ----- Join -----


def build_matrix(post_218: Dict[str, Any], cv_local: Dict[str, Any]) -> List[TrustRow]:
    """Join post_218 (per-run) and cv_local (per-case) into per-run trust rows."""
    # cv-local has one entry per case (status is per-case, not per-run).
    cv_status: Dict[str, str] = {}
    for case, runs in cv_local["by_case"].items():
        # All runs for a case share the cv-local status; first one is fine.
        cv_status[case] = runs[0].get("status", "?") if runs else "?"

    rows: List[TrustRow] = []
    for case, runs in post_218["by_case"].items():
        cvs = cv_status.get(case, "missing")
        cv_ok = (cvs == "ok")
        for r in runs:
            category = r.get("category", "?")
            rows.append(TrustRow(
                case=case,
                run=r["run"],
                outcome=categorize_outcome(category),
                category=category,
                phase_sep=r.get("phase_sep"),
                phase_check=r.get("phase_check", "?"),
                cv_status=cvs,
                cv_consistent=cv_ok,
            ))
    return rows


def build_matrix_from_recomputed(
    recomputed: Dict[str, Any],
    cv_local: Dict[str, Any],
) -> List[TrustRow]:
    """Join recomputed global-model signals (per-run, from phase2b_recompute)
    with cv_local (per-case). Populates the full TrustRow including all
    continuous signals. Records whose status != "ok" (e.g., model_fit_failed)
    become CATASTROPHIC rows with mostly-None signals — the verdict logic
    naturally treats this as 'retake'."""
    cv_status: Dict[str, str] = {}
    for case, runs in cv_local["by_case"].items():
        cv_status[case] = runs[0].get("status", "?") if runs else "?"

    rows: List[TrustRow] = []
    for case, runs in recomputed["by_case"].items():
        cvs = cv_status.get(case, "missing")
        cv_ok = (cvs == "ok")
        for r in runs:
            status = r.get("status")
            if status == "model_fit_failed":
                # Real model-fit failure: the global model couldn't fit at
                # all. Auto-retake regardless of any signal predicate.
                rows.append(TrustRow(
                    case=case,
                    run=r.get("run", -1),
                    outcome="CATASTROPHIC",
                    category=r.get("category", "TRUE_GEOMETRY_FAIL"),
                    phase_sep=None,
                    phase_check="?",
                    cv_status=cvs,
                    cv_consistent=cv_ok,
                    model_fit_failed=True,
                ))
                continue
            if status == "error":
                # Codex #233 round-3 P2: a harness exception during recompute
                # (corrupt image, transient dep error, etc.) is NOT a model-
                # fit failure — we never got far enough to know if the model
                # could fit. Conflating the two would silently distort the
                # benchmark (every harness error would count as a "perfectly-
                # caught catastrophic" via the auto-retake short-circuit, and
                # inflate recall). Skip these rows and warn loudly so the
                # user diagnoses the recompute before trusting the matrix.
                print(
                    f"  warn: skipping {case} run {r.get('run', '?')} — "
                    f"recompute status='error', error='{r.get('error', '?')}'. "
                    f"Re-run phase2b_recompute.py to get clean data for this case.",
                    file=sys.stderr,
                )
                continue
            if status != "ok":
                # Unknown status — should not happen, but fail loudly rather
                # than silently corrupting the matrix.
                raise ValueError(
                    f"unknown recompute status for {case} run {r.get('run', '?')}: "
                    f"{status!r}. Expected 'ok', 'model_fit_failed', or 'error'."
                )
            category = r.get("category", "?")
            rows.append(TrustRow(
                case=case,
                run=r["run"],
                outcome=categorize_outcome(category),
                category=category,
                phase_sep=r.get("phase_darkness_separation"),
                phase_check=r.get("phase_check", "?"),
                cv_status=cvs,
                cv_consistent=cv_ok,
                fit_residual_rms_px=r.get("fit_residual_rms_px"),
                pnp_rms_px=r.get("pnp_rms_px"),
                hexagon_centroid_vs_bezel_vertex_offset_px=r.get(
                    "hexagon_centroid_vs_bezel_vertex_offset_px"
                ),
                bezel_vs_fit_cube_center_offset_px=r.get(
                    "bezel_vs_fit_cube_center_offset_px"
                ),
                junction_score_at_ensemble=r.get("junction_score_at_ensemble"),
                ensemble_shift_px=r.get("ensemble_shift_px"),
                ensemble_n_candidates=r.get("ensemble_n_candidates"),
            ))
    return rows


# ----- Candidate rules -----


def evaluate_rule(
    rows: List[TrustRow],
    predicate: Callable[[TrustRow], bool],
    name: str,
    description: str,
) -> RuleEvalResult:
    # `model_fit_failed` rows short-circuit to "retake = True" regardless of
    # predicate — the model couldn't fit, so we cannot trust any signal-based
    # verdict and must retake. Codex's review of #233 (P2) caught the original
    # behavior where these were counted as uncaught catastrophics, undercounting
    # recall whenever a fit failure appeared in the corpus.
    def _fires(r: TrustRow) -> bool:
        return r.model_fit_failed or predicate(r)

    catastrophic_caught = sum(1 for r in rows if r.outcome == "CATASTROPHIC" and _fires(r))
    catastrophic_total = sum(1 for r in rows if r.outcome == "CATASTROPHIC")
    good_retaken = sum(1 for r in rows if r.outcome == "GOOD" and _fires(r))
    good_total = sum(1 for r in rows if r.outcome == "GOOD")
    marginal_retaken = sum(1 for r in rows if r.outcome == "MARGINAL" and _fires(r))
    marginal_total = sum(1 for r in rows if r.outcome == "MARGINAL")
    return RuleEvalResult(
        name=name,
        description=description,
        catastrophic_caught=catastrophic_caught,
        catastrophic_total=catastrophic_total,
        good_retaken=good_retaken,
        good_total=good_total,
        marginal_retaken=marginal_retaken,
        marginal_total=marginal_total,
    )


def _phase_low_confidence(row: TrustRow, t: float) -> bool:
    """Phase 2A's retake predicate: low |phase_sep| ≡ low confidence ≡ retake.

    Reverses the polarity that PR #232 v1 had wrong (`phase_sep >= T`, which
    yielded recall=41.7% / FPR=38.2% at T=11.7 — off from Phase 2A's
    documented 45.8% / 9.2% at the same threshold). Codex caught this in
    review. See `tools/phase2a_phase_confidence_calibration.py` line 178:
    'For each threshold T, the retake decision is `|phase_sep| < T`'.

    A small absolute phase separation means the phase detector couldn't
    confidently distinguish near vs far face darkness; that's the retake
    signal. Negative phase_sep values (where `sep<0 ≡ chirality correct`
    per the empirical calibration in `NEAR_FAR_PHASE_REPORT.md`) are
    handled correctly by the `abs()`.
    """
    return row.phase_sep is not None and abs(row.phase_sep) < t


def _signal_above(row: TrustRow, attr: str, t: float) -> bool:
    """Generic 'retake if this continuous signal >= threshold'.
    Returns False for rows where the signal is missing (None) — missing
    signal == no opinion == don't retake. Model-fit failures are surfaced
    via outcome=CATASTROPHIC + the row's row.category, not via this path."""
    v = getattr(row, attr)
    return v is not None and v >= t


def _signal_below(row: TrustRow, attr: str, t: float) -> bool:
    """For signals where LOW = suspect (e.g., junction_score_at_ensemble)."""
    v = getattr(row, attr)
    return v is not None and v < t


def candidate_rules() -> List[Tuple[str, str, Callable[[TrustRow], bool]]]:
    """Return (name, description, predicate) for each candidate rule.

    Currently evaluates rules over phase_sep + cv-local status only.
    When --recompute-global-model lands, this will extend with rules over
    fit_residual_rms_px, vertex_ensemble_stddev_px, two_view_consistency_deg.

    Note on lambda thresholds: each thresholded rule below uses the
    ``lambda r, t=t: ...`` default-argument idiom to bind ``t`` at lambda
    creation time. Without ``t=t``, Python's late-binding closure
    semantics would cause every lambda to read whatever ``t`` is at the
    end of the loop, making all thresholded rules collapse to the final
    threshold. The regression test
    ``test_candidate_rule_thresholds_are_correctly_bound_per_rule``
    pins this behavior.
    """
    rules: List[Tuple[str, str, Callable[[TrustRow], bool]]] = []

    # Single-signal rules — phase predicate is `|phase_sep| < T`, NOT
    # `phase_sep >= T` (Codex caught the polarity inversion in #232 review).
    # Low |phase_sep| ≡ low phase confidence ≡ retake. See _phase_low_confidence
    # docstring + tools/phase2a_phase_confidence_calibration.py L178.
    rules.append((
        "phase_sep_alone_T11.7",
        "Retake when |phase_sep| < 11.7 (Phase 2A operating point).",
        lambda r: _phase_low_confidence(r, PHASE_2A_THRESHOLD),
    ))
    # Sweep additional |phase_sep| thresholds to characterize the curve.
    for t in (0.5, 2.0, 5.0, 8.0, 15.0, 20.0):
        rules.append((
            f"phase_sep_alone_T{t}",
            f"Retake when |phase_sep| < {t}.",
            lambda r, t=t: _phase_low_confidence(r, t),
        ))
    rules.append((
        "cv_local_alone",
        "Retake when cv-local face-quad fit is NOT structurally consistent "
        "(status != 'ok').",
        lambda r: not r.cv_consistent,
    ))

    # Compound rules — OR (catch more catastrophic, but more GOOD false-retakes)
    for t in (8.0, 11.7, 15.0):
        rules.append((
            f"phase_or_cv_T{t}",
            f"Retake when |phase_sep| < {t} OR cv-local NOT consistent.",
            lambda r, t=t: _phase_low_confidence(r, t) or not r.cv_consistent,
        ))
    # Compound rules — AND (conservative, fewer false-retakes but lower recall)
    for t in (8.0, 11.7, 15.0):
        rules.append((
            f"phase_and_cv_T{t}",
            f"Retake when |phase_sep| < {t} AND cv-local NOT consistent.",
            lambda r, t=t: _phase_low_confidence(r, t) and not r.cv_consistent,
        ))
    # cv-local subdivision: maybe certain cv-local failure modes are stronger
    # signal than others. Specifically: fewer_than_3_face_quads is a harder
    # fail than cluster_pattern_mismatch.
    rules.append((
        "cv_severe_alone",
        "Retake when cv-local status is `fewer_than_3_face_quads` "
        "(the more severe failure mode — geometry couldn't even find 3 faces).",
        lambda r: r.cv_status == "fewer_than_3_face_quads",
    ))
    for t in (8.0, 11.7, 15.0):
        rules.append((
            f"phase_or_cv_severe_T{t}",
            f"Retake when |phase_sep| < {t} OR cv-local is `fewer_than_3_face_quads`.",
            lambda r, t=t: _phase_low_confidence(r, t) or r.cv_status == "fewer_than_3_face_quads",
        ))

    return rules


def recomputed_signal_rules() -> List[Tuple[str, str, Callable[[TrustRow], bool]]]:
    """Candidate rules for the continuous signals from --recompute-global-model.
    Returned IN ADDITION TO `candidate_rules()`; the matrix-evaluation main
    concatenates both when recomputed signals are available.

    Each new continuous signal gets a threshold sweep alone, then enters
    OR-compounds with the strongest existing rule (the `|phase_sep| < 8.0
    AND cv-local NOT consistent` AND-compound). The OR compounds test
    whether layering a new signal on top of the best existing rule lifts
    recall without exploding FPR.
    """
    rules: List[Tuple[str, str, Callable[[TrustRow], bool]]] = []

    # ----- fit_residual_rms_px: HIGH = bad fit -----
    for t in (60.0, 80.0, 100.0, 120.0, 150.0, 200.0):
        rules.append((
            f"fit_residual_alone_T{t}",
            f"Retake when fit_residual_rms_px >= {t}.",
            lambda r, t=t: _signal_above(r, "fit_residual_rms_px", t),
        ))

    # ----- pnp_rms_px: HIGH = bad PnP -----
    for t in (60.0, 100.0, 150.0):
        rules.append((
            f"pnp_rms_alone_T{t}",
            f"Retake when pnp_rms_px >= {t}.",
            lambda r, t=t: _signal_above(r, "pnp_rms_px", t),
        ))

    # ----- hex↔bezel disagreement: HIGH = vertex sources disagree -----
    for t in (30.0, 50.0, 80.0, 120.0, 200.0):
        rules.append((
            f"hex_bezel_disagree_T{t}",
            f"Retake when hexagon_centroid_vs_bezel_vertex_offset_px >= {t}.",
            lambda r, t=t: _signal_above(r, "hexagon_centroid_vs_bezel_vertex_offset_px", t),
        ))

    # ----- ensemble_shift_px: HIGH = candidates disagreed, ensemble shifted PnP -----
    for t in (20.0, 40.0, 60.0, 100.0):
        rules.append((
            f"ensemble_shift_T{t}",
            f"Retake when ensemble_shift_px >= {t}.",
            lambda r, t=t: _signal_above(r, "ensemble_shift_px", t),
        ))

    # ----- junction_score_at_ensemble: LOW = weak 3-way junction, suspect vertex -----
    for t in (50.0, 100.0, 150.0, 200.0):
        rules.append((
            f"junction_score_below_T{t}",
            f"Retake when junction_score_at_ensemble < {t} (low = weak vertex).",
            lambda r, t=t: _signal_below(r, "junction_score_at_ensemble", t),
        ))

    # ----- compound: best-existing (|phase| < 8 AND cv-fail) OR new signal -----
    # Uses the polarity-corrected _phase_low_confidence predicate. See
    # `_phase_low_confidence` docstring for why |phase_sep| < T is correct
    # (Codex caught the polarity inversion in PR #232 v1).
    BEST_EXISTING = lambda r: _phase_low_confidence(r, 8.0) and not r.cv_consistent

    for sig_attr, ts, descr in [
        ("fit_residual_rms_px", (80.0, 100.0, 150.0), "fit_residual"),
        ("hexagon_centroid_vs_bezel_vertex_offset_px", (50.0, 80.0, 120.0), "hex_bezel"),
        ("ensemble_shift_px", (20.0, 40.0, 60.0), "ensemble_shift"),
    ]:
        for t in ts:
            rules.append((
                f"phaseANDcv_OR_{descr}_T{t}",
                f"Retake when (|phase|<8 AND cv-fail) OR {descr} >= {t}.",
                lambda r, attr=sig_attr, t=t: BEST_EXISTING(r) or _signal_above(r, attr, t),
            ))

    # ----- triple: best-existing OR fit_residual OR hex_bezel -----
    for ft, ht in [(80.0, 50.0), (100.0, 80.0), (150.0, 120.0)]:
        rules.append((
            f"phaseANDcv_OR_fit{ft}_OR_hex{ht}",
            f"Retake when (|phase|<8 AND cv-fail) OR fit_residual >= {ft} OR hex_bezel >= {ht}.",
            lambda r, ft=ft, ht=ht: (
                BEST_EXISTING(r)
                or _signal_above(r, "fit_residual_rms_px", ft)
                or _signal_above(r, "hexagon_centroid_vs_bezel_vertex_offset_px", ht)
            ),
        ))

    # ----- AND-compound: both must fire (most conservative — see if this
    #       precision can clear the FPR bar while keeping recall) -----
    for ft, ht in [(80.0, 50.0), (60.0, 30.0)]:
        rules.append((
            f"fit{ft}_AND_hex{ht}",
            f"Retake when fit_residual >= {ft} AND hex_bezel >= {ht}.",
            lambda r, ft=ft, ht=ht: (
                _signal_above(r, "fit_residual_rms_px", ft)
                and _signal_above(r, "hexagon_centroid_vs_bezel_vertex_offset_px", ht)
            ),
        ))

    return rules


# ----- Report generation -----


def render_markdown(
    rows: List[TrustRow],
    results: List[RuleEvalResult],
    summary: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Phase 2B: trust-signal matrix")
    lines.append("")
    lines.append("**Status: diagnostics-only.** No production behavior change. "
                 "This report evaluates candidate trust rules against the Phase 2 bar:")
    lines.append("")
    lines.append(f"- catastrophic recall ≥ **{CATASTROPHIC_RECALL_BAR:.0%}**")
    lines.append(f"- GOOD false-retake ≤ **{GOOD_FALSE_RETAKE_BAR:.0%}**")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- {summary['n_rows']} per-case-per-run rows across "
                 f"{summary['n_cases']} cases")
    lines.append(f"- Outcome breakdown: "
                 f"**{summary['n_good']} GOOD**, "
                 f"**{summary['n_marginal']} MARGINAL**, "
                 f"**{summary['n_catastrophic']} CATASTROPHIC**")
    lines.append("")
    if "recomputed" in summary.get("mode", "").lower():
        lines.append("**Source**: `tests/fixtures/phase2b_recomputed_signals.json` "
                     "(per-run global-model re-fit on the 58-case axis-labeled "
                     "gallery, capturing `fit_residual_rms_px`, `pnp_rms_px`, "
                     "`hexagon_centroid_vs_bezel_vertex_offset_px`, "
                     "`junction_score_at_ensemble`, `ensemble_shift_px`, and "
                     "`phase_darkness_separation` at native precision) joined "
                     "with `tests/fixtures/cv_local_baseline.json` (per-case "
                     "cv-local face-quad structural status). "
                     "Outcome counts differ from `post_218_baseline.json` "
                     "(74/22/20 vs 76/16/24) because the re-fit is "
                     "non-deterministic (PnP basin-of-attraction) and runs "
                     "are paired with the signals from the same fit.")
    else:
        lines.append("Joined from `tests/fixtures/post_218_baseline.json` (per-run "
                     "phase_sep + outcome category) and `tests/fixtures/"
                     "cv_local_baseline.json` (per-case cv-local face-quad "
                     "structural status).")
    lines.append("")
    lines.append("## Candidate rule evaluation")
    lines.append("")
    lines.append("| rule | description | recall | GOOD FPR | MARGINAL routed | meets bar? |")
    lines.append("|---|---|---|---|---|---|")
    for res in results:
        meets = "✅" if res.meets_bar else "❌"
        lines.append(
            f"| `{res.name}` | {res.description} | "
            f"{res.catastrophic_recall:.1%} ({res.catastrophic_caught}/{res.catastrophic_total}) | "
            f"{res.good_false_retake_rate:.1%} ({res.good_retaken}/{res.good_total}) | "
            f"{res.marginal_routed_rate:.1%} ({res.marginal_retaken}/{res.marginal_total}) | "
            f"{meets} |"
        )
    lines.append("")

    # Headline finding
    passing = [r for r in results if r.meets_bar]
    lines.append("## Headline finding")
    lines.append("")
    if passing:
        best = max(passing, key=lambda r: r.catastrophic_recall - r.good_false_retake_rate)
        lines.append(f"**A rule meeting the Phase 2 bar exists:** `{best.name}`.")
        lines.append(f"- catastrophic recall: {best.catastrophic_recall:.1%}")
        lines.append(f"- GOOD false-retake:   {best.good_false_retake_rate:.1%}")
        lines.append(f"- MARGINAL routed:     {best.marginal_routed_rate:.1%}")
        lines.append(f"- Description: {best.description}")
        lines.append("")
        lines.append("Phase 3 can wire this as a conservative guardrail.")
    else:
        if "recomputed" in summary.get("mode", "").lower():
            lines.append("**No rule over the 6 evaluated signals "
                         "(phase_sep, cv-local structural status, "
                         "fit_residual_rms_px, pnp_rms_px, "
                         "hex↔bezel disagreement, ensemble_shift_px, "
                         "junction_score_at_ensemble — alone, in OR/AND "
                         "compounds, or as triples) meets the Phase 2 bar.**")
        else:
            lines.append("**No rule over the currently-available signals "
                         "(phase_sep + cv-local structural status, alone or "
                         "combined) meets the Phase 2 bar.**")
        lines.append("")
        # Find the closest-to-bar rule by both axes
        def shortfall(r: RuleEvalResult) -> float:
            # Distance to the (1.0, 0.0) corner under the (recall, FPR) axes.
            return max(0.0, CATASTROPHIC_RECALL_BAR - r.catastrophic_recall) + \
                   max(0.0, r.good_false_retake_rate - GOOD_FALSE_RETAKE_BAR)
        closest = min(results, key=shortfall)
        lines.append(f"Closest-to-bar rule: `{closest.name}`")
        lines.append(f"- catastrophic recall: {closest.catastrophic_recall:.1%} "
                     f"(bar: {CATASTROPHIC_RECALL_BAR:.0%}; shortfall "
                     f"{max(0.0, CATASTROPHIC_RECALL_BAR - closest.catastrophic_recall):.1%})")
        lines.append(f"- GOOD false-retake:   {closest.good_false_retake_rate:.1%} "
                     f"(bar: {GOOD_FALSE_RETAKE_BAR:.0%}; excess "
                     f"{max(0.0, closest.good_false_retake_rate - GOOD_FALSE_RETAKE_BAR):.1%})")
        lines.append("")
        if "recomputed" in summary.get("mode", "").lower():
            # ----- Recomputed-mode implications -----
            lines.append("## Implications")
            lines.append("")
            lines.append("1. **One rule clears the recall bar but not FPR**: "
                         "`phase_sep_alone_T20.0` hits 80% recall but at ~30% GOOD "
                         "false-retake — way over the 10% bar. Loosening phase_sep "
                         "high enough to catch all catastrophics necessarily catches "
                         "many GOOD runs whose phase_sep happens to be small.")
            lines.append("")
            lines.append("2. **One compound clears the FPR bar but not recall**: "
                         "`phaseANDcv_OR_ensemble_shift_T60.0` is the first compound "
                         "rule to land UNDER the 10% FPR bar — at 50% recall. "
                         "Layering ensemble_shift on top of the phase+cv AND-compound "
                         "demonstrably reduces false retakes without inheriting the "
                         "noise of cv-local-solo or `hex_bezel`. This is the most "
                         "encouraging multi-signal compound yet, but recall is still "
                         "30 pp short of the bar.")
            lines.append("")
            lines.append("3. **No rule simultaneously clears both bars.** "
                         "Hand-tuned thresholds and OR/AND compounds over 6 signals "
                         "(phase_sep, cv-local, fit_residual, hex_bezel, "
                         "ensemble_shift, junction_score) cannot get past the "
                         "(≥80% recall AND ≤10% FPR) frontier on this 58-case eval.")
            lines.append("")
            lines.append("4. **fit_residual_rms_px is weaker than expected**: alone, "
                         "the best fit-residual rule is `T120.0` at 45% recall / "
                         "12.2% FPR — close to but not better than `phase_sep_T11.7`. "
                         "Fit quality and outcome-correctness correlate but the "
                         "thresholds don't separate cleanly.")
            lines.append("")
            lines.append("5. **junction_score_at_ensemble doesn't help much**: at any "
                         "threshold sweep, junction-score-based rules sit below the "
                         "phase_sep curve. The image-space junction quality at the "
                         "ensemble vertex isn't carrying enough information about "
                         "phase/chirality correctness.")
            lines.append("")
            lines.append("## Conditional pivot (per Codex) — TRIGGERED")
            lines.append("")
            lines.append("> If Phase 2B finds a rule that meets the bar, then Phase 3 "
                         "becomes straightforward: wire it as a conservative guardrail. "
                         "If it does not, we will have clean evidence to pivot to "
                         "learned geometry or capture/UX instead of hand-tuning "
                         "another scalar.")
            lines.append("")
            lines.append("**Evidence is in. Hand-tuned rules over the current signal "
                         "set don't meet the bar.** The pivot options are now "
                         "evidence-backed:")
            lines.append("")
            lines.append("- **Learned geometry / ranker (Phase 4)** — train a "
                         "logistic-regression or small-MLP retake classifier on the "
                         "6 continuous signals captured here. The Phase 2B matrix "
                         "(`tests/fixtures/phase2b_trust_signal_matrix_recomputed.json`) "
                         "is already shaped as a labeled dataset (per-row features + "
                         "outcome). Likely lifts both axes simultaneously because the "
                         "model learns the boundary in 6-D space instead of "
                         "hand-tuning a few axis-aligned cuts.")
            lines.append("")
            lines.append("- **Better capture / UX (Phase 5)** — diagnostics from "
                         "Phase 2B (especially `ensemble_shift_px` and "
                         "`hex_bezel_disagree`) can tell the user 'cube partially "
                         "occluded, retake from a different angle' instead of just "
                         "abstaining. This is the architectural lever cube-snap's "
                         "two-photo UI relies on.")
            lines.append("")
            lines.append("- **Two-view consistency (still not yet captured)** — the "
                         "matrix has a `two_view_consistency_deg` column reserved "
                         "but unpopulated; this would require fitting BOTH A and B "
                         "views per set and comparing inferred orientations. Could "
                         "be the single missing piece if A/B disagreement turns out "
                         "to be strongly correlated with phase miss.")
        else:
            # ----- Existing-signals-only implications -----
            lines.append("## Implications")
            lines.append("")
            lines.append("1. **cv-local structural consistency alone is too aggressive.** "
                         "It catches 100% of catastrophic but trips on 85% of GOOD cases too "
                         "(Phase 1 already found 90% structural-fit-fail rate; that result "
                         "weakens cv-local as a retake gate — most PRs would be retaken).")
            lines.append("")
            lines.append("2. **phase_sep alone confirms Phase 2A's ceiling.** No threshold "
                         "sweep over phase_sep clears the bar; pushing recall up to ~70% "
                         "drives GOOD FPR past 30%.")
            lines.append("")
            lines.append("3. **OR-compounds inherit cv-local's noise**; AND-compounds "
                         "inherit phase_sep's weakness. Neither composition direction with "
                         "these two signals alone clears the bar.")
            lines.append("")
            lines.append("## Next signals to add (per Codex's spec)")
            lines.append("")
            lines.append("This PR's matrix is signal-light by design — it tests whether "
                         "the cheap-to-compute existing signals suffice. They don't. To "
                         "make Phase 2B's verdict actionable, the next iteration should "
                         "extend the matrix with:")
            lines.append("")
            lines.append("- **`fit_residual_rms_px`** (continuous, per-run): the global "
                         "model's affine/PnP residual is already stored in "
                         "`model.debug['fit_residual_rms_px']`. Re-running the global "
                         "model across the 58-case axis-labeled gallery captures this.")
            lines.append("- **`vertex_ensemble_stddev_px`** (continuous, per-run): "
                         "disagreement across the 3-vertex ensemble (hexagon-PnP, "
                         "bezel-detection, image refinement). Currently aggregated into "
                         "a mean inside the global model; needs exposing per-component.")
            lines.append("- **`two_view_consistency_deg`** (continuous, per-set): "
                         "pair-wise A↔B orientation agreement. Requires running both "
                         "photos of a set through the model and comparing the inferred "
                         "axis bearings. Two-view consistency is the architectural lever "
                         "the cube-snap product UI relies on.")
            lines.append("")
            lines.append("Implementation hook: `tools/phase2b_trust_matrix.py "
                         "--recompute-global-model` is reserved for this extension. The "
                         "matrix schema is already shaped for the additional fields "
                         "(currently null).")
            lines.append("")
            lines.append("## Conditional pivot (per Codex)")
            lines.append("")
            lines.append("> If Phase 2B finds a rule that meets the bar, then Phase 3 "
                         "becomes straightforward: wire it as a conservative guardrail. "
                         "If it does not, we will have clean evidence to pivot to "
                         "learned geometry or capture/UX instead of hand-tuning another "
                         "scalar.")
            lines.append("")
            lines.append("The current matrix (existing signals only) is insufficient. "
                         "Before triggering the pivot to learned-geometry or capture-UX, "
                         "extend with `--recompute-global-model` to fold in the three "
                         "signals above. If those still don't clear the bar with any "
                         "rule composition, the pivot decision becomes evidence-backed.")
    lines.append("")
    lines.append("## See also")
    lines.append("")
    lines.append("- `tools/PHASE_2A_PHASE_CONFIDENCE_CALIBRATION.md` — solo "
                 "phase_sep calibration (this PR's starting point).")
    lines.append("- `tools/PHASE_1_CV_LOCAL_BASELINE.md` — cv-local "
                 "structural-fit baseline (this PR's second signal).")
    lines.append("- `tools/POST_218_BASELINE_AND_TAXONOMY.md` — outcome "
                 "categorization (the labels this PR predicts against).")
    lines.append("- `tools/STATE_OF_THE_WORLD.md` — phased roadmap "
                 "(Phase 0 → Phase 5).")
    lines.append("")
    return "\n".join(lines)


def summarize(rows: List[TrustRow]) -> Dict[str, Any]:
    cases = {r.case for r in rows}
    return {
        "n_rows": len(rows),
        "n_cases": len(cases),
        "n_good": sum(1 for r in rows if r.outcome == "GOOD"),
        "n_marginal": sum(1 for r in rows if r.outcome == "MARGINAL"),
        "n_catastrophic": sum(1 for r in rows if r.outcome == "CATASTROPHIC"),
    }


# ----- CLI -----


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recompute-global-model", action="store_true",
                    help="Use recomputed global-model signals (fit_residual, "
                         "hex↔bezel disagreement, junction_score, etc.) "
                         "instead of the join-only post_218 + cv_local matrix. "
                         "Requires tests/fixtures/phase2b_recomputed_signals.json "
                         "(produce via `python3 tools/phase2b_recompute.py`).")
    ap.add_argument("--matrix-out", type=Path, default=None,
                    help="Output path for the per-run matrix fixture. Default "
                         "depends on --recompute-global-model.")
    ap.add_argument("--report-out", type=Path, default=None,
                    help="Output path for the markdown report. Default depends "
                         "on --recompute-global-model.")
    args = ap.parse_args(argv)

    cv_local = json.loads(CV_LOCAL_PATH.read_text())

    if args.recompute_global_model:
        if not RECOMPUTED_PATH.exists():
            print(f"error: --recompute-global-model requires "
                  f"{_display_path(RECOMPUTED_PATH)} to exist. "
                  f"Run: python3 tools/phase2b_recompute.py", file=sys.stderr)
            return 2
        recomputed = json.loads(RECOMPUTED_PATH.read_text())
        rows = build_matrix_from_recomputed(recomputed, cv_local)
        rules = candidate_rules() + recomputed_signal_rules()
        matrix_out_path = args.matrix_out or RECOMPUTED_MATRIX_OUT_PATH
        report_out_path = args.report_out or RECOMPUTED_REPORT_OUT_PATH
        mode_label = "recomputed (full global-model signals)"
    else:
        post_218 = json.loads(POST_218_PATH.read_text())
        rows = build_matrix(post_218, cv_local)
        rules = candidate_rules()
        matrix_out_path = args.matrix_out or MATRIX_OUT_PATH
        report_out_path = args.report_out or REPORT_OUT_PATH
        mode_label = "existing signals only (join of post_218 + cv_local)"

    summary = summarize(rows)
    summary["mode"] = mode_label
    results = [evaluate_rule(rows, pred, name, desc) for name, desc, pred in rules]

    # Save matrix fixture
    matrix_out_payload = {
        "summary": summary,
        "bar": {
            "catastrophic_recall_min": CATASTROPHIC_RECALL_BAR,
            "good_false_retake_max": GOOD_FALSE_RETAKE_BAR,
        },
        "phase_2a_reference": {
            "threshold": PHASE_2A_THRESHOLD,
            "reported_catastrophic_recall": PHASE_2A_REPORTED_RECALL,
            "reported_good_false_retake_rate": PHASE_2A_REPORTED_FPR,
        },
        "rules": [r.to_dict() for r in results],
        "rows": [r.to_dict() for r in rows],
    }
    matrix_out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_out_path.write_text(json.dumps(matrix_out_payload, indent=2))
    print(f"matrix → {_display_path(matrix_out_path)} "
          f"({summary['n_rows']} rows, {len(rules)} rules evaluated, "
          f"mode: {mode_label})",
          file=sys.stderr)

    # Render markdown
    md = render_markdown(rows, results, summary)
    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    report_out_path.write_text(md)
    print(f"report → {_display_path(report_out_path)}", file=sys.stderr)

    # Brief stdout summary
    passing = [r for r in results if r.meets_bar]
    if passing:
        best = max(passing, key=lambda r: r.catastrophic_recall - r.good_false_retake_rate)
        print(f"\nbest rule meeting bar: {best.name} "
              f"(recall={best.catastrophic_recall:.1%}, "
              f"FPR={best.good_false_retake_rate:.1%})")
    else:
        print(f"\nno rule meets the bar "
              f"(catastrophic recall ≥ {CATASTROPHIC_RECALL_BAR:.0%} "
              f"AND GOOD FPR ≤ {GOOD_FALSE_RETAKE_BAR:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
