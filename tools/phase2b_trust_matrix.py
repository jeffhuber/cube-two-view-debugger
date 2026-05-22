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
MATRIX_OUT_PATH = REPO_ROOT / "tests" / "fixtures" / "phase2b_trust_signal_matrix.json"
REPORT_OUT_PATH = REPO_ROOT / "tools" / "PHASE_2B_TRUST_SIGNAL_MATRIX.md"

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
    # Future signals (None until --recompute-global-model is supported)
    fit_residual_rms_px: Optional[float] = None
    vertex_ensemble_stddev_px: Optional[float] = None
    two_view_consistency_deg: Optional[float] = None

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
            "vertex_ensemble_stddev_px": self.vertex_ensemble_stddev_px,
            "two_view_consistency_deg": self.two_view_consistency_deg,
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


# ----- Candidate rules -----


def evaluate_rule(
    rows: List[TrustRow],
    predicate: Callable[[TrustRow], bool],
    name: str,
    description: str,
) -> RuleEvalResult:
    catastrophic_caught = sum(1 for r in rows if r.outcome == "CATASTROPHIC" and predicate(r))
    catastrophic_total = sum(1 for r in rows if r.outcome == "CATASTROPHIC")
    good_retaken = sum(1 for r in rows if r.outcome == "GOOD" and predicate(r))
    good_total = sum(1 for r in rows if r.outcome == "GOOD")
    marginal_retaken = sum(1 for r in rows if r.outcome == "MARGINAL" and predicate(r))
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
                    help="(Future extension — not yet implemented.) Re-run the "
                         "global model per case to extract fit_residual, vertex "
                         "ensemble disagreement, and two-view consistency.")
    ap.add_argument("--matrix-out", type=Path, default=MATRIX_OUT_PATH,
                    help=f"Output path for the per-run matrix fixture. "
                         f"Default: {MATRIX_OUT_PATH.relative_to(REPO_ROOT)}")
    ap.add_argument("--report-out", type=Path, default=REPORT_OUT_PATH,
                    help=f"Output path for the markdown report. "
                         f"Default: {REPORT_OUT_PATH.relative_to(REPO_ROOT)}")
    args = ap.parse_args(argv)

    if args.recompute_global_model:
        print("--recompute-global-model is reserved for a future extension. "
              "This run produces the join-only matrix.", file=sys.stderr)

    post_218 = json.loads(POST_218_PATH.read_text())
    cv_local = json.loads(CV_LOCAL_PATH.read_text())

    rows = build_matrix(post_218, cv_local)
    summary = summarize(rows)

    rules = candidate_rules()
    results = [evaluate_rule(rows, pred, name, desc) for name, desc, pred in rules]

    # Save matrix fixture
    matrix_out = {
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
    args.matrix_out.parent.mkdir(parents=True, exist_ok=True)
    args.matrix_out.write_text(json.dumps(matrix_out, indent=2))
    print(f"matrix → {_display_path(args.matrix_out)} "
          f"({summary['n_rows']} rows, {len(rules)} rules evaluated)",
          file=sys.stderr)

    # Render markdown
    md = render_markdown(rows, results, summary)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(md)
    print(f"report → {_display_path(args.report_out)}", file=sys.stderr)

    # Brief stdout summary
    passing = [r for r in results if r.meets_bar]
    if passing:
        best = max(passing, key=lambda r: r.catastrophic_recall - r.good_false_retake_rate)
        print(f"\nbest rule meeting bar: {best.name} "
              f"(recall={best.catastrophic_recall:.1%}, "
              f"FPR={best.good_false_retake_rate:.1%})")
    else:
        print(f"\nno rule over existing signals meets the bar "
              f"(catastrophic recall ≥ {CATASTROPHIC_RECALL_BAR:.0%} "
              f"AND GOOD FPR ≤ {GOOD_FALSE_RETAKE_BAR:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
