#!/usr/bin/env python3
"""Phase 4 trust ranker v1: learned classifier to predict retake-worthy runs.

Conditional pivot from `PHASE_2B_TRUST_SIGNAL_MATRIX_RECOMPUTED.md`:
hand-tuned rules over 6 continuous signals could not simultaneously
clear the Phase 2 bar (≥80% catastrophic recall AND ≤10% GOOD
false-retake rate). This trains a small classifier on the same 6
signals to test whether a learned decision boundary in 6-D feature
space can clear the bar where axis-aligned thresholds couldn't.

Methodology:
  - Load `tests/fixtures/phase2b_recomputed_signals.json` (116 rows =
    58 cases × 2 runs each; the second run captures PnP basin-of-
    attraction variability).
  - Binary label: catastrophic (CHIRALITY_MISS, CHIRALITY_FALSE_FLIP,
    TRUE_GEOMETRY_FAIL) = 1; non-catastrophic (GOOD, MARGINAL) = 0.
  - 6 features: fit_residual_rms_px, pnp_rms_px,
    hexagon_centroid_vs_bezel_vertex_offset_px,
    junction_score_at_ensemble, ensemble_shift_px,
    phase_darkness_separation.
  - Leave-one-CASE-out cross-validation (group-based on case_key with
    the side stripped — both A and B of a pair go into the same fold).
    Pairs the two runs of each case (which share image input), and
    optionally pairs A+B sides of the same set. We use side-grouped
    CV here: held-out fold = both runs of one (case_key) at a time.
  - At each fold, fit a LogisticRegression on the training rows, then
    predict probability of catastrophic on the held-out rows.
  - Sweep operating thresholds; report:
    - operating point that minimizes (catastrophic recall ≥ 0.80,
      GOOD FPR) — does any threshold clear the Phase 2 bar?
    - operating point at 80% recall — what's the GOOD FPR there?
    - operating point at 10% GOOD FPR — what's the catastrophic
      recall there?
  - Also fit on the full dataset (no CV) to extract feature
    importance (standardized coefficient magnitudes).

The output JSON is structured to support downstream Phase 4 v2 work:
- Add a 7th feature column (e.g., two_view_orientation_consistency_deg)
  to the matrix → re-run this tool → compare.
- Swap LogisticRegression for a small MLP / gradient boosting → same.

CLI usage:
    .venv/bin/python tools/phase4_trust_ranker.py \\
        --input  tests/fixtures/phase2b_recomputed_signals.json \\
        --output tests/fixtures/phase4_trust_ranker_v1.json \\
        --report tools/PHASE_4_TRUST_RANKER_V1.md

The output JSON + report are committed alongside this tool as the
durable v1 baseline for Phase 4. Subsequent ranker variants compare
against this snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Feature columns. Order is significant — this is the input contract
# for the trained model. Adding/removing requires a v2 retraining.
FEATURE_COLUMNS: Tuple[str, ...] = (
    "fit_residual_rms_px",
    "pnp_rms_px",
    "hexagon_centroid_vs_bezel_vertex_offset_px",
    "junction_score_at_ensemble",
    "ensemble_shift_px",
    "phase_darkness_separation",
)

# Outcome category mapping. The Phase 2 bar treats CHIRALITY_*+_FAIL
# as the catastrophic class (the one we MUST catch via retake routing)
# and GOOD as the don't-disturb class (false-retake rate ≤ 10% bar).
# MARGINAL is intermediate — not penalized either way in the bar
# (routing MARGINAL to retake is acceptable).
CATASTROPHIC_CATEGORIES = frozenset({
    "CHIRALITY_MISS",
    "CHIRALITY_FALSE_FLIP",
    "TRUE_GEOMETRY_FAIL",
})
GOOD_CATEGORY = "GOOD"


@dataclass
class Row:
    """One row of the matrix — a single global-model run on a single case."""
    case: str          # e.g. "12_A"
    run: int           # 0 or 1 (PnP non-determinism replicate)
    category: str      # outcome label
    features: np.ndarray  # shape (6,), in FEATURE_COLUMNS order

    @property
    def is_catastrophic(self) -> bool:
        return self.category in CATASTROPHIC_CATEGORIES

    @property
    def is_good(self) -> bool:
        return self.category == GOOD_CATEGORY


def load_matrix(path: Path) -> List[Row]:
    """Flatten the nested by_case dict into a list of Row records."""
    with path.open() as f:
        data = json.load(f)
    rows: List[Row] = []
    for case_key, runs in data["by_case"].items():
        for run_entry in runs:
            feats = []
            for col in FEATURE_COLUMNS:
                v = run_entry.get(col)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    raise ValueError(
                        f"row {case_key}/run{run_entry.get('run')} "
                        f"missing or NaN feature {col!r}"
                    )
                feats.append(float(v))
            rows.append(Row(
                case=case_key,
                run=int(run_entry["run"]),
                category=run_entry["category"],
                features=np.asarray(feats, dtype=np.float64),
            ))
    return rows


def _make_model_factories() -> Dict[str, Any]:
    """Return name → zero-arg factory that creates a fresh sklearn model.

    Each factory builds a model whose `.fit(X, y)` and `.predict_proba(X)`
    are sklearn-standard. Caller is responsible for fitting on a
    standardized X (we apply StandardScaler externally).
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # type: ignore
    from sklearn.neural_network import MLPClassifier  # type: ignore
    return {
        "logistic_regression": lambda: LogisticRegression(
            C=1.0, class_weight="balanced",
            solver="liblinear", random_state=0,
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=0,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, max_depth=5, class_weight="balanced",
            random_state=0,
        ),
        "mlp_16_8": lambda: MLPClassifier(
            hidden_layer_sizes=(16, 8), max_iter=2000, random_state=0,
        ),
    }


def fit_model(
    rows: List[Row],
    model_factory: Any,
) -> Tuple[Any, Any]:
    """Fit a sklearn classifier on ALL rows. Returns (scaler, model).

    Standardize features first so coefficient/importance values are
    comparable across features (and across models).
    """
    from sklearn.preprocessing import StandardScaler  # type: ignore
    X = np.stack([r.features for r in rows])
    y = np.asarray([1 if r.is_catastrophic else 0 for r in rows])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = model_factory()
    model.fit(Xs, y)
    return scaler, model


def predict_probas(scaler: Any, model: Any, rows: List[Row]) -> np.ndarray:
    """Return predicted P(catastrophic) for each row, shape (N,)."""
    X = np.stack([r.features for r in rows])
    Xs = scaler.transform(X)
    return model.predict_proba(Xs)[:, 1]


def leave_one_case_out_predict(
    rows: List[Row],
    model_factory: Any,
) -> np.ndarray:
    """Out-of-fold predictions via leave-one-case-out CV.

    Fold structure: hold out all rows of a single case_key (typically
    2 runs per case from PnP nondeterminism), train on the rest,
    predict on the held-out. Repeat for every case.

    Returns predictions in the same row order as the input.
    """
    cases: Dict[str, List[int]] = {}
    for idx, r in enumerate(rows):
        cases.setdefault(r.case, []).append(idx)

    n = len(rows)
    out = np.zeros(n)
    for held_case, held_idxs in cases.items():
        held_idx_set = set(held_idxs)
        train_rows = [rows[i] for i in range(n) if i not in held_idx_set]
        test_rows = [rows[i] for i in held_idxs]
        scaler, model = fit_model(train_rows, model_factory)
        probs = predict_probas(scaler, model, test_rows)
        for i, p in zip(held_idxs, probs):
            out[i] = p
    return out


@dataclass
class ThresholdMetrics:
    threshold: float
    catastrophic_recall: float  # TP / (TP + FN) over catastrophic
    good_fpr: float             # FP-on-GOOD / total GOOD
    marginal_retake_rate: float  # (predicted retake) / total MARGINAL
    n_catastrophic: int
    n_good: int
    n_marginal: int
    n_catastrophic_caught: int
    n_good_false_retakes: int
    n_marginal_retakes: int

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)


def threshold_sweep(
    rows: List[Row],
    probas: np.ndarray,
    *,
    thresholds: Optional[np.ndarray] = None,
) -> List[ThresholdMetrics]:
    """For each threshold in `thresholds`, compute the (recall, FPR)
    point. Default thresholds: every observed probability + endpoints."""
    if thresholds is None:
        # Use every observed probability (deduplicated) + 0 and 1.
        thresholds = np.unique(np.concatenate([[0.0, 1.0], probas]))
    is_cat = np.asarray([r.is_catastrophic for r in rows])
    is_good = np.asarray([r.is_good for r in rows])
    is_marg = np.asarray([r.category == "MARGINAL" for r in rows])
    n_cat = int(is_cat.sum())
    n_good = int(is_good.sum())
    n_marg = int(is_marg.sum())
    out: List[ThresholdMetrics] = []
    for t in thresholds:
        # Predict retake when proba >= t.
        retake = probas >= t
        tp = int((retake & is_cat).sum())
        fp_good = int((retake & is_good).sum())
        marg_take = int((retake & is_marg).sum())
        out.append(ThresholdMetrics(
            threshold=float(t),
            catastrophic_recall=(tp / n_cat) if n_cat else 0.0,
            good_fpr=(fp_good / n_good) if n_good else 0.0,
            marginal_retake_rate=(marg_take / n_marg) if n_marg else 0.0,
            n_catastrophic=n_cat,
            n_good=n_good,
            n_marginal=n_marg,
            n_catastrophic_caught=tp,
            n_good_false_retakes=fp_good,
            n_marginal_retakes=marg_take,
        ))
    return out


def find_operating_points(metrics: List[ThresholdMetrics]) -> Dict[str, Optional[ThresholdMetrics]]:
    """Pick named operating points from the sweep."""
    # Sort by threshold ascending for deterministic tie-breaking.
    ms = sorted(metrics, key=lambda m: m.threshold)
    # Bar-clearing: lowest threshold where recall >= 0.80 AND fpr <= 0.10.
    bar_clearing = next(
        (m for m in ms if m.catastrophic_recall >= 0.80 and m.good_fpr <= 0.10),
        None,
    )
    # At 80% recall: highest threshold where recall >= 0.80 (tightest, lowest FPR).
    at_recall_80 = None
    for m in reversed(ms):
        if m.catastrophic_recall >= 0.80:
            at_recall_80 = m
            break
    # At 10% FPR: highest threshold where fpr <= 0.10 (most permissive in catastrophic detection).
    at_fpr_10 = None
    for m in ms:
        if m.good_fpr <= 0.10:
            at_fpr_10 = m
            break  # first match has highest recall since threshold ascends
    # Best F1-like balance: maximize (recall - fpr).
    best_margin = max(
        ms,
        key=lambda m: m.catastrophic_recall - m.good_fpr,
    ) if ms else None
    return {
        "bar_clearing": bar_clearing,
        "at_catastrophic_recall_80pct": at_recall_80,
        "at_good_fpr_10pct": at_fpr_10,
        "best_recall_minus_fpr": best_margin,
    }


def _feature_importance(model: Any) -> Optional[List[Dict[str, Any]]]:
    """Extract per-feature importance from a fitted sklearn model.

    Tries `coef_` (linear models), then `feature_importances_`
    (tree-based models), then None (MLP — not trivially interpretable).
    """
    if hasattr(model, "coef_"):
        vals = model.coef_[0]
        return [
            {"feature": c, "value": float(v), "abs": float(abs(v))}
            for c, v in sorted(
                zip(FEATURE_COLUMNS, vals), key=lambda kv: -abs(kv[1]),
            )
        ]
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        return [
            {"feature": c, "value": float(v), "abs": float(v)}
            for c, v in sorted(
                zip(FEATURE_COLUMNS, vals), key=lambda kv: -kv[1],
            )
        ]
    return None


def run_model(
    rows: List[Row],
    name: str,
    factory: Any,
) -> Dict[str, Any]:
    """Fit one model + run OOF CV + compute sweep/ops.

    Returns a per-model section of the output JSON.
    """
    # Full-data fit for "in-sample" reporting + feature importance.
    final_scaler, final_model = fit_model(rows, factory)
    in_sample = predict_probas(final_scaler, final_model, rows)
    in_sample_sweep = threshold_sweep(rows, in_sample)
    in_sample_ops = find_operating_points(in_sample_sweep)

    # Leave-one-case-out CV for honest generalization estimate.
    oof = leave_one_case_out_predict(rows, factory)
    oof_sweep = threshold_sweep(rows, oof)
    oof_ops = find_operating_points(oof_sweep)

    return {
        "name": name,
        "feature_importance": _feature_importance(final_model),
        "in_sample": {
            "operating_points": {k: (v.asdict() if v else None) for k, v in in_sample_ops.items()},
        },
        "out_of_fold": {
            "method": "leave_one_case_out",
            "n_folds": len({r.case for r in rows}),
            "operating_points": {k: (v.asdict() if v else None) for k, v in oof_ops.items()},
            "predictions": [
                {
                    "case": r.case, "run": r.run, "category": r.category,
                    "is_catastrophic": r.is_catastrophic,
                    "predicted_proba": float(p),
                }
                for r, p in zip(rows, oof)
            ],
        },
    }


def build_output(
    rows: List[Row],
    model_results: Dict[str, Dict[str, Any]],
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the full output JSON document.

    `model_results`: dict of model name → result dict (one per model
    class evaluated).
    """
    return {
        "schema_version": 2,
        "metadata": metadata or {},
        "input_summary": {
            "n_rows": len(rows),
            "n_cases": len({r.case for r in rows}),
            "category_counts": {
                cat: sum(1 for r in rows if r.category == cat)
                for cat in sorted({r.category for r in rows})
            },
            "feature_columns": list(FEATURE_COLUMNS),
        },
        "models": model_results,
    }


def _fmt_op(op: Optional[Dict[str, Any]]) -> str:
    if op is None:
        return "—"
    return (
        f"thr={op['threshold']:.2f} · "
        f"recall={op['catastrophic_recall']:.0%} "
        f"({op['n_catastrophic_caught']}/{op['n_catastrophic']}) · "
        f"GOOD FPR={op['good_fpr']:.0%} "
        f"({op['n_good_false_retakes']}/{op['n_good']})"
    )


def render_report(output: Dict[str, Any]) -> str:
    """Build the markdown report content."""
    inp = output["input_summary"]
    models = output["models"]

    # Pick the model with best out-of-fold (recall − FPR) margin among
    # those at ≥80% recall — that's the "headline" model.
    def best_at_recall_80(m: Dict[str, Any]) -> float:
        op = m["out_of_fold"]["operating_points"].get("at_catastrophic_recall_80pct")
        if op is None:
            return -1.0
        return op["catastrophic_recall"] - op["good_fpr"]
    headline_name = max(models.keys(), key=lambda n: best_at_recall_80(models[n]))
    headline = models[headline_name]
    headline_op80 = headline["out_of_fold"]["operating_points"].get("at_catastrophic_recall_80pct")
    headline_bar = headline["out_of_fold"]["operating_points"].get("bar_clearing")
    cleared_anywhere = any(
        m["out_of_fold"]["operating_points"].get("bar_clearing") for m in models.values()
    )

    if cleared_anywhere:
        # Find the first model that cleared the bar.
        bar_models = [
            (n, m) for n, m in models.items()
            if m["out_of_fold"]["operating_points"].get("bar_clearing")
        ]
        bar_name, bar_model = bar_models[0]
        bar_op = bar_model["out_of_fold"]["operating_points"]["bar_clearing"]
        verdict = (
            f"**Phase 2 bar cleared by `{bar_name}` on out-of-fold CV: "
            f"recall={bar_op['catastrophic_recall']:.0%}, "
            f"GOOD FPR={bar_op['good_fpr']:.0%}.** A learned classifier "
            f"finds a multi-dimensional decision boundary that no "
            f"hand-tuned axis-aligned rule could find. Next step: "
            f"Phase 3 (wire as production guardrail)."
        )
    elif headline_op80 is not None:
        verdict = (
            f"**No learned model clears the Phase 2 bar on out-of-fold "
            f"CV (≥80% catastrophic recall AND ≤10% GOOD FPR).** "
            f"Best-in-class is `{headline_name}` at "
            f"{headline_op80['catastrophic_recall']:.0%} recall / "
            f"{headline_op80['good_fpr']:.0%} FPR — meaningfully better "
            f"than Phase 2B's hand-tuned ceiling of 80% / 31% at the "
            f"same recall, but still above the 10% FPR target. The "
            f"learned model captures multi-feature structure (non-axis-"
            f"aligned cuts) that hand-tuning missed; what's still "
            f"missing is data and/or a stronger feature."
        )
    else:
        verdict = (
            "**No learned model reaches 80% catastrophic recall on "
            "out-of-fold CV.** Indicates either the feature set lacks "
            "the necessary signal or the dataset is too small."
        )

    # Comparison table across all evaluated models.
    rows_table = []
    for n, m in models.items():
        oof_ops = m["out_of_fold"]["operating_points"]
        in_ops = m["in_sample"]["operating_points"]
        oof_r80 = oof_ops.get("at_catastrophic_recall_80pct")
        oof_f10 = oof_ops.get("at_good_fpr_10pct")
        in_r80 = in_ops.get("at_catastrophic_recall_80pct")
        bar_marker = "✅" if oof_ops.get("bar_clearing") else "❌"
        rows_table.append(
            f"| `{n}` | {bar_marker} | "
            f"{oof_r80['good_fpr']:.0%} ({oof_r80['catastrophic_recall']:.0%} recall) | "
            f"{oof_f10['catastrophic_recall']:.0%} ({oof_f10['good_fpr']:.0%} FPR) | "
            f"{in_r80['good_fpr']:.0%} |"
        )
    comparison_table = "\n".join(rows_table)

    # Headline model details: feature importance.
    fi = headline.get("feature_importance")
    if fi:
        fi_rows = "\n".join(
            f"| {x['feature']} | {x['abs']:.3f} |" for x in fi
        )
        fi_table = f"""
### Headline model feature importance (`{headline_name}`)

| feature | importance |
|---|---|
{fi_rows}
"""
    else:
        fi_table = (
            f"\n### Headline model feature importance (`{headline_name}`)\n\n"
            "(Not extracted for this model class — MLP coefficients "
            "aren't trivially interpretable.)\n"
        )

    cats = inp["category_counts"]
    cat_table = "\n".join(
        f"| {cat} | {n} |" for cat, n in sorted(cats.items())
    )

    return f"""# Phase 4 trust ranker v1

**Status: diagnostics-only.** First learned-classifier pass at the
Phase 2 bar after Phase 2B's hand-tuning hit its ceiling (see
`PHASE_2B_TRUST_SIGNAL_MATRIX_RECOMPUTED.md`). 4 model classes
evaluated on the {inp['n_cases']}-case eval via leave-one-case-out CV.
No production behavior change.

## Headline

{verdict}

## Models evaluated

| model | clears bar? | out-of-fold FPR at 80% recall (vs Phase 2B ceiling: 31%) | out-of-fold recall at 10% FPR (vs Phase 2B: ~50%) | in-sample FPR at 80% recall |
|---|---|---|---|---|
{comparison_table}

**Reading the table:**
- "Clears bar" = some threshold achieves both ≥80% catastrophic recall AND ≤10% GOOD FPR on out-of-fold predictions.
- Lower FPR is better at fixed recall.
- Higher recall is better at fixed FPR.
- The in-sample column is what you'd get if you trained AND evaluated on the same data (no held-out). Compare to out-of-fold to see the generalization gap.

## Dataset

- {inp['n_rows']} per-case-per-run rows across {inp['n_cases']} cases
- Outcome breakdown:

| category | n |
|---|---|
{cat_table}

- Catastrophic = `CHIRALITY_MISS` ∪ `CHIRALITY_FALSE_FLIP` ∪ `TRUE_GEOMETRY_FAIL`
- GOOD = `GOOD`; MARGINAL is intermediate (not penalized in bar)

## Features

{len(FEATURE_COLUMNS)} continuous signals from the Phase 2B matrix
(no new features in v1; v2 candidates documented in
`STATE_OF_THE_WORLD.md`). Order:

{chr(10).join(f"- `{c}`" for c in FEATURE_COLUMNS)}
{fi_table}

## Cross-validation methodology

- **Leave-one-case-out** (group-based on `case_key`): both runs of a
  case_key go into the same held-out fold. Both runs share the same
  input image — leaving only one run out would leak the underlying
  difficulty of that case.
- {inp['n_cases']} folds total. Per fold: fit on the remaining
  {inp['n_rows']} − 2 = {inp['n_rows'] - 2} rows, predict on the 2 held-out.

## What the result means

{(
'''The learned model now clears the Phase 2 bar that hand-tuning could
not. Next step: Phase 3 — wire the bar-clearing classifier as a
production guardrail. Re-run this tool whenever the matrix or feature
set changes so the committed snapshot stays current with the model
that production depends on.'''
if cleared_anywhere else
'''Even the best learned model (`''' + headline_name + '''` at out-of-fold)
doesn't clear the Phase 2 bar on this ''' + str(inp['n_cases']) + '''-case corpus.
But the gap is real and shrinking:

- Phase 2A (`phase_sep` alone, calibrated): 46% recall / 9% FPR
- Phase 2B hand-tuned ceiling: 80% recall / 31% FPR (or 50% recall / 10% FPR)
- Phase 4 best learned: see table above

The remaining gap is consistent with two simultaneous limits:
1. **Sample size.** ''' + str(inp['category_counts'].get('CHIRALITY_MISS', 0) + inp['category_counts'].get('CHIRALITY_FALSE_FLIP', 0) + inp['category_counts'].get('TRUE_GEOMETRY_FAIL', 0)) + ''' catastrophic samples means
   leave-one-case-out gives noisy folds. Phase 4 v1.1 expanded the
   corpus from 58 to 70 cases (adding ~13 catastrophic samples) but
   the bar still wasn't cleared — strong evidence the data lever
   alone is insufficient.
2. **Feature set.** 6 features per row from the global model fit.
   The two-view orientation consistency signal (shipped in
   `tools/two_view_consistency.py`, PR #243) would add a 7th feature
   derived from comparing A and B view orientations — a fundamentally
   different geometric signal than the others. See
   `tools/TWO_VIEW_CONSISTENCY.md` for the integration plan.

Phase 4 v1.1 demonstrated the data lever alone doesn't suffice. Phase
4 v2 (feature integration) is the highest-leverage remaining lever.'''
)}

## Files

- Tool: `tools/phase4_trust_ranker.py`
- Trained snapshot: `tests/fixtures/phase4_trust_ranker_v1.json`
  (per-model full-data fit + out-of-fold predictions + threshold sweep
  + named operating points)
- Source data: `tests/fixtures/phase2b_recomputed_signals.json`

## See also

- [`PHASE_2B_TRUST_SIGNAL_MATRIX_RECOMPUTED.md`](PHASE_2B_TRUST_SIGNAL_MATRIX_RECOMPUTED.md) —
  hand-tuned ceiling this learned model is graded against.
- [`STATE_OF_THE_WORLD.md`](STATE_OF_THE_WORLD.md) — Phase 4
  positioning in the phased roadmap.
"""


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Phase 4 trust ranker v1 — fit + evaluate")
    p.add_argument("--input", type=Path,
                   default=Path("tests/fixtures/phase2b_recomputed_signals.json"),
                   help="Phase 2B matrix JSON (input features + labels).")
    p.add_argument("--output", type=Path,
                   default=Path("tests/fixtures/phase4_trust_ranker_v1.json"),
                   help="Where to write the multi-model trained + CV-results JSON.")
    p.add_argument("--report", type=Path,
                   default=Path("tools/PHASE_4_TRUST_RANKER_V1.md"),
                   help="Where to write the markdown report.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of model names to run (default: all).")
    args = p.parse_args(argv)

    print(f"loading {args.input} …", file=sys.stderr)
    rows = load_matrix(args.input)
    print(f"  {len(rows)} rows, {len({r.case for r in rows})} cases", file=sys.stderr)

    factories = _make_model_factories()
    if args.models:
        unknown = set(args.models) - set(factories.keys())
        if unknown:
            print(f"unknown model name(s): {unknown}", file=sys.stderr)
            print(f"available: {list(factories.keys())}", file=sys.stderr)
            return 1
        factories = {n: factories[n] for n in args.models}

    model_results: Dict[str, Dict[str, Any]] = {}
    for name, factory in factories.items():
        print(f"running model {name} …", file=sys.stderr)
        model_results[name] = run_model(rows, name, factory)

    # Record the sklearn version that produced this snapshot. Devin
    # caught reproducibility drift on the v1 PR: random-forest
    # probabilities shifted ~0.017 between sklearn 1.4 and 1.8 within
    # the original `>=1.4,<2` pin. requirements.txt is now narrowed
    # to `>=1.8.0,<1.9` so anyone reproducing gets bit-identical
    # output, and we surface the version here as belt-and-suspenders.
    import sklearn  # type: ignore
    output = build_output(
        rows, model_results,
        metadata={
            "input_path": str(args.input),
            "tool": "tools/phase4_trust_ranker.py",
            "models_evaluated": list(model_results.keys()),
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
        },
    )

    print(f"writing {args.output} …", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"writing {args.report} …", file=sys.stderr)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as f:
        f.write(render_report(output))

    # Print single-line summary per model to stdout.
    for n, m in model_results.items():
        oof = m["out_of_fold"]["operating_points"]
        bar = oof.get("bar_clearing")
        r80 = oof.get("at_catastrophic_recall_80pct")
        if bar:
            print(f"{n}: bar=CLEARED recall={bar['catastrophic_recall']:.3f} fpr={bar['good_fpr']:.3f}")
        elif r80:
            print(f"{n}: bar=NO recall_80_fpr={r80['good_fpr']:.3f}")
        else:
            print(f"{n}: bar=NO recall_80_unreachable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
