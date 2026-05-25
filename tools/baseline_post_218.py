#!/usr/bin/env python3
"""Legacy post-#218 baseline + failure taxonomy on the axis-labeled
gallery.

Runs the global cube model on every photo whose vertex + 3 legacy-`near`
corners are recorded in the user-labeled ground truth, then
categorizes each case into one of:

IMPORTANT: the `near_*` fixture semantics are legacy after the
2026-05-23 full-corner convention reset. A 12-row seed audit shows those
fields match the far/double-axis triplet, not canonical one-edge labels.
Treat this baseline and its `CHIRALITY_*` row-level evidence as provisional
until regenerated from explicit `Va/Vb + 0..5` labels.

  GOOD                  mean per-axis bearing error < 10°
  MARGINAL              10° ≤ err < 25°
  CHIRALITY_MISS        err > 25° AND phase_check ∈ {correct, ambiguous}
                        AND best-perm rotation of model.far matches user.near
                        within 25° (i.e., the right 3 corners ARE in the
                        model's far set, the detector just missed)
  CHIRALITY_FALSE_FLIP  err > 25° AND phase_check == corrected_60deg_flip
                        AND best-perm rotation of model.far matches user.near
                        within 25° (the detector flipped a model that was
                        already correct, making it wrong)
  TRUE_GEOMETRY_FAIL    err > 25° AND neither model.near nor model.far matches
                        user.near within 25° — fit is bad regardless of
                        phase choice

Bearings are scale+translation invariant, so we work in gallery coords
directly (no crop reconstruction needed).

Usage:
    .venv/bin/python tools/baseline_post_218.py \\
        --truth tests/fixtures/gcm_axis_ground_truth.json \\
        --gallery /path/to/axis_labeling_gallery_dir \\
        --runs 2 \\
        --out tests/fixtures/post_218_baseline.json \\
        --report tools/POST_218_BASELINE_AND_TAXONOMY.md

If --truth is omitted, looks for tests/fixtures/gcm_axis_ground_truth.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from tools.global_cube_model import fit_global_cube_model  # noqa: E402
from tools.interior_bezel_detection import detect_interior_bezel_lines  # noqa: E402

DEFAULT_TRUTH = ROOT / "tests" / "fixtures" / "gcm_axis_ground_truth.json"
DEFAULT_GALLERY = Path.home() / "axis_labeling"
DEFAULT_OUT = ROOT / "tests" / "fixtures" / "post_218_baseline.json"
DEFAULT_REPORT = ROOT / "tools" / "POST_218_BASELINE_AND_TAXONOMY.md"


def _bearing(o: Tuple[float, float], t: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(t[1] - o[1], t[0] - o[0])) % 360


def _best_perm_err(model_bearings: List[float], user_bearings: List[float]) -> float:
    best = math.inf
    for perm in itertools.permutations(range(3)):
        diffs = []
        for i in range(3):
            d = abs((model_bearings[i] - user_bearings[perm[i]] + 180) % 360 - 180)
            diffs.append(d)
        m = sum(diffs) / 3.0
        if m < best:
            best = m
    return best


def _categorize(
    err_near: float,
    err_far: float,
    chir_status: str,
) -> str:
    if err_near < 10.0:
        return "GOOD"
    if err_near < 25.0:
        return "MARGINAL"
    if err_far < 25.0:
        # model.far matches user.near — phase is geometrically wrong
        if chir_status == "corrected_60deg_flip":
            return "CHIRALITY_FALSE_FLIP"
        return "CHIRALITY_MISS"
    return "TRUE_GEOMETRY_FAIL"


def _run_one_case(
    sess: Any,
    img_path: Path,
    user_v: Tuple[float, float],
    user_near: List[Tuple[float, float]],
    n_runs: int,
) -> List[Dict[str, Any]]:
    from rembg import remove  # noqa: E402  (deferred for optional dep)

    img = Image.open(img_path).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    rgba = remove(img, session=sess)
    mask = np.array(rgba.split()[-1], dtype=np.uint8) > 128
    det = detect_interior_bezel_lines(rgb, mask)

    user_bearings = sorted([_bearing(user_v, p) for p in user_near])

    results: List[Dict[str, Any]] = []
    for run in range(n_runs):
        model = fit_global_cube_model(det, rgb, mask, optimize=True)
        if model is None:
            results.append({"run": run, "status": "model_fit_failed"})
            continue

        v = model.cube_center_screen
        near = [model.visible_corners[k] for k in ["h_x", "h_y", "h_z"]]
        far = [model.visible_corners[k] for k in ["h_xy", "h_xz", "h_yz"]]
        near_b = sorted([_bearing(v, p) for p in near])
        far_b = sorted([_bearing(v, p) for p in far])

        err_near = _best_perm_err(near_b, user_bearings)
        err_far = _best_perm_err(far_b, user_bearings)
        chir = model.debug.get("phase_check", "?")
        sep = model.debug.get("phase_darkness_separation")
        category = _categorize(err_near, err_far, chir)

        results.append({
            "run": run,
            "err_near_deg": round(err_near, 1),
            "err_far_deg": round(err_far, 1),
            "phase_check": chir,
            "phase_sep": round(sep, 1) if sep is not None else None,
            "category": category,
        })
    return results


def _summarize(by_case: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    cat_counts = Counter()
    err_distribution = Counter()
    all_runs = []
    for case_results in by_case.values():
        for r in case_results:
            if "category" in r:
                cat_counts[r["category"]] += 1
                err = r["err_near_deg"]
                if err < 5:
                    err_distribution["<5°"] += 1
                elif err < 10:
                    err_distribution["5-10°"] += 1
                elif err < 25:
                    err_distribution["10-25°"] += 1
                elif err < 45:
                    err_distribution["25-45°"] += 1
                else:
                    err_distribution[">45°"] += 1
                all_runs.append(r)

    n_total = sum(cat_counts.values())

    # Case-level stability: for each case, the set of categories across runs
    stable_good = 0
    stable_bad = 0
    mixed = 0
    for runs in by_case.values():
        cats = {r.get("category") for r in runs if "category" in r}
        if cats == {"GOOD"}:
            stable_good += 1
        elif cats - {"GOOD", "MARGINAL"}:
            if any(c in cats for c in ("CHIRALITY_MISS", "CHIRALITY_FALSE_FLIP", "TRUE_GEOMETRY_FAIL")) and not (cats & {"GOOD"}):
                stable_bad += 1
            else:
                mixed += 1
        else:
            mixed += 1

    return {
        "n_runs": n_total,
        "n_cases": len(by_case),
        "category_counts": dict(cat_counts),
        "error_distribution": dict(err_distribution),
        "stable_good_cases": stable_good,
        "stable_bad_cases": stable_bad,
        "mixed_cases": mixed,
    }


def _render_markdown(summary: Dict[str, Any], by_case: Dict[str, List[Dict[str, Any]]]) -> str:
    n = summary["n_runs"]
    cats = summary["category_counts"]
    err_dist = summary["error_distribution"]

    def pct(x):
        return f"{100 * x / n:.1f}%" if n else "0.0%"

    lines = []
    lines.append("# Post-#218 baseline + failure taxonomy")
    lines.append("")
    lines.append("## Role of this document")
    lines.append("")
    lines.append("This is the **legacy decision spine** for first-principles geometry")
    lines.append("work as of 2026-05-22, per the Codex+Devin strategic synthesis:")
    lines.append("")
    lines.append("> A plausible cube model is cheap. A trustworthy cube model is")
    lines.append("> hard.")
    lines.append(">")
    lines.append("> First-principles work must either improve guardrails around")
    lines.append("> `cv-local`, produce labeled training data, or demonstrate")
    lines.append("> safe held-out improvement.")
    lines.append("")
    lines.append("The global cube model is **NOT** on a near-term path to replace")
    lines.append("`cv-local` as the primary recognizer. Its role is:")
    lines.append("")
    lines.append("- a **trust layer** around `cv-local`: low-confidence geometry")
    lines.append("  should route to manual/retake, not to legal repair;")
    lines.append("- a **source of labeled training data** for a future learned")
    lines.append("  vertex/axis ranker;")
    lines.append("- a **benchmark harness**: every geometry-sensitive PR should")
    lines.append("  emit row-level before/after deltas against this baseline, not")
    lines.append("  just aggregate pass rates (see `--diff` mode below).")
    lines.append("")
    lines.append("**2026-05-23 convention caution:** this report was computed against")
    lines.append("`tests/fixtures/gcm_axis_ground_truth.json`, whose 3 axis-endpoint")
    lines.append("fields (canonical name `axis_x/axis_y/axis_z`; legacy alias")
    lines.append("`near_x/near_y/near_z` still accepted) sit at the FAR / double-axis")
    lines.append("triplet (`A -> 0,2,4`, `B -> 1,3,5`), not canonical one-edge labels.")
    lines.append("The canonical corner convention is")
    lines.append("[`FULL_CORNER_LABELING.md`](FULL_CORNER_LABELING.md):")
    lines.append("`A slots: upper=Va+1,0,5; right=Va+3,2,1; front=Va+5,4,3` and")
    lines.append("`B slots: upper=Vb+2,3,4; right=Vb+0,1,2; front=Vb+4,5,0`. Canonical WCA")
    lines.append("face names for side slots depend on capture yaw. Use this report for")
    lines.append("historical context, but do not treat row-level `CHIRALITY_*` evidence as")
    lines.append("canonical until the baseline is regenerated from full-corner truth.")
    lines.append("")
    lines.append("## What the \"chirality\" concept actually is")
    lines.append("")
    lines.append("The detector and code use the word \"chirality\" but it's a")
    lines.append("slight misnomer — strict-geometry chirality means handedness")
    lines.append("under reflection. What we actually face is a **60° near/far")
    lines.append("phase ambiguity** stemming from the cube's 3-fold rotational")
    lines.append("symmetry around its body diagonal.")
    lines.append("")
    lines.append("In iso projection, the 6 outer hexagon-silhouette vertices")
    lines.append("alternate between 3 NEAR corners (1 cube-edge from the visible")
    lines.append("trihedral vertex) and 3 FAR corners (2 cube-edges via face")
    lines.append("diagonal). Swap the labels and you get a different valid cube")
    lines.append("pose that projects to the SAME silhouette under orthographic")
    lines.append("projection — the 7-anchor Procrustes fit has identical residual")
    lines.append("for either assignment.")
    lines.append("")
    lines.append("**This is real first-principles geometry, not invented")
    lines.append("complexity.** The 7 anchor points (vertex + 6 hex corners) give")
    lines.append("14 measurements for a 6-DOF cube model — over-determined")
    lines.append("linearly, but the solution set has two equally-valid options")
    lines.append("under the body-diagonal symmetry. With perspective there's a")
    lines.append("small distinguishing signal from foreshortening, but it's")
    lines.append("noise-shaped at our anchor-extraction precision.")
    lines.append("")
    lines.append("### What's principled vs empirical in the current detector")
    lines.append("")
    lines.append("- **Principled (keep)**: that we need to resolve the phase at")
    lines.append("  all, that the resolution requires evidence beyond the 7")
    lines.append("  anchors, and that the model should expose a confidence")
    lines.append("  signal for whether the resolution is trustworthy.")
    lines.append("- **Empirical (treat as stopgap)**: the line-darkness signal,")
    lines.append("  its inverted polarity (`sep<0` ≡ correct, opposite of naive")
    lines.append("  bezel-darkness reasoning), and the |sep| < 10 ambiguous band.")
    lines.append("  These are calibrated against the 58 cases, not derived from")
    lines.append("  first principles. PR #218's 33pp accuracy lift from a single")
    lines.append("  block reorder (vertex ensemble BEFORE phase check) is")
    lines.append("  itself evidence that the detector is sensitive to non-")
    lines.append("  geometric noise.")
    lines.append("")
    lines.append("The architectural answer is to fold additional evidence INTO")
    lines.append("the Procrustes fit (bezel-line angles, two-view consistency,")
    lines.append("learned priors) rather than detect-and-correct downstream.")
    lines.append("That's what the recommended sequence's step 6 (learned")
    lines.append("vertex/axis ranker with calibrated abstention) actually is.")
    lines.append("The current darkness detector becomes vestigial once that")
    lines.append("lands.")
    lines.append("")
    lines.append("## Baseline snapshot")
    lines.append("")
    lines.append("Re-baseline of the global cube model after PR #213 (phase")
    lines.append("auto-correct) and PR #218 (vertex ensemble before phase")
    lines.append("check) landed.")
    lines.append("")
    lines.append(f"**Eval set**: {summary['n_cases']} cases × "
                 f"{n // summary['n_cases'] if summary['n_cases'] else 0} runs each "
                 f"= {n} total runs.")
    lines.append("")
    lines.append("Ground truth: legacy user-labeled vertex + 3 `near_*` corners per photo")
    lines.append("(`tests/fixtures/gcm_axis_ground_truth.json`). The eval compares")
    lines.append("model bearings (in gallery coords) to user bearings (in original")
    lines.append("coords); bearings are exactly invariant under the gallery's")
    lines.append("uniform-scale-and-translation crop, so no crop reconstruction is")
    lines.append("needed. The `near_*` target semantics are provisional pending the")
    lines.append("full-corner migration.")
    lines.append("")
    lines.append("## Headline accuracy")
    lines.append("")
    lines.append("| accuracy band                    | runs |  %  |")
    lines.append("|----------------------------------|-----:|----:|")
    for band in ["<5°", "5-10°", "10-25°", "25-45°", ">45°"]:
        c = err_dist.get(band, 0)
        lines.append(f"| {band:32s} | {c:>4d} | {pct(c)} |")
    lines.append("")

    n_good = cats.get("GOOD", 0) + cats.get("MARGINAL", 0)
    n_bad = cats.get("CHIRALITY_MISS", 0) + cats.get("CHIRALITY_FALSE_FLIP", 0) + cats.get("TRUE_GEOMETRY_FAIL", 0)
    lines.append(f"- **{pct(n_good)} of runs land at <25° bearing error** (GOOD + MARGINAL).")
    lines.append(f"- **{pct(n_bad)} of runs are catastrophic** (>25° err in one of the failure")
    lines.append(f"  modes below).")
    lines.append("")
    lines.append("## Failure taxonomy")
    lines.append("")
    lines.append("| category                  | runs |  %  | meaning |")
    lines.append("|---------------------------|-----:|----:|---------|")
    descriptions = {
        "GOOD": "All axes within 10° of user labels — model is essentially right.",
        "MARGINAL": "10–25° err — small jitter, color sampling probably still OK.",
        "CHIRALITY_MISS": "Model.far matches user.near; detector said `correct` or `ambiguous` — flip needed but missed.",
        "CHIRALITY_FALSE_FLIP": "Model.far matches user.near; detector said `corrected_60deg_flip` — wrongly flipped a previously-correct model.",
        "TRUE_GEOMETRY_FAIL": "Neither model.near nor model.far matches user.near — fit is bad regardless of phase.",
    }
    for cat in ["GOOD", "MARGINAL", "CHIRALITY_MISS", "CHIRALITY_FALSE_FLIP", "TRUE_GEOMETRY_FAIL"]:
        c = cats.get(cat, 0)
        lines.append(f"| {cat:25s} | {c:>4d} | {pct(c)} | {descriptions[cat]} |")
    lines.append("")
    lines.append("## Case-level stability across runs")
    lines.append("")
    lines.append("| outcome                       | cases |  %  |")
    lines.append("|-------------------------------|------:|----:|")
    tot = summary["n_cases"]
    g = summary["stable_good_cases"]
    b = summary["stable_bad_cases"]
    m = summary["mixed_cases"]
    pct_c = lambda x: f"{100*x/tot:.1f}%" if tot else "0.0%"
    lines.append(f"| always GOOD across runs       | {g:>5d} | {pct_c(g)} |")
    lines.append(f"| always BAD (catastrophic+)    | {b:>5d} | {pct_c(b)} |")
    lines.append(f"| mixed / varies between runs   | {m:>5d} | {pct_c(m)} |")
    lines.append("")
    lines.append("Mixed cases are where the Procrustes 6! brute-force still picks")
    lines.append("different symmetry-equivalent permutations across runs and the")
    lines.append("phase detector doesn't reliably rescue all of them. The")
    lines.append("deterministic-tie-breaker path was scoped (see NEAR_FAR_PHASE_REPORT.md")
    lines.append("\"What's next\") but deprioritized in favor of higher-leverage work")
    lines.append("on vertex localization.")
    lines.append("")

    # Worst cases (worst run per case)
    case_worst = []
    for k, runs in by_case.items():
        worst = max(runs, key=lambda r: r.get("err_near_deg", 0))
        if "err_near_deg" in worst:
            case_worst.append((k, worst))
    case_worst.sort(key=lambda r: -r[1]["err_near_deg"])
    lines.append("## 10 worst cases (worst run per case)")
    lines.append("")
    lines.append("| set | err_near | category | chir_check | sep |")
    lines.append("|-----|---------:|----------|------------|----:|")
    for k, r in case_worst[:10]:
        sep = r["phase_sep"] if r.get("phase_sep") is not None else "n/a"
        sep_s = f"{sep:+.1f}" if isinstance(sep, (int, float)) else str(sep)
        lines.append(f"| {k} | {r['err_near_deg']:>6.1f}° | {r['category']:>22s} | {r['phase_check']:>30s} | {sep_s} |")
    lines.append("")

    lines.append("## What this baseline says")
    lines.append("")
    lines.append("1. The phase auto-correction (PRs #210/#213/#218) handles the")
    lines.append("   dominant failure mode at the symptom level. The remaining")
    lines.append("   catastrophic band is largely phase-decision miscalls")
    lines.append("   (detector said correct/ambig but should have flipped, or")
    lines.append("   flipped when shouldn't have).")
    lines.append("2. Detector confidence (|sep|) tracks success. Strong-|sep| commits")
    lines.append("   are almost always right; the failures cluster in the weak-|sep|")
    lines.append("   band where neither commit-correct nor commit-flip is reliable.")
    lines.append("3. The case-level non-determinism is real — the Procrustes 6!")
    lines.append("   brute-force is the root cause. A deterministic tie-breaker")
    lines.append("   would address it directly, but is **explicitly deprioritized**")
    lines.append("   per the Codex+Devin strategic shift (no more handcrafted")
    lines.append("   vertex/phase heuristics — the bar is now \"safe held-out")
    lines.append("   improvement\" or \"feeds the trust layer\").")
    lines.append("")
    lines.append("## Recommended next sequence (per Codex+Devin)")
    lines.append("")
    lines.append("In order, gated on the previous step demonstrating value:")
    lines.append("")
    lines.append("1. **This baseline + taxonomy artifact** — done by this PR. Sets")
    lines.append("   the regression gate.")
    lines.append("2. **Geometry trust-policy diagnostics** — add `model.debug`")
    lines.append("   fields for phase confidence, axis agreement against the")
    lines.append("   detected bezels, face-quad-consistency, grid/source-")
    lines.append("   contamination. Diagnostics-only; no behavior change.")
    lines.append("3. **Guardrail experiment** — route low-trust cases to")
    lines.append("   manual/retake. Success metric is *fewer confident wrong")
    lines.append("   solves with tolerable abstention*, NOT \"more solves.\"")
    lines.append("   The product framing here is Devin's: \"use first-principles")
    lines.append("   diagnostics to decide when to ask the user for a second")
    lines.append("   capture / manual fixer path.\"")
    lines.append("4. **Stable labeled benchmark harness** — every geometry-")
    lines.append("   sensitive PR runs `tools/baseline_post_218.py --diff`")
    lines.append("   against the merge-base baseline and reports row-level")
    lines.append("   deltas. PR body must include the diff summary. Aggregate")
    lines.append("   mean/p95 metrics are NOT sufficient (Devin: \"some changes")
    lines.append("   improve averages while worsening critical rows\").")
    lines.append("5. **Active-label queue for the eval set** — extend the 58")
    lines.append("   labels systematically rather than opportunistically. Pick")
    lines.append("   the next photos to label based on which model-disagreement")
    lines.append("   regions are currently under-sampled, not on which photos")
    lines.append("   look visually interesting.")
    lines.append("6. **Learned vertex/axis ranker** — only after (2)–(5)")
    lines.append("   demonstrate the trust layer works. Must train/eval with")
    lines.append("   held-out splits and calibrated abstention. Note Devin's")
    lines.append("   caveat: \"a candidate oracle means a localizer is near")
    lines.append("   production\" is FALSE — the search space containing truth")
    lines.append("   doesn't mean ranking/confidence is solved. Calibrated")
    lines.append("   abstention is the actual deliverable, not top-1 accuracy.")
    lines.append("")
    lines.append("## What's explicitly off the table")
    lines.append("")
    lines.append("Per the Codex+Devin synthesis, these are NOT next bets:")
    lines.append("")
    lines.append("- More handcrafted vertex/phase heuristics (dark-line variants,")
    lines.append("  junction extractors, scalar scorers). Diminishing returns; the")
    lines.append("  sprint repeatedly falsified them at the safe-coverage bar.")
    lines.append("- More SAM3 prompt bakeoffs without a materially new signal.")
    lines.append("- Replacing `cv-local` with the global cube model. The global")
    lines.append("  model is scaffolding, not a recognizer.")
    lines.append("- Aggregate-metric-only A/B comparisons (without row-level diff).")
    lines.append("")
    lines.append("## How to use this as a regression gate")
    lines.append("")
    lines.append("Before a geometry-sensitive PR:")
    lines.append("```bash")
    lines.append(".venv/bin/python tools/baseline_post_218.py \\")
    lines.append("  --out /tmp/baseline_my_branch.json --report /dev/null")
    lines.append("```")
    lines.append("")
    lines.append("Then diff:")
    lines.append("```bash")
    lines.append(".venv/bin/python tools/baseline_post_218.py \\")
    lines.append("  --diff tests/fixtures/post_218_baseline.json /tmp/baseline_my_branch.json")
    lines.append("```")
    lines.append("")
    lines.append("Paste the diff summary into the PR body. A PR that regresses")
    lines.append("any case from GOOD → catastrophic without offsetting wins is a")
    lines.append("blocker for merge regardless of aggregate metrics.")
    lines.append("")
    lines.append("## Reproducing")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python tools/baseline_post_218.py \\")
    lines.append("  --truth tests/fixtures/gcm_axis_ground_truth.json \\")
    lines.append("  --gallery ~/axis_labeling \\")
    lines.append("  --runs 2 \\")
    lines.append("  --out tests/fixtures/post_218_baseline.json \\")
    lines.append("  --report tools/POST_218_BASELINE_AND_TAXONOMY.md")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _render_diff(prev_path: Path, curr_path: Path) -> str:
    """Compare two baseline JSONs case-by-case and report row-level
    deltas suitable for pasting into a PR body."""
    prev = json.loads(prev_path.read_text())
    curr = json.loads(curr_path.read_text())
    prev_by = prev["by_case"]
    curr_by = curr["by_case"]
    prev_sum = prev["summary"]
    curr_sum = curr["summary"]

    def _best_run(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        scored = [r for r in runs if "err_near_deg" in r]
        if not scored:
            return {}
        return min(scored, key=lambda r: r["err_near_deg"])

    def _worst_run(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        scored = [r for r in runs if "err_near_deg" in r]
        if not scored:
            return {}
        return max(scored, key=lambda r: r["err_near_deg"])

    common = sorted(set(prev_by) & set(curr_by))
    deltas = []
    for k in common:
        p_worst = _worst_run(prev_by[k])
        c_worst = _worst_run(curr_by[k])
        if not p_worst or not c_worst:
            continue
        p_err = p_worst.get("err_near_deg", 0)
        c_err = c_worst.get("err_near_deg", 0)
        delta = c_err - p_err
        p_cat = p_worst.get("category", "?")
        c_cat = c_worst.get("category", "?")
        deltas.append((k, p_err, c_err, delta, p_cat, c_cat))

    deltas.sort(key=lambda r: r[3])  # biggest improvements first
    improvements = [d for d in deltas if d[3] < -2]
    regressions = [d for d in deltas if d[3] > 2]
    unchanged = [d for d in deltas if abs(d[3]) <= 2]

    lines = []
    lines.append("# Baseline diff")
    lines.append("")
    lines.append(f"prev: {prev_path}  ({prev_sum['n_cases']} cases × runs)")
    lines.append(f"curr: {curr_path}  ({curr_sum['n_cases']} cases × runs)")
    lines.append("")
    lines.append("## Aggregate deltas")
    lines.append("")
    lines.append("| metric | prev | curr | delta |")
    lines.append("|--------|-----:|-----:|------:|")
    for cat in ["GOOD", "MARGINAL", "CHIRALITY_MISS", "CHIRALITY_FALSE_FLIP", "TRUE_GEOMETRY_FAIL"]:
        p = prev_sum["category_counts"].get(cat, 0)
        c = curr_sum["category_counts"].get(cat, 0)
        d = c - p
        sign = "+" if d > 0 else ""
        lines.append(f"| {cat:25s} | {p:>4d} | {c:>4d} | {sign}{d:>4d} |")
    lines.append("")
    lines.append(f"- stable GOOD cases: {prev_sum['stable_good_cases']} → {curr_sum['stable_good_cases']}  ({curr_sum['stable_good_cases']-prev_sum['stable_good_cases']:+d})")
    lines.append(f"- stable BAD cases:  {prev_sum['stable_bad_cases']} → {curr_sum['stable_bad_cases']}  ({curr_sum['stable_bad_cases']-prev_sum['stable_bad_cases']:+d})")
    lines.append("")
    lines.append(f"## Case-level changes  (worst-run err per case, threshold ±2°)")
    lines.append("")
    lines.append(f"- improvements: {len(improvements)}")
    lines.append(f"- regressions:  {len(regressions)}")
    lines.append(f"- unchanged:    {len(unchanged)}")
    lines.append("")
    if regressions:
        lines.append("### Regressions (curr worse than prev)")
        lines.append("")
        lines.append("| set | prev err | curr err | Δ | prev cat → curr cat |")
        lines.append("|-----|---------:|---------:|--:|---------------------|")
        for k, p, c, d, pc, cc in sorted(regressions, key=lambda r: -r[3]):
            lines.append(f"| {k} | {p:>6.1f}° | {c:>6.1f}° | {d:+5.1f}° | {pc} → {cc} |")
        lines.append("")
    if improvements:
        lines.append("### Improvements (curr better than prev)")
        lines.append("")
        lines.append("| set | prev err | curr err | Δ | prev cat → curr cat |")
        lines.append("|-----|---------:|---------:|--:|---------------------|")
        for k, p, c, d, pc, cc in improvements[:20]:  # top 20 if many
            lines.append(f"| {k} | {p:>6.1f}° | {c:>6.1f}° | {d:+5.1f}° | {pc} → {cc} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", default=str(DEFAULT_TRUTH))
    ap.add_argument("--gallery", default=str(DEFAULT_GALLERY))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument(
        "--render-only", action="store_true",
        help="Skip model fitting; load existing --out JSON and re-render "
             "the markdown report from it. Useful when iterating on the "
             "report template without re-running the slow eval.",
    )
    ap.add_argument(
        "--diff", nargs=2, metavar=("PREV", "CURR"),
        help="Compare two baseline JSONs and emit a row-level diff "
             "report to stdout. Suitable for pasting into a PR body.",
    )
    args = ap.parse_args()

    if args.diff:
        print(_render_diff(Path(args.diff[0]), Path(args.diff[1])))
        return 0

    out_path = Path(args.out)
    report_path = Path(args.report)

    if args.render_only:
        if not out_path.exists():
            print(f"--render-only requires existing {out_path}", file=sys.stderr)
            return 2
        payload = json.loads(out_path.read_text())
        summary = payload["summary"]
        by_case = payload["by_case"]
        report = _render_markdown(summary, by_case)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        print(f"re-rendered {report_path}", file=sys.stderr)
        return 0

    truth_path = Path(args.truth)
    gallery_dir = Path(args.gallery)

    if not truth_path.exists():
        print(f"truth file not found: {truth_path}", file=sys.stderr)
        return 2
    if not gallery_dir.exists():
        print(f"gallery dir not found: {gallery_dir}", file=sys.stderr)
        return 2

    labels = json.loads(truth_path.read_text())
    from rembg import new_session  # type: ignore
    sess = new_session("u2net")

    by_case: Dict[str, List[Dict[str, Any]]] = {}
    keys = sorted(labels.keys())
    print(f"running {len(keys)} cases × {args.runs} runs each", file=sys.stderr)
    for i, key in enumerate(keys, 1):
        L = labels[key]
        if not L.get("approved"):
            continue
        path = gallery_dir / f"set_{key}.png"
        if not path.exists():
            print(f"  [{i}/{len(keys)}] {key}: gallery PNG missing", file=sys.stderr)
            continue
        user_v = tuple(L["vertex"])
        # Canonical schema uses axis_x/y/z (see FULL_CORNER_LABELING.md
        # "Axis-truth schema convention"). Legacy fixtures may use the
        # old near_x/y/z key set — both name the same 3 FAR-corner
        # positions; only the spelling differs. Read either.
        user_near = [
            tuple(L.get("axis_x", L.get("near_x"))),
            tuple(L.get("axis_y", L.get("near_y"))),
            tuple(L.get("axis_z", L.get("near_z"))),
        ]
        try:
            by_case[key] = _run_one_case(sess, path, user_v, user_near, args.runs)
        except Exception as e:  # noqa: BLE001
            by_case[key] = [{"status": "error", "error": f"{type(e).__name__}: {e}"}]
        if i % 5 == 0 or i == len(keys):
            print(f"  [{i}/{len(keys)}] {key}", file=sys.stderr, flush=True)

    summary = _summarize(by_case)
    payload = {"summary": summary, "by_case": by_case}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)

    report = _render_markdown(summary, by_case)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"wrote {report_path}", file=sys.stderr)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
