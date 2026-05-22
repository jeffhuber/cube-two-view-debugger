#!/usr/bin/env python3
"""Phase 2B recompute helper — re-run the global model per case to extract
the per-run continuous signals that Codex's spec calls out as the next
layer beyond phase_sep + cv-local status:

  - fit_residual_rms_px                       global fit quality
  - pnp_rms_px                                PnP-specific residual
  - hexagon_centroid_vs_bezel_vertex_offset_px   vertex-ensemble disagreement
                                              proxy (bezel ↔ hexagon centroid)
  - junction_score_at_ensemble                image-space vertex 3-way junction
                                              darkness score
  - ensemble_shift_px                         how far the mean-of-3 ensemble
                                              shifted PnP — proxy for
                                              candidate-vertex disagreement

This module is invoked by `tools/phase2b_trust_matrix.py --recompute-global-model`
to produce `tests/fixtures/phase2b_trust_signal_matrix_recomputed.json` —
the augmented matrix. Separated into its own module to keep the main tool
import-light for users who just want to inspect the existing-signals
matrix (which doesn't need numpy / rembg / Pillow).

Re-running is non-deterministic per Phase 1/Phase 2A history (PnP basin-of-
attraction). The harness runs each case N times (default 2, matching
post_218_baseline.json) and emits per-run records. Categorization (GOOD /
MARGINAL / CATASTROPHIC) uses the same logic as `tools/baseline_post_218.py`.

Caveats:
  - Cases without ground-truth `approved: true` are skipped.
  - Cases whose gallery PNG is missing are skipped with a warning.
  - Model-fit failures are recorded as `status: "model_fit_failed"` with no
    continuous signals; the candidate-rule evaluation must handle that.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GALLERY = Path.home() / "axis_labeling"
DEFAULT_TRUTH = ROOT / "tests" / "fixtures" / "gcm_axis_ground_truth.json"
DEFAULT_OUT = ROOT / "tests" / "fixtures" / "phase2b_recomputed_signals.json"

# Make `tools.*` importable when this script is invoked directly
# (`python3 tools/phase2b_recompute.py`). Python puts the script's dir on
# sys.path[0], not the project root, so `from tools.X import Y` would
# otherwise fail with ModuleNotFoundError.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bearing(o: Tuple[float, float], t: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(t[1] - o[1], t[0] - o[0])) % 360


def _best_perm_err(model_bearings: List[float], user_bearings: List[float]) -> float:
    import itertools
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


def _categorize(err_near: float, err_far: float, chir_status: str) -> str:
    """Same categorization as baseline_post_218.py — re-implemented here to
    avoid importing the whole module (which has its own argparse main, etc.)."""
    if err_near < 10.0:
        return "GOOD"
    if err_near < 25.0:
        return "MARGINAL"
    if err_far < 25.0:
        if chir_status == "corrected_60deg_flip":
            return "CHIRALITY_FALSE_FLIP"
        return "CHIRALITY_MISS"
    return "TRUE_GEOMETRY_FAIL"


def _extract_signals(model_debug: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Pull the continuous trust signals out of model.debug.
    Returns a flat dict with consistent keys (None for missing values)."""
    def _f(key: str) -> Optional[float]:
        v = model_debug.get(key)
        if v is None or v == "n/a":
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    return {
        "fit_residual_rms_px": _f("fit_residual_rms_px"),
        "pnp_rms_px": _f("pnp_rms_px"),
        "hexagon_centroid_vs_bezel_vertex_offset_px": _f("hexagon_centroid_vs_bezel_vertex_offset_px"),
        "bezel_vs_fit_cube_center_offset_px": _f("bezel_vs_fit_cube_center_offset_px"),
        "junction_score_at_ensemble": _f("junction_score_at_ensemble"),
        "ensemble_shift_px": _f("ensemble_shift_px"),
        "ensemble_n_candidates": _f("ensemble_n_candidates"),
        "phase_darkness_separation": _f("phase_darkness_separation"),
    }


def _run_one_case(
    sess: Any,
    img_path: Path,
    user_v: Tuple[float, float],
    user_near: List[Tuple[float, float]],
    n_runs: int,
) -> List[Dict[str, Any]]:
    """Run the global model n_runs times on one case, returning per-run
    records that include the recomputed category + all continuous signals."""
    # Defer heavy imports to runtime so the matrix tool can be imported
    # without numpy/rembg/Pillow.
    import numpy as np
    from PIL import Image
    from rembg import remove
    from tools.global_cube_model import fit_global_cube_model
    from tools.interior_bezel_detection import detect_interior_bezel_lines

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
            results.append({
                "run": run,
                "status": "model_fit_failed",
                "category": "TRUE_GEOMETRY_FAIL",
            })
            continue

        v = model.cube_center_screen
        near = [model.visible_corners[k] for k in ["h_x", "h_y", "h_z"]]
        far = [model.visible_corners[k] for k in ["h_xy", "h_xz", "h_yz"]]
        near_b = sorted([_bearing(v, p) for p in near])
        far_b = sorted([_bearing(v, p) for p in far])
        err_near = _best_perm_err(near_b, user_bearings)
        err_far = _best_perm_err(far_b, user_bearings)
        chir = model.debug.get("phase_check", "?")
        category = _categorize(err_near, err_far, chir)

        signals = _extract_signals(model.debug)
        record = {
            "run": run,
            "status": "ok",
            "err_near_deg": round(err_near, 1),
            "err_far_deg": round(err_far, 1),
            "phase_check": chir,
            "category": category,
        }
        # Round signals for readability; keep None as None.
        for k, v in signals.items():
            record[k] = round(v, 2) if v is not None else None
        results.append(record)
    return results


def recompute_all(
    truth_path: Path = DEFAULT_TRUTH,
    gallery_dir: Path = DEFAULT_GALLERY,
    n_runs: int = 2,
    progress_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Iterate every approved case in the truth file, run the model n times
    each, return the full augmented matrix (same outer shape as
    post_218_baseline.json + new continuous signal fields per run).

    If `progress_path` is given, writes a JSON checkpoint after each case so
    a crash mid-run doesn't lose progress."""
    # Codex #233 round-3 P2: fail-fast on missing gallery_dir. Without this,
    # every approved case logs "PNG missing — skipping" and main() writes an
    # empty 0-cases fixture that `--recompute-global-model` would happily
    # consume, silently overwriting the benchmark with no-data.
    if not gallery_dir.exists():
        raise FileNotFoundError(
            f"gallery_dir does not exist: {gallery_dir}. "
            f"Run from a machine with the axis-labeling gallery, or pass "
            f"--gallery to a valid path."
        )

    from rembg import new_session  # type: ignore

    labels = json.loads(truth_path.read_text())
    sess = new_session("u2net")

    by_case: Dict[str, List[Dict[str, Any]]] = {}
    keys = sorted(labels.keys())
    approved_keys = [k for k in keys if labels[k].get("approved")]
    print(f"recomputing {len(approved_keys)} cases × {n_runs} runs each "
          f"(of {len(keys)} total in truth)", file=sys.stderr, flush=True)

    t_total_start = time.time()
    for i, key in enumerate(approved_keys, 1):
        L = labels[key]
        path = gallery_dir / f"set_{key}.png"
        if not path.exists():
            print(f"  [{i}/{len(approved_keys)}] {key}: PNG missing — skipping", file=sys.stderr)
            continue
        user_v = tuple(L["vertex"])
        user_near = [tuple(L["near_x"]), tuple(L["near_y"]), tuple(L["near_z"])]
        t_case_start = time.time()
        try:
            by_case[key] = _run_one_case(sess, path, user_v, user_near, n_runs)
            dt = time.time() - t_case_start
            elapsed_total = time.time() - t_total_start
            avg = elapsed_total / i
            eta = avg * (len(approved_keys) - i)
            print(f"  [{i}/{len(approved_keys)}] {key} ({dt:.1f}s, "
                  f"avg {avg:.1f}s/case, ETA {eta/60:.1f}min)",
                  file=sys.stderr, flush=True)
        except Exception as e:  # noqa: BLE001
            by_case[key] = [{"status": "error", "error": f"{type(e).__name__}: {e}"}]
            print(f"  [{i}/{len(approved_keys)}] {key} ERROR: {e}",
                  file=sys.stderr, flush=True)

        # Checkpoint after each case for crash safety.
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps({"by_case": by_case}, indent=2))

    # Codex #233 round-3 P2: fail-fast if nothing got processed. This guards
    # against the gallery_dir existing but being empty / mis-named.
    if approved_keys and not by_case:
        raise RuntimeError(
            f"recompute processed 0 cases out of {len(approved_keys)} approved. "
            f"Check that gallery PNGs exist as set_<key>.png under {gallery_dir}."
        )

    summary = _summarize(by_case)

    # Codex #233 round-4/6 P2: fail-fast if zero cases yielded a successful
    # fit. `summary["n_runs"]` counts everything with a `category` field
    # (including `status="model_fit_failed"` rows, which `_run_one_case`
    # marks with category=TRUE_GEOMETRY_FAIL) — so it's NOT a clean proxy
    # for "did any model actually fit". Count status=="ok" separately.
    n_ok = sum(
        1 for runs in by_case.values()
        for r in runs if r.get("status") == "ok"
    )
    if approved_keys and n_ok == 0:
        n_errors = sum(
            1 for runs in by_case.values()
            for r in runs if r.get("status") == "error"
        )
        n_failed = sum(
            1 for runs in by_case.values()
            for r in runs if r.get("status") == "model_fit_failed"
        )
        raise RuntimeError(
            f"recompute produced 0 successful fits out of {len(approved_keys)} "
            f"approved cases (errors: {n_errors}, model_fit_failed: {n_failed}). "
            f"Diagnose recompute before trusting the matrix; refusing to write "
            f"a fixture with no continuous-signal data."
        )

    return {"summary": summary, "by_case": by_case}


def _summarize(by_case: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    from collections import Counter
    cat_counts: Counter = Counter()
    n_runs = 0
    for runs in by_case.values():
        for r in runs:
            if "category" in r:
                cat_counts[r["category"]] += 1
                n_runs += 1
    return {
        "n_cases": len(by_case),
        "n_runs": n_runs,
        "category_counts": dict(cat_counts),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", default=str(DEFAULT_TRUTH))
    ap.add_argument("--gallery", default=str(DEFAULT_GALLERY))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--progress", default=None,
                    help="Optional progress-checkpoint JSON path (writes after "
                         "each case; useful for crash recovery on long runs).")
    args = ap.parse_args()

    payload = recompute_all(
        truth_path=Path(args.truth),
        gallery_dir=Path(args.gallery),
        n_runs=args.runs,
        progress_path=Path(args.progress) if args.progress else None,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out.relative_to(ROOT) if out.is_absolute() and ROOT in out.parents else out} "
          f"({payload['summary']['n_runs']} runs across "
          f"{payload['summary']['n_cases']} cases)",
          file=sys.stderr)
    print(f"categories: {payload['summary']['category_counts']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
