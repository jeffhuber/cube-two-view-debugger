#!/usr/bin/env python3
"""Train a learned face-quad vertex regressor on hand-labeled hull data.

Devin's predicted "if RANSAC plateaus" path, confirmed needed by:
  * PR #137 face IoU plateau at 0.666 mean (12% pass at ≥0.85)
  * PR #140 equalize negative result (baseline 97.28% on rectified-with-
    human-quads → bottleneck is geometry, not classification)

Approach (sklearn-only; no PyTorch dependency):

  1. For each of the ~81 hull-labeled (set, side) pairs:
     - Run rembg → mask, compute the cheap angular-sector hexagon
     - Extract input features: 12 floats (6 hexagon vertices, normalized
       to image dimensions). Also include 3 derived features (centroid,
       aspect ratio) → 15-D input.
     - Extract target: 24 floats (3 face quads × 4 corners × 2 coords,
       normalized to image dimensions).
  2. Train sklearn MLPRegressor on (X, y).
  3. Evaluate via leave-one-set-out CV: face IoU + corner pixel error.
  4. Save the trained model + normalization params to .pkl for
     `tools/propose_geometry_labels.py` to load as a new proposer.

Tooling-only, no production-recognizer changes. The trained model lives
under runs/ (gitignored). Drop in as `learned_vertex_hybrid` proposer
in a follow-up edit to propose_geometry_labels.py.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.extract_color_samples import (  # noqa: E402
    discover_additional_tasks,
    load_corpus_tasks,
)
from tools.propose_geometry_labels import (  # noqa: E402
    _fit_hexagon_to_hull,
    _hull_from_mask,
    _get_rembg_session,
)
from tools.sample_stickers_from_hull import (  # noqa: E402
    canonical_corner_order,
    latest_hull_label,
    load_hull_label,
    scaled_face_quads,
)

CORPUS_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "corpus_manifest.json"
DEFAULT_MODEL_OUT = REPO_ROOT / "runs" / "vertex_regressor.pkl"
DEFAULT_REPORT = REPO_ROOT / "runs" / "vertex_regressor_eval.json"
PROCESSING_MAX = 1150
EXPECTED_FACES_BY_SIDE = {"A": ("U", "R", "F"), "B": ("D", "L", "B")}


def load_image_processed(image_path: Path) -> Tuple[Image.Image, int, int]:
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    natural_max = max(image.size)
    if natural_max > PROCESSING_MAX:
        scale = PROCESSING_MAX / float(natural_max)
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return image, image.width, image.height


def rembg_hexagon(image: Image.Image) -> Optional[List[Tuple[float, float]]]:
    """Run rembg + cheap angular-sector hexagon fit, returning 6 vertices
    in canonical CW-from-N order."""
    from rembg import remove
    rgba = remove(image, session=_get_rembg_session("u2net"))
    alpha = np.array(rgba.split()[-1], dtype=np.uint8)
    mask = alpha > 128
    if not mask.any():
        return None
    hull = _hull_from_mask(mask)
    if len(hull) < 6:
        return None
    return _fit_hexagon_to_hull(hull)


def make_features(hexagon: Sequence[Tuple[float, float]], w: int, h: int) -> np.ndarray:
    """15-D feature vector from a 6-vertex hexagon. All coords normalized
    to [0, 1] using image dimensions so the model is image-size-invariant.

    Features:
      * 12 normalized hexagon vertex coords (CW from N)
      * Centroid x, y
      * Hexagon area (as % of image area; rough scale indicator)
    """
    canonical = canonical_corner_order(list(hexagon))
    coords = np.array(canonical, dtype=np.float64)
    coords[:, 0] /= w
    coords[:, 1] /= h
    cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
    # Shoelace-ish area (using normalized coords)
    n = len(coords)
    area = 0.5 * abs(sum(
        coords[i, 0] * coords[(i + 1) % n, 1] - coords[(i + 1) % n, 0] * coords[i, 1]
        for i in range(n)
    ))
    return np.concatenate([coords.flatten(), [cx, cy, area]])


def make_targets(
    face_quads: Dict[str, Sequence[Tuple[float, float]]],
    expected_faces: Sequence[str],
    w: int, h: int,
) -> Optional[np.ndarray]:
    """24-D target vector from 3 face quads. Each face quad → 4 corners
    in canonical CW-from-N order, normalized to [0, 1]. Concatenated in
    the expected_faces order (URF for side A, DLB for side B)."""
    out = []
    for face in expected_faces:
        quad = face_quads.get(face)
        if quad is None or len(quad) != 4:
            return None
        canonical = canonical_corner_order([tuple(p) for p in quad])
        for x, y in canonical:
            out.append(x / w)
            out.append(y / h)
    return np.array(out, dtype=np.float64)


def discover_training_pairs() -> List[Tuple[str, str, Path, Path]]:
    """Returns (setId, side, image_path, hull_label_path) for every
    hull-labeled (set, side) we can train on."""
    tasks = load_corpus_tasks(CORPUS_MANIFEST)
    tasks.extend(discover_additional_tasks({t.set_id for t in tasks}))
    out: List[Tuple[str, str, Path, Path]] = []
    for task in tasks:
        for side, image_path in (("A", task.image_a), ("B", task.image_b)):
            hull_path = latest_hull_label(task.set_id, side)
            if hull_path is None or not image_path.exists():
                continue
            out.append((task.set_id, side, image_path, hull_path))
    return out


def build_dataset() -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """Returns (X, y, identifiers) for sklearn training."""
    pairs = discover_training_pairs()
    print(f"discovered {len(pairs)} hull-labeled (set, side) pairs", file=sys.stderr)
    X_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    ids: List[Tuple[str, str]] = []
    for i, (set_id, side, image_path, hull_path) in enumerate(pairs, 1):
        try:
            image, w, h = load_image_processed(image_path)
            hex_ = rembg_hexagon(image)
            if hex_ is None or len(hex_) != 6:
                print(f"  [{i:>2}/{len(pairs)}] set {set_id}{side}: SKIP (hexagon fit failed)", file=sys.stderr, flush=True)
                continue
            X = make_features(hex_, w, h)
            doc = load_hull_label(hull_path)
            face_quads = scaled_face_quads(doc, w, h)
            y = make_targets(face_quads, EXPECTED_FACES_BY_SIDE[side], w, h)
            if y is None:
                print(f"  [{i:>2}/{len(pairs)}] set {set_id}{side}: SKIP (incomplete face quads)", file=sys.stderr, flush=True)
                continue
            X_rows.append(X)
            y_rows.append(y)
            ids.append((set_id, side))
            print(f"  [{i:>2}/{len(pairs)}] set {set_id}{side}: ok", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"  [{i:>2}/{len(pairs)}] set {set_id}{side}: ERROR {e}", file=sys.stderr, flush=True)
    return np.array(X_rows), np.array(y_rows), ids


def train_and_eval(X: np.ndarray, y: np.ndarray, ids: Sequence[Tuple[str, str]]) -> Dict:
    """Leave-one-set-out CV. For each held-out (setId, side), train on
    everyone else, predict on the held-out, report per-pair errors.
    Then train a final model on all data + report aggregate."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    set_ids_list = sorted(set(s for (s, _) in ids), key=lambda v: int(v) if v.isdigit() else 999)
    print(f"\nleave-one-SET-out CV on {len(set_ids_list)} sets ({len(ids)} samples)", file=sys.stderr)

    rows: List[Dict] = []
    for test_set in set_ids_list:
        test_mask = np.array([s == test_set for (s, _) in ids])
        if test_mask.sum() == 0:
            continue
        X_train, y_train = X[~test_mask], y[~test_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Two candidate models — pick the one with lower CV error per fold.
        # MLP is more powerful but can overfit on 80 samples; Ridge is the
        # safer baseline.
        for model_name, model in (
            ("ridge", Pipeline([("scale", StandardScaler()),
                                ("model", Ridge(alpha=1.0))])),
            ("mlp",   Pipeline([("scale", StandardScaler()),
                                ("model", MLPRegressor(
                                    hidden_layer_sizes=(64, 32),
                                    max_iter=2000, random_state=42,
                                    early_stopping=False))])),
        ):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # Per-pair MSE + max corner error (in normalized [0,1] units)
            for i, (held_set, held_side) in enumerate([(s, sd) for (s, sd) in ids if s == test_set]):
                err = y_pred[i] - y_test[i]
                rows.append({
                    "setId": held_set,
                    "side": held_side,
                    "model": model_name,
                    "mse_normalized": float(np.mean(err ** 2)),
                    "max_corner_err_normalized": float(np.max(np.abs(err))),
                    "mean_corner_err_normalized": float(np.mean(np.abs(err))),
                })

    by_model: Dict[str, List[Dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    aggregate: Dict[str, Dict] = {}
    for m, mrows in by_model.items():
        mse = np.mean([r["mse_normalized"] for r in mrows])
        max_err = np.mean([r["max_corner_err_normalized"] for r in mrows])
        mean_err = np.mean([r["mean_corner_err_normalized"] for r in mrows])
        aggregate[m] = {
            "mean_mse_normalized": round(float(mse), 6),
            "mean_max_corner_err_normalized": round(float(max_err), 4),
            "mean_mean_corner_err_normalized": round(float(mean_err), 4),
            "samples": len(mrows),
        }
    return {"aggregate": aggregate, "perFold": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-out", default=str(DEFAULT_MODEL_OUT))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    args = ap.parse_args()

    X, y, ids = build_dataset()
    print(f"\nbuilt dataset: X={X.shape} y={y.shape}", file=sys.stderr)
    if len(X) < 20:
        print(f"too few training samples ({len(X)}) — need at least 20", file=sys.stderr)
        return 2

    eval_result = train_and_eval(X, y, ids)
    print("\n=== Leave-one-set-out CV ===", file=sys.stderr)
    for model_name, agg in eval_result["aggregate"].items():
        print(f"  {model_name}:", file=sys.stderr)
        for k, v in agg.items():
            print(f"    {k}: {v}", file=sys.stderr)

    # Train final model on ALL data with the best candidate (mlp expected; pick by CV)
    best_model_name = min(
        eval_result["aggregate"],
        key=lambda m: eval_result["aggregate"][m]["mean_mse_normalized"],
    )
    print(f"\nBest CV model: {best_model_name}. Training final on all data.", file=sys.stderr)
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    if best_model_name == "ridge":
        final = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    else:
        final = Pipeline([("scale", StandardScaler()),
                          ("model", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42))])
    final.fit(X, y)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump({
            "model": final,
            "model_name": best_model_name,
            "feature_dim": X.shape[1],
            "target_dim": y.shape[1],
            "training_samples": len(X),
            "training_sets": sorted(set(s for (s, _) in ids), key=lambda v: int(v) if v.isdigit() else 999),
        }, f)
    Path(args.report).write_text(json.dumps(eval_result, indent=2))

    print(f"\nwrote {args.model_out}", file=sys.stderr)
    print(f"wrote {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
