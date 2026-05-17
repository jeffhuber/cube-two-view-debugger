#!/usr/bin/env python3
"""Train and A/B-evaluate candidate color classifiers against the current
``classify_rgb`` baseline, using leave-one-set-out cross-validation.

No new runtime deps: all candidates are implemented in numpy. The "winning"
candidate's parameters can be hand-extracted (centroids, coefficients) and
shipped as constants in ``rubik_recognizer/colors.py``.

Candidates evaluated:
  * baseline_canonical: current ``classify_rgb`` with default canonical palette
  * centroid_lab:       per-color centroid in (L, a, b), classify by nearest
  * centroid_lab_hsv:   per-color centroid in (L, a, b, H*scaled, S, V)
  * knn_lab(k=5):       k-nearest-neighbor in (L, a, b) with majority vote
  * knn_lab_hsv(k=7):   k-nearest-neighbor in 6-feature space
  * softmax_lab_hsv:    multinomial logistic regression (numpy gradient descent)

Each candidate is evaluated under leave-one-set-out CV: hold all samples
from one set as the test fold; train on the rest; aggregate predictions.

Reports:
  * overall accuracy per candidate
  * per-color precision/recall
  * confusion matrix vs ground truth
  * head-to-head delta vs baseline per set
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rubik_recognizer.colors import classify_rgb, rgb_to_hsv, rgb_to_lab  # noqa: E402


COLOR_ORDER = ("white", "yellow", "red", "orange", "green", "blue")
COLOR_INDEX = {c: i for i, c in enumerate(COLOR_ORDER)}
DEFAULT_INPUT = REPO_ROOT / "runs" / "color_samples_v0.jsonl"
DEFAULT_REPORT = REPO_ROOT / "runs" / "color_classifier_report.json"


# ---------- featurization ----------


def lab_features(rgbs: np.ndarray) -> np.ndarray:
    """rgbs: (N, 3) uint8 → (N, 3) float Lab."""
    return np.array([rgb_to_lab(tuple(int(v) for v in row)) for row in rgbs], dtype=np.float64)


def hsv_features(rgbs: np.ndarray) -> np.ndarray:
    """rgbs: (N, 3) uint8 → (N, 3) float HSV (each in [0,1])."""
    return np.array([rgb_to_hsv(tuple(int(v) for v in row)) for row in rgbs], dtype=np.float64)


def lab_hsv_features(rgbs: np.ndarray) -> np.ndarray:
    """Combined (L, a, b, H_wrapped_cos, H_wrapped_sin, S, V).
    Hue is on a circle so we encode as (cos, sin) to handle the 0/1 wrap."""
    lab = lab_features(rgbs)
    hsv = hsv_features(rgbs)
    h = hsv[:, 0:1]
    s = hsv[:, 1:2]
    v = hsv[:, 2:3]
    h_cos = np.cos(2 * np.pi * h)
    h_sin = np.sin(2 * np.pi * h)
    # Scale Lab roughly to similar magnitude as HSV
    lab_scaled = lab / np.array([100.0, 128.0, 128.0])
    return np.hstack([lab_scaled, h_cos, h_sin, s, v])


# ---------- candidate models ----------


class BaselineCanonical:
    name = "baseline_canonical"

    def fit(self, X_rgb: np.ndarray, y: np.ndarray) -> None:
        pass  # nothing to fit

    def predict(self, X_rgb: np.ndarray) -> np.ndarray:
        out = np.empty(len(X_rgb), dtype=np.int64)
        for i, row in enumerate(X_rgb):
            color = classify_rgb(tuple(int(v) for v in row)).color
            out[i] = COLOR_INDEX[color]
        return out


class CentroidClassifier:
    def __init__(self, feature_fn, name: str):
        self.feature_fn = feature_fn
        self.name = name

    def fit(self, X_rgb: np.ndarray, y: np.ndarray) -> None:
        X = self.feature_fn(X_rgb)
        self.centroids = np.zeros((len(COLOR_ORDER), X.shape[1]), dtype=np.float64)
        for ci in range(len(COLOR_ORDER)):
            mask = (y == ci)
            if mask.any():
                self.centroids[ci] = X[mask].mean(axis=0)
            else:
                self.centroids[ci] = np.nan

    def predict(self, X_rgb: np.ndarray) -> np.ndarray:
        X = self.feature_fn(X_rgb)
        # squared euclidean to each centroid
        d = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d, axis=1)


class KNNClassifier:
    def __init__(self, feature_fn, k: int, name: str):
        self.feature_fn = feature_fn
        self.k = k
        self.name = name

    def fit(self, X_rgb: np.ndarray, y: np.ndarray) -> None:
        self.X_train = self.feature_fn(X_rgb)
        self.y_train = y
        # Standardize for distance comparability
        self.mu = self.X_train.mean(axis=0)
        self.sigma = self.X_train.std(axis=0) + 1e-9
        self.X_train_std = (self.X_train - self.mu) / self.sigma

    def predict(self, X_rgb: np.ndarray) -> np.ndarray:
        X = (self.feature_fn(X_rgb) - self.mu) / self.sigma
        out = np.empty(len(X), dtype=np.int64)
        for i, row in enumerate(X):
            d = ((self.X_train_std - row) ** 2).sum(axis=1)
            nearest = np.argpartition(d, kth=min(self.k, len(d) - 1))[:self.k]
            counts = np.bincount(self.y_train[nearest], minlength=len(COLOR_ORDER))
            out[i] = int(np.argmax(counts))
        return out


class SoftmaxClassifier:
    """Multinomial logistic regression via numpy gradient descent."""

    def __init__(self, feature_fn, name: str, lr: float = 0.3, epochs: int = 600, l2: float = 1e-3):
        self.feature_fn = feature_fn
        self.name = name
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2

    def fit(self, X_rgb: np.ndarray, y: np.ndarray) -> None:
        X = self.feature_fn(X_rgb)
        # Standardize
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-9
        Xs = (X - self.mu) / self.sigma
        # Add bias column
        Xs = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        K = len(COLOR_ORDER)
        N, D = Xs.shape
        W = np.zeros((D, K))
        Y_onehot = np.zeros((N, K))
        Y_onehot[np.arange(N), y] = 1.0

        for _ in range(self.epochs):
            logits = Xs @ W  # (N, K)
            logits -= logits.max(axis=1, keepdims=True)  # stability
            exps = np.exp(logits)
            probs = exps / exps.sum(axis=1, keepdims=True)
            grad = Xs.T @ (probs - Y_onehot) / N + self.l2 * W
            W -= self.lr * grad
        self.W = W

    def predict(self, X_rgb: np.ndarray) -> np.ndarray:
        X = (self.feature_fn(X_rgb) - self.mu) / self.sigma
        Xs = np.hstack([X, np.ones((X.shape[0], 1))])
        logits = Xs @ self.W
        return np.argmax(logits, axis=1)


# ---------- evaluation ----------


def load_samples(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    rgbs: List[List[int]] = []
    ys: List[int] = []
    set_ids: List[str] = []
    current_colors: List[str] = []
    with path.open() as f:
        for line in f:
            sample = json.loads(line)
            rgbs.append(sample["rgb"])
            ys.append(COLOR_INDEX[sample["gtColor"]])
            set_ids.append(sample["setId"])
            current_colors.append(sample.get("currentColor") or sample.get("defaultClassifier"))
    X = np.array(rgbs, dtype=np.float64)
    y = np.array(ys, dtype=np.int64)
    return X, y, set_ids, current_colors


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    K = len(COLOR_ORDER)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    K = cm.shape[0]
    for k in range(K):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out[COLOR_ORDER[k]] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(cm[k, :].sum()),
        }
    return out


def leave_one_set_out_eval(
    candidate_factory,
    X: np.ndarray,
    y: np.ndarray,
    set_ids: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return (predictions in original order, per-set accuracy)."""
    unique_sets = sorted(set(set_ids), key=lambda s: int(s) if s.isdigit() else s)
    preds = np.full(len(y), -1, dtype=np.int64)
    per_set_acc: Dict[str, float] = {}
    for s in unique_sets:
        test_mask = np.array([sid == s for sid in set_ids])
        train_mask = ~test_mask
        model = candidate_factory()
        model.fit(X[train_mask], y[train_mask])
        p = model.predict(X[test_mask])
        preds[test_mask] = p
        per_set_acc[s] = float((p == y[test_mask]).mean())
    return preds, per_set_acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    args = ap.parse_args()

    X, y, set_ids, current_colors = load_samples(Path(args.input))
    print(f"loaded {len(y)} samples across {len(set(set_ids))} sets", file=sys.stderr)

    candidates: list = [
        ("baseline_canonical",   lambda: BaselineCanonical()),
        ("centroid_lab",         lambda: CentroidClassifier(lab_features, "centroid_lab")),
        ("centroid_lab_hsv",     lambda: CentroidClassifier(lab_hsv_features, "centroid_lab_hsv")),
        ("knn5_lab",             lambda: KNNClassifier(lab_features, 5, "knn5_lab")),
        ("knn7_lab_hsv",         lambda: KNNClassifier(lab_hsv_features, 7, "knn7_lab_hsv")),
        ("knn15_lab_hsv",        lambda: KNNClassifier(lab_hsv_features, 15, "knn15_lab_hsv")),
        ("softmax_lab_hsv",      lambda: SoftmaxClassifier(lab_hsv_features, "softmax_lab_hsv")),
    ]

    # If sklearn is available, add stronger candidates as a sanity check on
    # the numpy implementations + try tree-based models that can exploit
    # non-linear feature interactions.
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # noqa: E402
        from sklearn.linear_model import LogisticRegression  # noqa: E402
        from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
        from sklearn.preprocessing import StandardScaler  # noqa: E402

        class SklearnAdapter:
            def __init__(self, feature_fn, factory, name: str, scale: bool = True):
                self.feature_fn = feature_fn
                self.factory = factory
                self.name = name
                self.scale = scale

            def fit(self, X_rgb: np.ndarray, y: np.ndarray) -> None:
                X = self.feature_fn(X_rgb)
                if self.scale:
                    self.scaler = StandardScaler().fit(X)
                    X = self.scaler.transform(X)
                self.model = self.factory()
                self.model.fit(X, y)

            def predict(self, X_rgb: np.ndarray) -> np.ndarray:
                X = self.feature_fn(X_rgb)
                if self.scale:
                    X = self.scaler.transform(X)
                return self.model.predict(X)

        candidates.extend([
            ("sk_logreg_lab_hsv",
             lambda: SklearnAdapter(lab_hsv_features,
                                    lambda: LogisticRegression(max_iter=2000, C=1.0),
                                    "sk_logreg_lab_hsv")),
            ("sk_knn15_lab_hsv",
             lambda: SklearnAdapter(lab_hsv_features,
                                    lambda: KNeighborsClassifier(n_neighbors=15),
                                    "sk_knn15_lab_hsv")),
            ("sk_rf_200_lab_hsv",
             lambda: SklearnAdapter(lab_hsv_features,
                                    lambda: RandomForestClassifier(n_estimators=200, max_depth=None, random_state=0, n_jobs=-1),
                                    "sk_rf_200_lab_hsv", scale=False)),
            ("sk_gb_200_lab_hsv",
             lambda: SklearnAdapter(lab_hsv_features,
                                    lambda: GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0),
                                    "sk_gb_200_lab_hsv", scale=False)),
        ])
        print("sklearn available — added 4 reference candidates", file=sys.stderr)
    except ImportError:
        print("sklearn not available — running numpy candidates only", file=sys.stderr)

    report: Dict[str, dict] = {"samples": int(len(y)), "n_sets": len(set(set_ids)), "candidates": {}}

    for name, factory in candidates:
        print(f"\n=== {name} ===", file=sys.stderr)
        if name == "baseline_canonical":
            # No CV needed — it's a fixed function. Predict on all samples.
            preds = factory().predict(X)
            per_set_acc = {}
            for s in sorted(set(set_ids), key=lambda x: int(x) if x.isdigit() else x):
                mask = np.array([sid == s for sid in set_ids])
                per_set_acc[s] = float((preds[mask] == y[mask]).mean())
        else:
            preds, per_set_acc = leave_one_set_out_eval(factory, X, y, set_ids)

        acc = float((preds == y).mean())
        cm = confusion_matrix(y, preds)
        per_color = per_class_metrics(cm)
        print(f"overall accuracy: {acc:.4f}", file=sys.stderr)
        for color, m in per_color.items():
            print(
                f"  {color:7s}  precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}  support={m['support']}",
                file=sys.stderr,
            )

        report["candidates"][name] = {
            "overall_accuracy": round(acc, 4),
            "per_color": per_color,
            "confusion_matrix": cm.tolist(),
            "confusion_labels": list(COLOR_ORDER),
            "per_set_accuracy": {k: round(v, 4) for k, v in per_set_acc.items()},
        }

    # Head-to-head: how often does the best non-baseline beat baseline per set?
    baseline_acc = {s: report["candidates"]["baseline_canonical"]["per_set_accuracy"][s]
                    for s in report["candidates"]["baseline_canonical"]["per_set_accuracy"]}
    best_name = max(
        (n for n in report["candidates"] if n != "baseline_canonical"),
        key=lambda n: report["candidates"][n]["overall_accuracy"],
    )
    best_acc = report["candidates"][best_name]["per_set_accuracy"]
    print(f"\n=== head-to-head: {best_name} vs baseline_canonical ===", file=sys.stderr)
    wins = losses = ties = 0
    per_set_delta = {}
    for s, b in baseline_acc.items():
        c = best_acc.get(s, 0.0)
        d = c - b
        per_set_delta[s] = round(d, 4)
        if d > 1e-9: wins += 1
        elif d < -1e-9: losses += 1
        else: ties += 1
    print(f"  per-set: {wins} wins, {losses} losses, {ties} ties", file=sys.stderr)
    sorted_delta = sorted(per_set_delta.items(), key=lambda kv: kv[1])
    if sorted_delta:
        print(f"  worst regressions: {sorted_delta[:3]}", file=sys.stderr)
        print(f"  biggest wins:      {sorted_delta[-3:]}", file=sys.stderr)
    report["head_to_head"] = {
        "best_candidate": best_name,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "per_set_delta": per_set_delta,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote report to {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
