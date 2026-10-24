"""Evaluation helpers: ranking quality, probability quality and threshold trade-offs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             log_loss, precision_recall_curve, roc_auc_score, roc_curve)


def score_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Threshold-free metrics. Brier of a constant base-rate predictor is the reference."""
    y = np.asarray(y)
    base = float(np.mean(y))
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "brier_baseline": float(brier_score_loss(y, np.full(len(y), base))),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "ece": expected_calibration_error(y, p),
        "mean_predicted": float(np.mean(p)),
        "observed_rate": base,
    }


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold), "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "flagged_rate": float(pred.mean()), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Quantile-binned reliability table: mean predicted vs observed default rate."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    rows = []
    for idx in np.array_split(order, n_bins):
        if len(idx):
            rows.append({"mean_predicted": float(p[idx].mean()), "observed_rate": float(y[idx].mean()),
                         "count": int(len(idx))})
    return rows


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    table = calibration_table(y, p, n_bins)
    n = sum(r["count"] for r in table)
    return float(sum(r["count"] / n * abs(r["mean_predicted"] - r["observed_rate"]) for r in table))


def curve_points(y: np.ndarray, p: np.ndarray, max_points: int = 200) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    s1, s2 = max(1, len(fpr) // max_points), max(1, len(prec) // max_points)
    return {"roc": {"fpr": fpr[::s1].tolist(), "tpr": tpr[::s1].tolist()},
            "pr": {"recall": rec[::s2].tolist(), "precision": prec[::s2].tolist()}}


def threshold_sweep(y: np.ndarray, p: np.ndarray, n: int = 50) -> list[dict]:
    grid = np.unique(np.quantile(p, np.linspace(0.02, 0.98, n)))
    return [threshold_metrics(y, p, t) for t in grid]


def select_threshold(y: np.ndarray, p: np.ndarray, policy: str, target_recall: float = 0.60,
                     cost_fn: float = 5.0, cost_fp: float = 1.0) -> dict:
    """Pick an operating threshold on validation data. Returns threshold + the assumption used."""
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 400)))
    rows = [threshold_metrics(y, p, t) for t in grid]
    if policy == "recall_target":
        ok = [r for r in rows if r["recall"] >= target_recall]
        best = max(ok, key=lambda r: r["precision"]) if ok else min(rows, key=lambda r: abs(r["recall"] - target_recall))
        assumption = f"highest precision subject to recall >= {target_recall:.2f} on the validation split"
    elif policy == "cost":
        best = min(rows, key=lambda r: cost_fn * r["fn"] + cost_fp * r["fp"])
        assumption = f"minimise {cost_fn:g} x false negatives + {cost_fp:g} x false positives on the validation split"
    else:
        raise ValueError("policy must be 'recall_target' or 'cost'")
    return {"threshold": best["threshold"], "policy": policy, "assumption": assumption}


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric=roc_auc_score, n: int = 200, seed: int = 0) -> list[float]:
    """95% bootstrap interval of a metric on a fixed set of predictions."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(metric(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def permutation_importance(predict_fn, X, y, features, n_repeats: int = 3, seed: int = 0) -> list[dict]:
    """Drop in ROC-AUC when one raw input column is shuffled (works for any model).

    Computed on held-out rows; it measures what the *model* relies on, not causal effect."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    base = roc_auc_score(y, predict_fn(X))
    rows = []
    for col in features:
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(base - roc_auc_score(y, predict_fn(shuffled)))
        rows.append({"feature": col, "auc_drop_mean": float(np.mean(drops)), "auc_drop_std": float(np.std(drops))})
    return sorted(rows, key=lambda r: r["auc_drop_mean"], reverse=True)
