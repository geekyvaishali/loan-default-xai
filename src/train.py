"""Reproducible training workflow.   Run:  python -m src.train

Data usage (no split influences another):
  train (60%)  -> fit every candidate; 3-fold CV *inside* train for model comparison
  val   (20%)  -> fit the probability calibrator and choose the decision threshold
  test  (20%)  -> evaluated once for the selected model; never used for any choice
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.base import clone

from . import evaluate as ev
from .config import (ARTIFACT_DIR, CATEGORICAL_COLS, NUMERIC_COLS, RAW_DATA_PATH, RAW_FEATURES,
                     TARGET, TrainConfig)
from .pipeline import (COMPLEXITY_ORDER, CalibratedRiskModel, ProbabilityCalibrator,
                       available_models, build_pipeline)
from .schema import (ALLOWED_CATEGORIES, HARD_BOUNDS, TRAINING_RANGE, data_quality_report,
                     validate_frame)

log = logging.getLogger("loan_risk.train")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_data(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig):
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y)
    val_frac = cfg.val_size / (1 - cfg.test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, random_state=cfg.random_state, stratify=y_tmp)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def choose_model(cv_results: dict[str, dict], metric: str, one_se: bool) -> tuple[str, str]:
    """Best CV mean; with the one-SE rule, the simplest model within 1 std of the best."""
    key = f"cv_{metric}_mean"
    best = max(cv_results, key=lambda m: cv_results[m][key])
    if not one_se:
        return best, f"highest CV {metric} ({cv_results[best][key]:.4f})"
    floor = cv_results[best][key] - cv_results[best][f"cv_{metric}_std"]
    eligible = [m for m in cv_results if cv_results[m][key] >= floor]
    pick = min(eligible, key=COMPLEXITY_ORDER.index)
    reason = (f"best CV {metric} is {best} ({cv_results[best][key]:.4f}); one-standard-error rule picked the simplest "
              f"model within 1 std of it ({', '.join(eligible)} eligible)")
    return pick, reason


def _reference_stats(X_train: pd.DataFrame, val_probs: np.ndarray) -> dict:
    """Training-time distributions used later for drift checks (no raw rows are stored)."""
    numeric = {c: {"edges": np.unique(np.quantile(X_train[c], np.linspace(0, 1, 11))).tolist()} for c in NUMERIC_COLS}
    categorical = {c: X_train[c].value_counts(normalize=True).round(6).to_dict() for c in CATEGORICAL_COLS}
    return {"numeric": numeric, "categorical": categorical,
            "score_quantiles": np.quantile(val_probs, np.linspace(0, 1, 21)).tolist()}


def run_training(cfg: TrainConfig | None = None, data_path: Path = RAW_DATA_PATH,
                 artifact_dir: Path = ARTIFACT_DIR) -> dict:
    cfg = cfg or TrainConfig()
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    df = pd.read_csv(data_path)
    dq = data_quality_report(df)
    log.info("Loaded %s rows | default rate %.4f | duplicates %s | looks_synthetic=%s",
             dq["n_rows"], dq["default_rate"], dq["n_duplicate_rows_excluding_id"], dq["looks_synthetic"])
    report = validate_frame(df)
    if report.missing_columns:
        raise ValueError(f"Training data is missing columns: {report.missing_columns}")
    if report.row_errors:
        bad = set(report.invalid_positions)
        log.warning("Dropping %d rows that fail hard validation", len(bad))
        df = df.drop(df.index[sorted(bad)])
    X, y = df[RAW_FEATURES], df[TARGET].astype(int)

    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(X, y, cfg)
    log.info("Split sizes train/val/test = %d / %d / %d", len(X_tr), len(X_va), len(X_te))

    names, skipped = available_models(cfg.models, cfg.random_state, cfg.n_jobs)
    for n, why in skipped.items():
        log.warning("Skipping %s: %s", n, why)

    candidates: dict[str, dict] = {}
    fitted = {}
    for name in names:
        t0 = time.time()
        pipe = build_pipeline(name, cfg.random_state, cfg.n_jobs)
        cv = cross_validate(clone(pipe), X_tr, y_tr, cv=cfg.cv_folds, n_jobs=1,
                            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"})
        pipe.fit(X_tr, y_tr)
        raw_val = pipe.predict_proba(X_va)[:, 1]
        fitted[name] = pipe
        candidates[name] = {
            "cv_roc_auc_mean": float(cv["test_roc_auc"].mean()), "cv_roc_auc_std": float(cv["test_roc_auc"].std()),
            "cv_pr_auc_mean": float(cv["test_pr_auc"].mean()), "cv_pr_auc_std": float(cv["test_pr_auc"].std()),
            "val": ev.score_metrics(y_va, raw_val), "seconds": round(time.time() - t0, 1),
        }
        log.info("%-24s CV AUC %.4f±%.4f | val AUC %.4f PR-AUC %.4f Brier %.4f (%.0fs)", name,
                 candidates[name]["cv_roc_auc_mean"], candidates[name]["cv_roc_auc_std"],
                 candidates[name]["val"]["roc_auc"], candidates[name]["val"]["pr_auc"],
                 candidates[name]["val"]["brier"], candidates[name]["seconds"])

    selected, reason = choose_model(candidates, cfg.selection_metric, cfg.one_se_rule)
    log.info("Selected model: %s | %s", selected, reason)

    # --- calibrate + choose threshold on validation only ---
    pipe = fitted[selected]
    raw_val = pipe.predict_proba(X_va)[:, 1]
    calibrator = ProbabilityCalibrator(cfg.calibration_method).fit(raw_val, y_va.to_numpy())
    model = CalibratedRiskModel(pipe, calibrator, selected)
    cal_val = model.predict_proba(X_va)[:, 1]
    thr = ev.select_threshold(y_va, cal_val, cfg.threshold_policy, cfg.target_recall, cfg.cost_fn, cfg.cost_fp)

    # --- single, final test evaluation ---
    raw_te = model.predict_proba_raw(X_te)
    cal_te = model.predict_proba(X_te)[:, 1]
    test = {
        "raw": ev.score_metrics(y_te, raw_te),
        "calibrated": ev.score_metrics(y_te, cal_te),
        "at_threshold": ev.threshold_metrics(y_te, cal_te, thr["threshold"]),
        "roc_auc_ci95": ev.bootstrap_ci(y_te, cal_te),
        "curves": ev.curve_points(y_te, cal_te),
        "calibration_table_raw": ev.calibration_table(y_te, raw_te),
        "calibration_table_calibrated": ev.calibration_table(y_te, cal_te),
        "threshold_sweep": ev.threshold_sweep(y_te, cal_te),
        "n": int(len(y_te)),
    }
    baseline = None
    if selected != "logistic_regression" and "logistic_regression" in fitted:
        b = fitted["logistic_regression"].predict_proba(X_te)[:, 1]
        baseline = {"model": "logistic_regression", **ev.score_metrics(y_te, b)}

    # --- metadata / artifacts ---
    data_hash = _sha256(Path(data_path))
    version = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{data_hash[:7]}"
    metadata = {
        "model_version": version, "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": selected, "selection_reason": reason,
        "threshold": thr, "calibration_method": cfg.calibration_method,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        "skipped_models": skipped,
        "data": {"file": Path(data_path).name, "sha256": data_hash, "n_rows": int(len(df)),
                 "default_rate": float(y.mean()), "looks_synthetic": dq["looks_synthetic"]},
        "splits": {"train": int(len(X_tr)), "val": int(len(X_va)), "test": int(len(X_te)),
                   "train_default_rate": float(y_tr.mean()), "val_default_rate": float(y_va.mean()),
                   "test_default_rate": float(y_te.mean())},
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
        "training_seconds": round(time.time() - t_start, 1),
        "headline_test_metrics": {k: test["calibrated"][k] for k in ("roc_auc", "pr_auc", "brier", "ece")},
    }
    out_names = list(pipe[:-1].transform(X_tr.head(2)).columns)
    explain_info = {
        "reference_values": {**{c: float(X_tr[c].median()) for c in NUMERIC_COLS},
                             **{c: str(X_tr[c].mode().iloc[0]) for c in CATEGORICAL_COLS}},
    }
    if selected == "logistic_regression":
        z_train = pipe[:-1].transform(X_tr)
        explain_info["transformed_means"] = {c: float(v) for c, v in z_train.mean().items()}
    sample = X_va.sample(min(5000, len(X_va)), random_state=cfg.random_state)
    global_importance = ev.permutation_importance(
        lambda d: model.predict_proba(d)[:, 1], sample, y_va.loc[sample.index], RAW_FEATURES,
        n_repeats=3, seed=cfg.random_state)
    schema = {"raw_features": RAW_FEATURES, "numeric": NUMERIC_COLS, "categorical": CATEGORICAL_COLS,
              "hard_bounds": HARD_BOUNDS, "training_range": TRAINING_RANGE,
              "allowed_categories": ALLOWED_CATEGORIES, "model_input_columns": out_names,
              "explain": explain_info}
    evaluation = {"candidates": candidates, "selected": selected, "test": test,
                  "global_importance": global_importance,
                  "global_importance_note": f"Permutation importance (ROC-AUC drop) on {len(sample)} validation rows.",
                  "baseline_logistic_regression_test": baseline,
                  "validation_selected": {"raw": ev.score_metrics(y_va, raw_val),
                                          "calibrated": ev.score_metrics(y_va, cal_val)}}

    joblib.dump(model, artifact_dir / "model.joblib")
    for fname, obj in [("metadata.json", metadata), ("schema.json", schema), ("evaluation.json", evaluation),
                       ("data_quality.json", dq), ("reference_stats.json", _reference_stats(X_tr, cal_val))]:
        (artifact_dir / fname).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    with open(artifact_dir / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"model_version": version, "trained_at_utc": metadata["trained_at_utc"],
                            "model": selected, **metadata["headline_test_metrics"],
                            "threshold": thr["threshold"]}) + "\n")
    (artifact_dir / "model_card.md").write_text(render_model_card(metadata, evaluation, dq), encoding="utf-8")
    log.info("Artifacts written to %s (version %s)", artifact_dir, version)
    return {"metadata": metadata, "evaluation": evaluation}


def render_model_card(meta: dict, evaluation: dict, dq: dict) -> str:
    t = evaluation["test"]
    cal, raw, thr = t["calibrated"], t["raw"], t["at_threshold"]
    lines = [
        "# Model card: Loan Default Risk (educational)", "",
        f"**Version:** {meta['model_version']}  |  **Trained (UTC):** {meta['trained_at_utc']}",
        f"**Algorithm:** {meta['selected_model']} (+ {meta['calibration_method']} probability calibration)",
        f"**Why this model:** {meta['selection_reason']}", "",
        "## Data",
        f"- {meta['data']['n_rows']:,} rows, default rate {meta['data']['default_rate']:.2%}; file `{meta['data']['file']}` "
        f"(sha256 `{meta['data']['sha256'][:12]}...`).",
        f"- Split: train {meta['splits']['train']:,} / validation {meta['splits']['val']:,} / test {meta['splits']['test']:,} (stratified, seed {meta['config']['random_state']}).",
        f"- Duplicate rows: {dq['n_duplicate_rows_excluding_id']}; null cells: {dq['n_null_cells']}.",
        f"- **Synthetic-data warning:** numeric columns are almost perfectly uniform (flatness check: {dq['looks_synthetic']}). "
        "This strongly suggests a synthetic dataset, so results do not describe real lending behaviour.", "",
        "## Test-set results (touched once, after all choices were made)",
        f"- ROC-AUC {cal['roc_auc']:.4f} (95% bootstrap CI {t['roc_auc_ci95'][0]:.4f} to {t['roc_auc_ci95'][1]:.4f}); PR-AUC {cal['pr_auc']:.4f} "
        f"(a no-skill model scores about {cal['observed_rate']:.3f}).",
        f"- Brier {cal['brier']:.4f} vs {cal['brier_baseline']:.4f} for always predicting the base rate; ECE {cal['ece']:.4f} "
        f"(before calibration: Brier {raw['brier']:.4f}, ECE {raw['ece']:.4f}).",
        f"- At threshold {thr['threshold']:.4f} ({meta['threshold']['assumption']}): precision {thr['precision']:.3f}, "
        f"recall {thr['recall']:.3f}, F1 {thr['f1']:.3f}, flagged {thr['flagged_rate']:.1%} of applicants.", "",
        ("- Calibration changed little: the model was already close to calibrated (Brier moved by "
         f"{abs(raw['brier'] - cal['brier']):.4f}). It is kept because it is cheap, preserves ranking, and matters more for tree models."
         if abs(raw["brier"] - cal["brier"]) < 0.0005 else
         "- Calibration materially changed the probabilities; compare the raw and calibrated rows above."), "",
        "## Intended use", "Portfolio / education only. Not validated for real credit decisions, not regulatory-compliant.", "",
        "## Limitations",
        "- Hyperparameters are fixed, regularised values, not extensively tuned; model differences are within noise (see CV std).",
        "- Age, marital status and education are model inputs. Many jurisdictions restrict such attributes in credit decisions; "
        "no fairness audit was possible here (no protected-group labels, no outcome data by group).",
        "- Single static snapshot: no time-based validation, no drift history, no real-world outcome feedback.",
        "- Probabilities are calibrated on one validation split of this dataset only.", ""]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train the loan default risk model")
    ap.add_argument("--data", default=str(RAW_DATA_PATH))
    ap.add_argument("--out", default=str(ARTIFACT_DIR))
    ap.add_argument("--policy", choices=["recall_target", "cost"], default=TrainConfig.threshold_policy)
    ap.add_argument("--target-recall", type=float, default=TrainConfig.target_recall)
    ap.add_argument("--cost-fn", type=float, default=TrainConfig.cost_fn)
    ap.add_argument("--cost-fp", type=float, default=TrainConfig.cost_fp)
    ap.add_argument("--calibration", choices=["isotonic", "sigmoid"], default=TrainConfig.calibration_method)
    ap.add_argument("--seed", type=int, default=TrainConfig.random_state)
    a = ap.parse_args()
    cfg = TrainConfig(random_state=a.seed, threshold_policy=a.policy, target_recall=a.target_recall,
                      cost_fn=a.cost_fn, cost_fp=a.cost_fp, calibration_method=a.calibration)
    run_training(cfg, Path(a.data), Path(a.out))


if __name__ == "__main__":
    main()
