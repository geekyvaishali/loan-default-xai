"""
PHASE 3 — Model Training + Tuning (Days 5-7)

Run:  python src/train_model.py
Requires: data/processed/ files from feature_engineering.py (run that first)
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                              f1_score, precision_recall_curve, classification_report,
                              confusion_matrix, roc_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"


def load_data():
    X_train = pd.read_parquet(f"{PROCESSED_DIR}/X_train.parquet")
    X_val = pd.read_parquet(f"{PROCESSED_DIR}/X_val.parquet")
    y_train = pd.read_parquet(f"{PROCESSED_DIR}/y_train.parquet")["Default"]
    y_val = pd.read_parquet(f"{PROCESSED_DIR}/y_val.parquet")["Default"]
    return X_train, X_val, y_train, y_val


def evaluate(model, X_val, y_val, name):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "model": name,
        "roc_auc": round(roc_auc_score(y_val, proba), 4),
        "precision": round(precision_score(y_val, preds), 4),
        "recall": round(recall_score(y_val, preds), 4),
        "f1": round(f1_score(y_val, preds), 4),
    }
    print(f"\n--- {name} ---")
    print(metrics)
    print(classification_report(y_val, preds))
    return metrics


def find_business_threshold(model, X_val, y_val, target_recall=0.60):
    proba = model.predict_proba(X_val)[:, 1]
    prec, rec, thresh = precision_recall_curve(y_val, proba)
    valid = [(t, p, r) for p, r, t in zip(prec[:-1], rec[:-1], thresh) if r >= target_recall]
    if not valid:
        return 0.5
    best = max(valid, key=lambda x: x[1])
    return round(float(best[0]), 3)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    X_train, X_val, y_train, y_val = load_data()
    print("Train:", X_train.shape, "Val:", X_val.shape)

    results = []

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train, y_train)
    results.append(evaluate(logreg, X_val, y_val, "LogisticRegression"))

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate(rf, X_val, y_val, "RandomForest"))

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="auc",
        random_state=42, n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate(xgb, X_val, y_val, "XGBoost"))

    lgbm = LGBMClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=42
    )
    lgbm.fit(X_train, y_train)
    results.append(evaluate(lgbm, X_val, y_val, "LightGBM"))

    best_name = max(results, key=lambda r: r["roc_auc"])["model"]
    candidates = {"LogisticRegression": logreg, "RandomForest": rf,
                  "XGBoost": xgb, "LightGBM": lgbm}
    best_model = candidates[best_name]
    print(f"\n>>> Best model by ROC-AUC: {best_name}")

    if best_name == "XGBoost":
        param_dist = {
            "n_estimators": [300, 400, 600],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
        }
        search = RandomizedSearchCV(
            XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="auc",
                           random_state=42, n_jobs=-1),
            param_distributions=param_dist, n_iter=15, scoring="roc_auc",
            cv=3, random_state=42, n_jobs=-1, verbose=1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print("Best params:", search.best_params_)
        results.append(evaluate(best_model, X_val, y_val, "XGBoost_Tuned"))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\n5-fold CV ROC-AUC: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")

    threshold = find_business_threshold(best_model, X_val, y_val, target_recall=0.60)
    print(f"Chosen business threshold (>= this probability -> flag for manual review): {threshold}")

    joblib.dump(best_model, f"{MODELS_DIR}/final_model.pkl")
    feature_columns = X_train.columns.tolist()
    joblib.dump(feature_columns, f"{MODELS_DIR}/feature_columns.pkl")

    with open(f"{MODELS_DIR}/model_card.md", "w") as f:
        f.write(f"""# Model Card — Loan Default Risk

**Algorithm:** {type(best_model).__name__}
**Validation ROC-AUC:** {results[-1]['roc_auc']}
**Validation Precision / Recall / F1:** {results[-1]['precision']} / {results[-1]['recall']} / {results[-1]['f1']}
**5-fold CV ROC-AUC:** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
**Business decision threshold:** {threshold} (probability >= threshold -> flag for manual review)
**Class imbalance handling:** class_weight='balanced' / scale_pos_weight
**Known limitations:** trained on a single static snapshot dataset; no time-based
validation was possible (no application-date column); should be re-validated
periodically against real approval outcomes before any production use.
""")

    with open(f"{MODELS_DIR}/all_model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"{MODELS_DIR}/threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f)

    cm = confusion_matrix(y_val, (best_model.predict_proba(X_val)[:, 1] >= threshold).astype(int)).tolist()
    fpr, tpr, _ = roc_curve(y_val, best_model.predict_proba(X_val)[:, 1])
    step = max(1, len(fpr) // 200)
    roc_out = {
        "confusion_matrix": cm,
        "roc": {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()},
        "labels": ["No Default", "Default"]
    }
    with open(f"{MODELS_DIR}/roc_confusion.json", "w") as f:
        json.dump(roc_out, f)

    print(f"\nSaved: {MODELS_DIR}/final_model.pkl, model_card.md, feature_columns.pkl, "
          f"threshold.json, roc_confusion.json")


if __name__ == "__main__":
    main()