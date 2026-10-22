"""Model pipelines + probability calibration.

One sklearn ``Pipeline`` takes the *raw* applicant columns and returns
probabilities, so training, batch scoring, the API and the dashboard all run the
exact same transformations (no re-implementation in app.py).
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLS, NUMERIC_COLS
from .features import ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC, LoanFeatureEngineer

# Complexity order, simplest first. Used by the one-standard-error rule.
COMPLEXITY_ORDER = ["logistic_regression", "random_forest", "hist_gradient_boosting", "lightgbm", "xgboost"]


class SanitizeColumns(BaseEstimator, TransformerMixin):
    """Make column names safe for XGBoost/LightGBM (no [ ] < > quotes, spaces ...)."""

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = [re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_") for c in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out.columns = self.columns_
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)


def _preprocessor() -> ColumnTransformer:
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_COLS + ENGINEERED_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLS + ENGINEERED_CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return ct


def _estimators(seed: int, n_jobs: int) -> dict[str, Callable[[], object]]:
    """Factories for each candidate. Optional libraries are imported lazily."""

    def logreg():
        # No class_weight: re-weighting classes distorts the probabilities. Imbalance is
        # handled by threshold selection and evaluated with PR-AUC, not by skewing the fit.
        return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)

    def rf():
        return RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=50, max_features="sqrt",
            n_jobs=n_jobs, random_state=seed,
        )

    def hgb():
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=100,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=seed,
        )

    def xgb():
        from xgboost import XGBClassifier  # optional dependency
        return XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=5, reg_lambda=1.0, eval_metric="auc",
            n_jobs=n_jobs, random_state=seed, tree_method="hist",
        )

    def lgbm():
        from lightgbm import LGBMClassifier  # optional dependency
        return LGBMClassifier(
            n_estimators=400, max_depth=4, num_leaves=15, learning_rate=0.05,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.85, min_child_samples=50,
            reg_lambda=1.0, n_jobs=n_jobs, random_state=seed, verbose=-1,
        )

    return {
        "logistic_regression": logreg, "random_forest": rf, "hist_gradient_boosting": hgb,
        "xgboost": xgb, "lightgbm": lgbm,
    }


def available_models(requested: tuple[str, ...], seed: int = 42, n_jobs: int = 1) -> tuple[list[str], dict[str, str]]:
    """Return (usable model names, {skipped name: reason})."""
    factories = _estimators(seed, n_jobs)
    usable, skipped = [], {}
    for name in requested:
        if name not in factories:
            skipped[name] = "unknown model name"
            continue
        try:
            factories[name]()
            usable.append(name)
        except ImportError as exc:
            skipped[name] = f"not installed ({exc.name})"
    return usable, skipped


def build_pipeline(model_name: str, seed: int = 42, n_jobs: int = 1) -> Pipeline:
    factories = _estimators(seed, n_jobs)
    if model_name not in factories:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {sorted(factories)}")
    return Pipeline(
        [
            ("features", LoanFeatureEngineer()),
            ("prep", _preprocessor()),
            ("clean_names", SanitizeColumns()),
            ("model", factories[model_name]()),
        ]
    )


class ProbabilityCalibrator:
    """Maps raw model scores to calibrated probabilities (fit on a held-out split)."""

    def __init__(self, method: str = "isotonic"):
        if method not in {"isotonic", "sigmoid"}:
            raise ValueError("method must be 'isotonic' or 'sigmoid'")
        self.method = method

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        scores = np.asarray(scores, dtype=float)
        if self.method == "isotonic":
            self._m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, y)
        else:
            self._m = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(scores).reshape(-1, 1), y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.method == "isotonic":
            return self._m.predict(scores)
        return self._m.predict_proba(self._logit(scores).reshape(-1, 1))[:, 1]

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))


class CalibratedRiskModel:
    """Fitted pipeline + calibrator, exposing a sklearn-style ``predict_proba``."""

    def __init__(self, pipeline: Pipeline, calibrator: ProbabilityCalibrator, name: str):
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.name = name

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.calibrator.predict(self.predict_proba_raw(X))
        return np.column_stack([1 - p, p])
