"""Row-wise feature engineering as a scikit-learn transformer.

The old ``engineer_features`` computed a 99th-percentile cap and medians from
*whatever batch it was handed*. That leaks information at training time and
makes batch scoring depend on batch composition (a single row was never capped).

Here the only learned statistic (the winsorisation caps) is fitted on the
training split inside ``fit`` and applied unchanged afterwards. Everything else
is a pure row-wise function with fixed bin edges.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import NUMERIC_COLS, RAW_FEATURES

CAP_COLUMNS = ["Income", "LoanAmount"]

EMPLOYMENT_BINS = ([-np.inf, 12, 36, 84, np.inf], ["under_1yr", "1_3yr", "3_7yr", "7yr_plus"])
AGE_BINS = ([-np.inf, 25, 35, 45, 55, np.inf], ["18_25", "26_35", "36_45", "46_55", "56_plus"])
CREDIT_BINS = ([-np.inf, 579, 669, 739, 799, np.inf], ["Poor", "Fair", "Good", "VeryGood", "Excellent"])

ENGINEERED_NUMERIC = ["LoanToIncomeRatio", "InterestBurden"]
ENGINEERED_CATEGORICAL = ["EmploymentLengthBucket", "AgeBucket", "CreditScoreBucket"]


def _bucket(series: pd.Series, spec: tuple[list[float], list[str]]) -> pd.Series:
    bins, labels = spec
    return pd.cut(series, bins=bins, labels=labels, right=True).astype(str)


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds ratio and bucket features. Learns only the winsorisation caps."""

    def __init__(self, cap_quantile: float = 0.99):
        self.cap_quantile = cap_quantile

    def fit(self, X: pd.DataFrame, y=None):
        self._check_columns(X)
        self.caps_ = {c: float(pd.to_numeric(X[c]).quantile(self.cap_quantile)) for c in CAP_COLUMNS}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(X)
        out = X[RAW_FEATURES].copy()
        out[NUMERIC_COLS] = out[NUMERIC_COLS].apply(pd.to_numeric)
        for c, cap in self.caps_.items():
            out[c] = out[c].clip(upper=cap)

        income = out["Income"].to_numpy(dtype=float)
        out["LoanToIncomeRatio"] = np.where(income > 0, out["LoanAmount"] / np.where(income > 0, income, 1.0), 0.0)
        out["InterestBurden"] = out["InterestRate"] * out["LoanAmount"] / 100.0  # rough annual interest amount
        out["EmploymentLengthBucket"] = _bucket(out["MonthsEmployed"], EMPLOYMENT_BINS)
        out["AgeBucket"] = _bucket(out["Age"], AGE_BINS)
        out["CreditScoreBucket"] = _bucket(out["CreditScore"], CREDIT_BINS)
        return out

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        missing = [c for c in RAW_FEATURES if c not in X.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
