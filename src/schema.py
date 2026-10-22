"""Input schema + data-quality validation.

Two tiers of numeric checks:
* HARD bounds  -> the value is invalid (negative income, credit score 9999 ...).
                  The row cannot be scored.
* TRAINING range -> the value is valid but outside what the model saw.
                  The row is scored with a warning (the model is extrapolating).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CATEGORICAL_COLS, NUMERIC_COLS, RAW_FEATURES, TARGET

HARD_BOUNDS: dict[str, tuple[float, float]] = {
    "Age": (18, 100), "Income": (1, 10_000_000), "LoanAmount": (1, 10_000_000),
    "CreditScore": (300, 850), "MonthsEmployed": (0, 720), "NumCreditLines": (0, 50),
    "InterestRate": (0, 100), "LoanTerm": (1, 480), "DTIRatio": (0, 10),
}
# Observed range of the Kaggle-style Loan_default.csv used for training.
TRAINING_RANGE: dict[str, tuple[float, float]] = {
    "Age": (18, 69), "Income": (15_000, 149_999), "LoanAmount": (5_000, 249_999),
    "CreditScore": (300, 849), "MonthsEmployed": (0, 119), "NumCreditLines": (1, 4),
    "InterestRate": (2.0, 25.0), "LoanTerm": (12, 60), "DTIRatio": (0.1, 0.9),
}
ALLOWED_CATEGORIES: dict[str, list[str]] = {
    "Education": ["Bachelor's", "High School", "Master's", "PhD"],
    "EmploymentType": ["Full-time", "Part-time", "Self-employed", "Unemployed"],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "HasMortgage": ["No", "Yes"],
    "HasDependents": ["No", "Yes"],
    "LoanPurpose": ["Auto", "Business", "Education", "Home", "Other"],
    "HasCoSigner": ["No", "Yes"],
}


class SchemaError(ValueError):
    """Raised when the input cannot be scored at all (e.g. missing columns)."""


@dataclass
class ValidationReport:
    n_rows: int
    missing_columns: list[str] = field(default_factory=list)
    row_errors: dict[int, list[str]] = field(default_factory=dict)      # positional idx -> messages
    row_warnings: dict[int, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.missing_columns and not self.row_errors

    @property
    def invalid_positions(self) -> list[int]:
        return sorted(self.row_errors)

    def summary(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "missing_columns": self.missing_columns,
            "n_invalid_rows": len(self.row_errors),
            "n_rows_with_warnings": len(self.row_warnings),
        }


def validate_frame(df: pd.DataFrame) -> ValidationReport:
    """Validate raw applicant rows. Never mutates ``df``."""
    report = ValidationReport(n_rows=len(df))
    report.missing_columns = [c for c in RAW_FEATURES if c not in df.columns]
    if report.missing_columns:
        return report

    errors: dict[int, list[str]] = {}
    warnings: dict[int, list[str]] = {}

    def add(store: dict[int, list[str]], mask: np.ndarray, msg: str) -> None:
        for pos in np.flatnonzero(mask):
            store.setdefault(int(pos), []).append(msg)

    for col in NUMERIC_COLS:
        numeric = pd.to_numeric(df[col], errors="coerce")
        add(errors, df[col].isna().to_numpy(), f"{col}: missing")
        add(errors, (numeric.isna() & df[col].notna()).to_numpy(), f"{col}: not numeric")
        lo, hi = HARD_BOUNDS[col]
        add(errors, ((numeric < lo) | (numeric > hi)).to_numpy(), f"{col}: outside valid range [{lo}, {hi}]")
        tlo, thi = TRAINING_RANGE[col]
        outside = ((numeric < tlo) | (numeric > thi)) & ~((numeric < lo) | (numeric > hi))
        add(warnings, outside.to_numpy(), f"{col}: outside training range [{tlo}, {thi}] (model extrapolates)")

    for col in CATEGORICAL_COLS:
        add(errors, df[col].isna().to_numpy(), f"{col}: missing")
        unknown = df[col].notna() & ~df[col].isin(ALLOWED_CATEGORIES[col])
        add(errors, unknown.to_numpy(), f"{col}: unknown category (allowed: {ALLOWED_CATEGORIES[col]})")

    report.row_errors = errors
    report.row_warnings = warnings
    return report


def data_quality_report(df: pd.DataFrame) -> dict:
    """Dataset-level checks for training data (duplicates, nulls, target, ranges)."""
    feats = [c for c in df.columns if c in RAW_FEATURES]
    report = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_null_cells": int(df.isna().sum().sum()),
        "n_duplicate_rows_excluding_id": int(df[feats + ([TARGET] if TARGET in df else [])].duplicated().sum()),
        "n_duplicate_ids": int(df["LoanID"].duplicated().sum()) if "LoanID" in df else None,
        "default_rate": float(df[TARGET].mean()) if TARGET in df else None,
        "numeric_summary": {
            c: {"min": float(df[c].min()), "max": float(df[c].max()), "mean": float(df[c].mean())}
            for c in NUMERIC_COLS if c in df
        },
        "category_levels": {c: sorted(map(str, df[c].unique())) for c in CATEGORICAL_COLS if c in df},
    }
    # How "flat" is each numeric column? 0 = perfectly uniform histogram. Real-world
    # financial data is almost never this flat, which suggests synthetic data.
    flat = {}
    for c in NUMERIC_COLS:
        if c in df:
            counts, _ = np.histogram(df[c], bins=10)
            flat[c] = round(float(counts.std() / counts.mean()), 4)
    report["histogram_flatness_cv"] = flat
    report["looks_synthetic"] = bool(sum(v < 0.02 for v in flat.values()) >= 4)
    return report
