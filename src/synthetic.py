"""Synthetic applicants for tests and the demo CSV. NOT real data and NOT a benchmark."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import ALLOWED_CATEGORIES, TRAINING_RANGE


def make_synthetic_applicants(n: int = 500, seed: int = 0, with_target: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Age": rng.integers(18, 70, n),
        "Income": rng.integers(15_000, 150_000, n),
        "LoanAmount": rng.integers(5_000, 250_000, n),
        "CreditScore": rng.integers(300, 850, n),
        "MonthsEmployed": rng.integers(0, 120, n),
        "NumCreditLines": rng.integers(1, 5, n),
        "InterestRate": np.round(rng.uniform(2.0, 25.0, n), 2),
        "LoanTerm": rng.choice([12, 24, 36, 48, 60], n),
        "DTIRatio": np.round(rng.uniform(0.1, 0.9, n), 2),
    })
    for col, levels in ALLOWED_CATEGORIES.items():
        df[col] = rng.choice(levels, n)
    if with_target:
        z = (-2.6 + 0.09 * (df.InterestRate - 13) + 0.9 * (df.LoanAmount / df.Income - 1.2)
             - 0.012 * (df.Age - 44) - 0.004 * (df.CreditScore - 575) / 10)
        df["Default"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


assert set(TRAINING_RANGE) <= set(make_synthetic_applicants(5).columns)
