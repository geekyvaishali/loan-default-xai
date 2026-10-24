"""Per-applicant attribution of the model score.

Two methods, chosen by model type:

* LOGISTIC REGRESSION: exact. Each transformed column contributes  coef * (x - training_mean)  to the
  log-odds, so  base + sum(contributions) == the model's own log-odds. (Identical to SHAP's
  LinearExplainer for independent features, with no extra dependency.)
* ANY OTHER MODEL: reference-value replacement. For each input, swap its value for a typical training
  value (median / mode) and measure how the score changes. Approximate and NOT additive.

Contributions describe how the model uses its inputs. They are association, not causation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import RAW_FEATURES

_BUCKET_TO_RAW = {"AgeBucket": "Age", "CreditScoreBucket": "CreditScore", "EmploymentLengthBucket": "MonthsEmployed"}
_PREFIXES = sorted(list(RAW_FEATURES) + list(_BUCKET_TO_RAW), key=len, reverse=True)

LABELS = {
    "Age": "Age", "Income": "Annual income", "LoanAmount": "Loan amount", "CreditScore": "Credit score",
    "MonthsEmployed": "Employment length", "NumCreditLines": "Open credit lines", "InterestRate": "Interest rate",
    "LoanTerm": "Loan tenure", "DTIRatio": "Debt-to-income ratio", "Education": "Education",
    "EmploymentType": "Employment type", "MaritalStatus": "Marital status", "HasMortgage": "Mortgage",
    "HasDependents": "Dependents", "LoanPurpose": "Loan purpose", "HasCoSigner": "Co-signer",
    "LoanToIncomeRatio": "Loan-to-income ratio", "InterestBurden": "Annual interest cost",
}


def feature_group(column: str) -> str:
    """Fold an encoded/bucket column back to the input the applicant actually supplied."""
    for name in _PREFIXES:
        if column == name or column.startswith(name + "_"):
            return _BUCKET_TO_RAW.get(name, name)
    return column


def _final_estimator(model):
    return model.pipeline[-1]


def linear_contributions(model, row: pd.DataFrame, transformed_means: dict) -> dict:
    """Exact log-odds contributions per input for a logistic-regression pipeline."""
    pipe = model.pipeline
    Z = pipe[:-1].transform(row)
    est = pipe[-1]
    means = np.array([transformed_means[c] for c in Z.columns], dtype=float)
    coef = est.coef_[0]
    per_column = (Z.to_numpy(dtype=float)[0] - means) * coef
    grouped: dict[str, float] = {}
    for col, val in zip(Z.columns, per_column):
        g = feature_group(col)
        grouped[g] = grouped.get(g, 0.0) + float(val)
    return {"base_logit": float(est.intercept_[0] + coef @ means), "contributions": grouped}


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def _replacement_contributions(model, row: pd.DataFrame, reference_values: dict) -> dict:
    base = _logit(float(model.predict_proba(row)[0, 1]))
    out = {}
    for col in RAW_FEATURES:
        if col not in reference_values:
            continue
        probe = row.copy()
        probe[col] = reference_values[col]
        out[col] = base - _logit(float(model.predict_proba(probe)[0, 1]))
    return out


def _display_value(row: pd.DataFrame, group: str):
    if group in row.columns:
        v = row[group].iloc[0]
        return v.item() if hasattr(v, "item") else v
    return None


def explain_row(model, row: pd.DataFrame, schema: dict, top_n: int = 8) -> dict:
    """Return {'method', 'additive', 'factors': [{label, direction, contribution, feature, value}]}."""
    info = (schema or {}).get("explain", {})
    if isinstance(_final_estimator(model), LogisticRegression) and info.get("transformed_means"):
        res = linear_contributions(model, row, info["transformed_means"])
        contributions, additive = res["contributions"], True
        method = ("Exact linear contributions to the model's log-odds, relative to the average training applicant "
                  "(equivalent to SHAP for a linear model)")
    else:
        contributions = _replacement_contributions(model, row, info.get("reference_values", {}))
        additive = False
        method = ("Reference-value replacement: how much the score changes if this input were typical instead of "
                  "the value given (approximate, not additive)")
    ordered = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:top_n]
    factors = [{"feature": g, "label": LABELS.get(g, g), "value": _display_value(row, g),
                "contribution": float(c), "direction": "raises" if c > 0 else "lowers"} for g, c in ordered]
    return {"method": method, "additive": additive, "factors": factors}
