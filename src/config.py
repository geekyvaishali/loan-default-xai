"""Central configuration: paths, column definitions and training settings.

Everything that was previously hard-coded across train_model.py, app.py and
feature_engineering.py lives here, so there is one place to change it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "Loan_default.csv"
SAMPLE_CSV_PATH = ROOT / "data" / "sample_applicants.csv"
ARTIFACT_DIR = ROOT / "artifacts"

TARGET = "Default"
ID_COL = "LoanID"

NUMERIC_COLS = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
]
CATEGORICAL_COLS = [
    "Education", "EmploymentType", "MaritalStatus",
    "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner",
]
RAW_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS


@dataclass(frozen=True)
class TrainConfig:
    """Training settings. Every value is recorded in the saved metadata."""

    random_state: int = 42
    test_size: float = 0.20          # held out, touched once at the very end
    val_size: float = 0.20           # of the full data; used for calibration + threshold
    cv_folds: int = 3                # CV on the training split only (model comparison)
    selection_metric: str = "roc_auc"
    one_se_rule: bool = True         # prefer the simplest model within 1 std of the best
    calibration_method: str = "sigmoid"   # "sigmoid" keeps ranking; "isotonic" can create score ties
    threshold_policy: str = "recall_target"  # "recall_target" or "cost"
    target_recall: float = 0.60      # used by "recall_target"
    cost_fn: float = 5.0             # relative cost of a missed default (used by "cost")
    cost_fp: float = 1.0             # relative cost of a false alarm (used by "cost")
    models: tuple[str, ...] = (
        "logistic_regression", "random_forest", "hist_gradient_boosting",
        "xgboost", "lightgbm",
    )
    n_jobs: int = 1
