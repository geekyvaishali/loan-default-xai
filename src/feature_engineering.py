"""
PHASE 2 — Data Cleaning + Feature Engineering (Days 3-4)

Run standalone:  python src/feature_engineering.py
This reads data/raw/Loan_default.csv, engineers features, splits train/val,
and saves everything to data/processed/ so Phase 3 can just load it.
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

RAW_PATH = "data/raw/Loan_default.csv"
PROCESSED_DIR = "data/processed"

NUMERIC_COLS = ["Age", "Income", "LoanAmount", "CreditScore",
                 "MonthsEmployed", "NumCreditLines", "InterestRate",
                 "LoanTerm", "DTIRatio"]
CATEGORICAL_COLS = ["Education", "EmploymentType", "MaritalStatus",
                     "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"]
TARGET = "Default"
ID_COL = "LoanID"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure function: takes a raw dataframe (same columns as Loan_default.csv,
    WITHOUT the target if scoring a new applicant) and returns it with
    engineered features added. Used both at training time and inside the
    Streamlit app at inference time — keep this the single source of truth
    so train/serve preprocessing never drifts apart.
    """
    df = df.copy()

    # --- This dataset has no missing values, but we guard anyway in case
    # a new applicant's row (from the Streamlit form) has a blank field ---
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # --- Outlier capping (99th percentile) on skewed money columns ---
    for col in ["Income", "LoanAmount"]:
        cap = df[col].quantile(0.99)
        df[col] = np.where(df[col] > cap, cap, df[col])

    # --- Engineered ratio features ---
    df["LoanToIncomeRatio"] = df["LoanAmount"] / df["Income"].replace(0, np.nan)
    df["LoanToIncomeRatio"] = df["LoanToIncomeRatio"].fillna(df["LoanToIncomeRatio"].median())

    df["InterestBurden"] = df["InterestRate"] * df["LoanAmount"] / 100  # rough annual interest amount

    # DTIRatio already exists in the raw data (Debt-to-Income) — keep as is.

    # --- Employment length buckets ---
    df["EmploymentLengthBucket"] = pd.cut(
        df["MonthsEmployed"],
        bins=[-1, 12, 36, 84, 1000],
        labels=["<1yr", "1-3yr", "3-7yr", "7yr+"]
    ).astype(str)

    # --- Age buckets ---
    df["AgeBucket"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"]
    ).astype(str)

    # --- Credit score buckets (domain-standard-ish bands) ---
    df["CreditScoreBucket"] = pd.cut(
        df["CreditScore"],
        bins=[0, 579, 669, 739, 799, 900],
        labels=["Poor", "Fair", "Good", "VeryGood", "Excellent"]
    ).astype(str)

    return df


def build_preprocessing(df: pd.DataFrame):
    """
    One-hot encodes all categorical + engineered-bucket columns.
    Returns the encoded dataframe and the final list of feature columns
    (needed later so the Streamlit app can align single-row inputs to the
    same columns the model was trained on).
    """
    cat_cols_final = CATEGORICAL_COLS + ["EmploymentLengthBucket", "AgeBucket", "CreditScoreBucket"]
    df_encoded = pd.get_dummies(df, columns=cat_cols_final, drop_first=False)

    # XGBoost/LightGBM reject column names containing [, ], <, etc.
    # (bucket labels like "<1yr" and "18-25" trip this up) — sanitize them.
    df_encoded.columns = (
        df_encoded.columns
        .str.replace(r"[\[\]<>]", "", regex=True)
        .str.replace(" ", "_")
    )
    return df_encoded


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    print("Raw shape:", df.shape)

    y = df[TARGET]
    X_raw = df.drop(columns=[TARGET, ID_COL])

    X_eng = engineer_features(X_raw)
    X_encoded = build_preprocessing(X_eng)

    feature_columns = X_encoded.columns.tolist()
    print("Final feature count:", len(feature_columns))

    # Stratified split so the ~11.6% default rate is preserved in both sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_parquet(f"{PROCESSED_DIR}/X_train.parquet")
    X_val.to_parquet(f"{PROCESSED_DIR}/X_val.parquet")
    y_train.to_frame().to_parquet(f"{PROCESSED_DIR}/y_train.parquet")
    y_val.to_frame().to_parquet(f"{PROCESSED_DIR}/y_val.parquet")
    joblib.dump(feature_columns, f"{PROCESSED_DIR}/feature_columns.pkl")

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
        # Dataset snapshot stats (used by the app's top banner)
    stats = {
        "total_applicants": int(len(df)),
        "default_rate_pct": round(float(y.mean() * 100), 1),
        "num_features": len(feature_columns)
    }
    with open(f"{PROCESSED_DIR}/dataset_stats.json", "w") as f:
        json.dump(stats, f)
    print(f"Saved processed data + feature_columns.pkl to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
