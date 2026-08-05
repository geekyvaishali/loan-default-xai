"""
Reusable explainability module — imported by app/app.py.
Wraps a SHAP TreeExplainer once (with a small sampled background set) so it's
NOT rebuilt on every Streamlit rerun — that's the perf trick from the roadmap.
"""

import shap
import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"


@st.cache_resource
def get_explainer():
    """Cached across reruns/sessions — built once per app deployment."""
    model = joblib.load(f"{MODELS_DIR}/final_model.pkl")
    model_type = type(model).__name__
    if model_type == "LogisticRegression":
        X_val = pd.read_parquet(f"{PROCESSED_DIR}/X_val.parquet").astype(float)
        background = shap.sample(X_val, 100, random_state=42)
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.TreeExplainer(model)
    return explainer, model


def explain_applicant(row_df: pd.DataFrame):
    """
    row_df: single-row dataframe, already encoded with the SAME columns
    the model was trained on (see src/feature_engineering.py + app/app.py
    alignment step).
    Returns: (probability_of_default, shap.Explanation object for waterfall plot)
    """
    explainer, model = get_explainer()
    proba = model.predict_proba(row_df)[0, 1]
    explanation = explainer(row_df)
    return proba, explanation[0]
