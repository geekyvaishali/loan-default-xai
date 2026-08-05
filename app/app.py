"""
PHASE 5 — Streamlit App (Days 10-12)

Run:  streamlit run app/app.py   (run this command from the project ROOT folder)
Requires: models/final_model.pkl, models/feature_columns.pkl, models/threshold.json
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # so we can import src/

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import shap

from src.feature_engineering import engineer_features, CATEGORICAL_COLS
from src.explain import explain_applicant

st.set_page_config(page_title="Loan Default Risk Predictor", layout="wide")

MODELS_DIR = "models"


@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{MODELS_DIR}/final_model.pkl")
    feature_columns = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")
    with open(f"{MODELS_DIR}/threshold.json") as f:
        threshold = json.load(f)["threshold"]
    return model, feature_columns, threshold


model, feature_columns, threshold = load_artifacts()

# ---------------- Sidebar: model info ----------------
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown(f"""
    **Algorithm:** {type(model).__name__}
    **Decision threshold:** {threshold}
    **Note:** This is a portfolio / educational project,
    **not** financial advice and not a real lending decision tool.
    """)

st.title("🏦 Loan Default Risk Predictor")
st.caption("Enter applicant details, get a risk score + a SHAP explanation of *why*.")

# ---------------- Input form ----------------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 75, 35)
    income = st.number_input("Annual Income ($)", min_value=1000, value=60000, step=1000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=500, value=20000, step=500)
    credit_score = st.slider("Credit Score", 300, 850, 650)

with col2:
    months_employed = st.slider("Months Employed", 0, 480, 36)
    num_credit_lines = st.slider("Number of Credit Lines", 0, 20, 3)
    interest_rate = st.slider("Interest Rate (%)", 1.0, 30.0, 10.0, step=0.1)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])

with col3:
    dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, step=0.01)
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

col4, col5, col6 = st.columns(3)
with col4:
    has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
with col5:
    has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])
with col6:
    has_cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])

# ---------------- What-if panel ----------------
st.divider()
st.subheader("🎚️ What-If: adjust income / loan amount live")
whatif_income = st.slider("What-if Income", 1000, 300000, income, step=1000)
whatif_loan = st.slider("What-if Loan Amount", 500, 300000, loan_amount, step=500)

if st.button("🔍 Predict Risk", type="primary"):
    input_row = pd.DataFrame([{
        "Age": age, "Income": whatif_income, "LoanAmount": whatif_loan,
        "CreditScore": credit_score, "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines, "InterestRate": interest_rate,
        "LoanTerm": loan_term, "DTIRatio": dti_ratio,
        "Education": education, "EmploymentType": employment_type,
        "MaritalStatus": marital_status, "HasMortgage": has_mortgage,
        "HasDependents": has_dependents, "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner,
    }])

    # Mirror EXACT training-time preprocessing
    engineered = engineer_features(input_row)
    encoded = pd.get_dummies(
        engineered,
        columns=CATEGORICAL_COLS + ["EmploymentLengthBucket", "AgeBucket", "CreditScoreBucket"]
    )
    encoded.columns = (
        encoded.columns.str.replace(r"[\[\]<>]", "", regex=True).str.replace(" ", "_")
    )
    encoded = encoded.astype(float)
    # Align columns to what the model was trained on (missing dummy cols -> 0)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    try:
        proba, shap_row = explain_applicant(encoded)
    except Exception as e:
        st.error(f"Prediction failed — check that models/ and data/processed/ exist. Error: {e}")
        st.stop()

    # ---------------- Output ----------------
    st.divider()
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric("Risk Probability", f"{proba*100:.1f}%")
        if proba >= threshold:
            if proba >= threshold + 0.2:
                st.error("🚫 REJECT — high default risk")
            else:
                st.warning("⚠️ MANUAL REVIEW — borderline risk")
        else:
            st.success("✅ APPROVE — low default risk")
        st.caption(f"Decision threshold: {threshold}")

    with res_col2:
        st.markdown("**Why this score? (SHAP waterfall)**")
        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(shap_row, show=False, max_display=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

st.divider()
st.caption("Built with scikit-learn / XGBoost / SHAP · Portfolio project by Vaishali Singh")
