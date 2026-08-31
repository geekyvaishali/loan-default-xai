"""
PHASE 5 — Streamlit App (Enhanced v3: batch CSV upload, model comparison tab,
prediction history, fixed income/loan sync bug, top-risk-factors summary)
Run:  streamlit run app/app.py   (run this command from the project ROOT folder)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import shap
import plotly.graph_objects as go
from datetime import datetime

from src.feature_engineering import engineer_features, CATEGORICAL_COLS
from src.explain import explain_applicant

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="🏦", layout="wide")

MODELS_DIR = "models"

# ---------------- Custom CSS ----------------
st.markdown("""
<style>
    .main { background: linear-gradient(180deg, #0E1117 0%, #131826 100%); }
    h1 { background: linear-gradient(90deg, #00C2A8, #6C63FF);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-weight: 800; }
    .stButton>button {
        background: linear-gradient(90deg, #00C2A8, #6C63FF);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 1.2rem; font-weight: 600; width: 100%;
    }
    .stButton>button:hover { opacity: 0.85; }
    div[data-testid="stMetric"] {
        background-color: #1A1F2B; border-radius: 12px; padding: 15px;
        border: 1px solid #2A2F3B;
    }
    .result-card {
        background-color: #1A1F2B; border-radius: 14px; padding: 20px;
        border-left: 5px solid #00C2A8; margin-top: 10px;
    }
    .factor-pill {
        display: inline-block; padding: 6px 14px; border-radius: 20px;
        margin: 4px 6px 4px 0; font-size: 0.85rem; font-weight: 600;
    }
    .factor-up { background-color: #3A1F26; color: #FF7A8A; }
    .factor-down { background-color: #1A3A32; color: #4DE8C4; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{MODELS_DIR}/final_model.pkl")
    feature_columns = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")
    with open(f"{MODELS_DIR}/threshold.json") as f:
        threshold = json.load(f)["threshold"]
    return model, feature_columns, threshold


def build_encoded_features(df_raw: pd.DataFrame, feature_columns):
    """Shared preprocessing path — used by both single-applicant and batch prediction,
    so the two can never silently drift apart."""
    engineered = engineer_features(df_raw)
    encoded = pd.get_dummies(
        engineered,
        columns=CATEGORICAL_COLS + ["EmploymentLengthBucket", "AgeBucket", "CreditScoreBucket"]
    )
    encoded.columns = (
        encoded.columns.str.replace(r"[\[\]<>]", "", regex=True).str.replace(" ", "_")
    )
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded.astype(float)


model, feature_columns, threshold = load_artifacts()

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("https://em-content.zobj.net/source/apple/391/bank_1f3e6.png", width=60)
    st.header("Model Info")
    st.markdown(f"""
    **Algorithm:** `{type(model).__name__}`
    **Decision threshold:** `{threshold}`
    """)
    st.divider()
    st.caption("⚠️ Portfolio / educational project — **not** financial advice, "
               "not a real lending decision tool.")
    st.divider()
    st.caption(f"Built by Vaishali Singh · {datetime.now().year}")

st.title("🏦 Loan Default Risk Predictor")
st.caption("Enter applicant details, get a risk score + a SHAP explanation of *why*.")

tab_predict, tab_batch, tab_models, tab_about = st.tabs(
    ["🔍 Predict", "📂 Batch Scoring", "📊 Model Comparison", "ℹ️ About"]
)

# ==================== TAB 1: SINGLE PREDICTION ====================
with tab_predict:
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 75, 35)
        income = st.slider("Annual Income ($)", 1000, 300000, 60000, step=1000)
        loan_amount = st.slider("Loan Amount ($)", 500, 300000, 20000, step=500)
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

    st.divider()
    predict_clicked = st.button("🔍 Predict Risk", type="primary")

    if predict_clicked:
        input_row = pd.DataFrame([{
            "Age": age, "Income": income, "LoanAmount": loan_amount,
            "CreditScore": credit_score, "MonthsEmployed": months_employed,
            "NumCreditLines": num_credit_lines, "InterestRate": interest_rate,
            "LoanTerm": loan_term, "DTIRatio": dti_ratio,
            "Education": education, "EmploymentType": employment_type,
            "MaritalStatus": marital_status, "HasMortgage": has_mortgage,
            "HasDependents": has_dependents, "LoanPurpose": loan_purpose,
            "HasCoSigner": has_cosigner,
        }])

        encoded = build_encoded_features(input_row, feature_columns)

        try:
            proba, shap_row = explain_applicant(encoded)
        except Exception as e:
            st.error(f"Prediction failed — check that models/ and data/processed/ exist. Error: {e}")
            st.stop()

        decision = "REJECT" if proba >= threshold + 0.2 else "MANUAL REVIEW" if proba >= threshold else "APPROVE"
        st.session_state.history.insert(0, {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Age": age, "Income": income, "LoanAmount": loan_amount,
            "Risk %": round(proba * 100, 1), "Decision": decision
        })
        st.session_state.history = st.session_state.history[:5]  # keep last 5

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)

            gauge_color = "#00C2A8" if proba < threshold else ("#FFB703" if proba < threshold + 0.2 else "#EF476F")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                number={'suffix': "%", 'font': {'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickvals': [0, 25, 50, 75, 100]},
                    'bar': {'color': gauge_color},
                    'bgcolor': "#1A1F2B",
                    'steps': [
                        {'range': [0, threshold * 100], 'color': "#2A2F3B"},
                        {'range': [threshold * 100, 100], 'color': "#3A2A2F"},
                    ],
                    'threshold': {'line': {'color': "white", 'width': 3},
                                   'thickness': 0.8, 'value': threshold * 100}
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=40, r=60, t=50, b=20),
                                     paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'size': 14})
            st.plotly_chart(fig_gauge, use_container_width=True)

            if decision == "REJECT":
                st.error("🚫 **REJECT** — high default risk")
            elif decision == "MANUAL REVIEW":
                st.warning("⚠️ **MANUAL REVIEW** — borderline risk")
            else:
                st.success("✅ **APPROVE** — low default risk")
            st.caption(f"Decision threshold: {threshold}")

            report = f"""Loan Default Risk Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Applicant Profile:
- Age: {age} | Income: ${income:,} | Loan Amount: ${loan_amount:,}
- Credit Score: {credit_score} | DTI Ratio: {dti_ratio} | Interest Rate: {interest_rate}%
- Employment: {employment_type} ({months_employed} months) | Education: {education}
- Marital Status: {marital_status} | Loan Purpose: {loan_purpose}

Result:
- Predicted Default Probability: {proba*100:.2f}%
- Decision Threshold: {threshold}
- Recommendation: {decision}
"""
            st.download_button("📄 Download Report", report, file_name="loan_risk_report.txt")
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown("**Why this score? (SHAP waterfall)**")
            fig = plt.figure(figsize=(8, 5))
            shap.plots.waterfall(shap_row, show=False, max_display=8)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            vals = shap_row.values
            names = shap_row.feature_names if hasattr(shap_row, "feature_names") else feature_columns
            pairs = sorted(zip(names, vals), key=lambda x: abs(x[1]), reverse=True)[:5]
            st.markdown("**Top factors for THIS applicant:**")
            pills_html = ""
            for name, val in pairs:
                cls = "factor-up" if val > 0 else "factor-down"
                arrow = "▲ increases risk" if val > 0 else "▼ decreases risk"
                pills_html += f'<span class="factor-pill {cls}">{name}: {arrow} ({val:+.2f})</span>'
            st.markdown(pills_html, unsafe_allow_html=True)

    # ---------------- Prediction history ----------------
    if st.session_state.history:
        st.divider()
        st.markdown("**🕒 Recent predictions (this session)**")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

# ==================== TAB 2: BATCH CSV SCORING ====================
with tab_batch:
    st.markdown("### Score many applicants at once")
    st.caption(
        "Upload a CSV with columns: Age, Income, LoanAmount, CreditScore, MonthsEmployed, "
        "NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, "
        "MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner"
    )

    uploaded = st.file_uploader("Upload applicants CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            required_cols = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
                              "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
                              "Education", "EmploymentType", "MaritalStatus", "HasMortgage",
                              "HasDependents", "LoanPurpose", "HasCoSigner"]
            missing = [c for c in required_cols if c not in batch_df.columns]
            if missing:
                st.error(f"CSV is missing required columns: {missing}")
            else:
                encoded_batch = build_encoded_features(batch_df[required_cols].copy(), feature_columns)
                probs = model.predict_proba(encoded_batch)[:, 1]

                results = batch_df.copy()
                results["Default_Probability_%"] = np.round(probs * 100, 2)
                results["Decision"] = np.select(
                    [probs >= threshold + 0.2, probs >= threshold],
                    ["REJECT", "MANUAL REVIEW"],
                    default="APPROVE"
                )

                st.success(f"Scored {len(results)} applicants.")
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Approved", int((results['Decision'] == 'APPROVE').sum()))
                mcol2.metric("Manual Review", int((results['Decision'] == 'MANUAL REVIEW').sum()))
                mcol3.metric("Rejected", int((results['Decision'] == 'REJECT').sum()))

                st.dataframe(results, use_container_width=True, hide_index=True)

                csv_out = results.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download scored results (CSV)", csv_out,
                                    file_name="scored_applicants.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not process file: {e}")
    else:
        st.info("No file uploaded yet. Try exporting a few rows from your dataset "
                "(without the 'Default' and 'LoanID' columns) to test this.")

# ==================== TAB 3: MODEL COMPARISON ====================
with tab_models:
    st.markdown("### How the 4 candidate models compared")
    comp_path = f"{MODELS_DIR}/all_model_comparison.json"
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comparison = json.load(f)
        comp_df = pd.DataFrame(comparison)

        metric = st.radio("Metric", ["roc_auc", "precision", "recall", "f1"], horizontal=True)
        fig_bar = go.Figure(go.Bar(
            x=comp_df["model"], y=comp_df[metric],
            marker_color="#00C2A8",
            text=comp_df[metric], textposition="outside"
        ))
        fig_bar.update_layout(
            height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}, yaxis_title=metric.replace("_", " ").upper()
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption(f"**Selected model:** {type(model).__name__} "
                   f"(chosen by highest validation ROC-AUC)")
    else:
        st.info("Run `python src/train_model.py` first — it writes "
                "`models/all_model_comparison.json` with all 4 models' metrics.")

# ==================== TAB 4: ABOUT ====================
with tab_about:
    st.markdown("""
    ### About
    This app predicts loan default probability and explains every prediction
    using **SHAP** (SHapley Additive exPlanations), so risk decisions aren't
    a black box.

    **Pipeline:** EDA → Feature Engineering → Model Comparison
    (Logistic Regression / Random Forest / XGBoost / LightGBM) →
    Explainability (SHAP + LIME) → this app.

    **Tech stack:** scikit-learn, XGBoost, LightGBM, SHAP, LIME, Streamlit, Plotly
    """)
    st.caption("Not financial advice. Educational portfolio project.")