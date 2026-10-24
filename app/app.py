"""
PHASE 5 — Streamlit App (Enhanced v7: light/white professional theme,
corporate navy+teal palette, Material line-icons, applicant presets)
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

st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon=":material/account_balance:",
    layout="wide"
)

MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"

NAVY = "#1E3A5F"
NAVY_LIGHT = "#2C5282"
TEAL = "#0E7C7B"
TEAL_LIGHT = "#149D9B"
DANGER = "#C0392B"
WARN = "#B7791F"
SUCCESS = "#0E7C7B"
BG_PAGE = "#F7F9FB"
BG_CARD = "#FFFFFF"
BORDER = "#E1E6EC"
TEXT = "#1A2333"
MUTED = "#5B6472"

st.markdown(f"""
<style>
    .main {{ background-color: {BG_PAGE}; }}
    [data-testid="stAppViewContainer"] {{ background-color: {BG_PAGE}; }}
    [data-testid="stHeader"] {{ background-color: {BG_PAGE}; }}
    [data-testid="stSidebar"] {{ background-color: {BG_CARD}; border-right: 1px solid {BORDER}; }}
    h1, h2, h3, h4 {{ color: {TEXT} !important; }}
    p, span, label, div {{ color: {TEXT}; }}
    h1 {{ font-weight: 700; letter-spacing: -0.5px; }}
    h1 span.accent {{ color: {TEAL}; }}
    .stButton>button {{
        background-color: {NAVY}; color: #FFFFFF; border: none; border-radius: 6px;
        padding: 0.55rem 1.1rem; font-weight: 500; width: 100%;
    }}
    .stButton>button:hover {{ background-color: {NAVY_LIGHT}; color: #FFFFFF; }}
    div[data-testid="stMetric"] {{
        background-color: {BG_CARD}; border-radius: 8px; padding: 14px;
        border: 1px solid {BORDER}; border-top: 3px solid {TEAL};
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    div[data-testid="stMetricValue"] {{ color: {TEXT}; }}
    .result-card {{
        background-color: {BG_CARD}; border-radius: 10px; padding: 20px;
        border: 1px solid {BORDER}; border-left: 4px solid {NAVY}; margin-top: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    .factor-pill {{
        display: inline-block; padding: 5px 12px; border-radius: 4px;
        margin: 4px 6px 4px 0; font-size: 0.82rem; font-weight: 500;
        border: 1px solid {BORDER};
    }}
    .factor-up {{ background-color: #FCEEEC; color: #A6392A; }}
    .factor-down {{ background-color: #E9F6F5; color: #0B6564; }}
    .stats-banner {{
        background-color: {BG_CARD}; border-radius: 8px; padding: 14px 22px;
        margin-bottom: 16px; border: 1px solid {BORDER}; border-left: 4px solid {TEAL};
        display: flex; gap: 36px; flex-wrap: wrap;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    .stats-item {{ color: {NAVY}; font-weight: 700; font-size: 1.05rem; }}
    .stats-label {{ color: {MUTED}; font-size: 0.78rem; font-weight: 400; }}
    .step-card {{
        background-color: {BG_CARD}; border-radius: 8px; padding: 16px;
        text-align: center; border: 1px solid {BORDER}; height: 140px;
        border-top: 3px solid {NAVY};
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    .step-num {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 28px; height: 28px; border-radius: 50%;
        background-color: {NAVY}; color: white; font-weight: 700;
        font-size: 0.85rem; margin-bottom: 8px;
    }}
    .badge {{
        display: inline-block; padding: 5px 12px; border-radius: 4px;
        margin: 3px 5px 3px 0; font-size: 0.8rem; border: 1px solid {BORDER};
        background-color: #F0F5F4; color: {TEAL};
    }}
    [data-testid="stMarkdownContainer"] p {{ color: {TEXT}; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{MODELS_DIR}/final_model.pkl")
    feature_columns = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")
    with open(f"{MODELS_DIR}/threshold.json") as f:
        threshold = json.load(f)["threshold"]
    return model, feature_columns, threshold


@st.cache_data
def load_dataset_stats():
    path = f"{PROCESSED_DIR}/dataset_stats.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_roc_confusion():
    path = f"{MODELS_DIR}/roc_confusion.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def build_encoded_features(df_raw: pd.DataFrame, feature_columns):
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
dataset_stats = load_dataset_stats()

if "history" not in st.session_state:
    st.session_state.history = []

PRESETS = {
    "high_risk": {
        "age": 23, "income": 18000, "loan_amount": 45000, "credit_score": 420,
        "months_employed": 3, "num_credit_lines": 1, "interest_rate": 27.5,
        "loan_term": 60, "dti_ratio": 0.85, "education": "High School",
        "employment_type": "Unemployed", "marital_status": "Single",
        "has_mortgage": "No", "has_dependents": "Yes", "has_cosigner": "No",
        "loan_purpose": "Business",
    },
    "low_risk": {
        "age": 42, "income": 145000, "loan_amount": 12000, "credit_score": 790,
        "months_employed": 180, "num_credit_lines": 6, "interest_rate": 4.5,
        "loan_term": 12, "dti_ratio": 0.12, "education": "Master's",
        "employment_type": "Full-time", "marital_status": "Married",
        "has_mortgage": "Yes", "has_dependents": "No", "has_cosigner": "Yes",
        "loan_purpose": "Home",
    },
}


def apply_preset(name):
    for field, value in PRESETS[name].items():
        st.session_state[field] = value


with st.sidebar:
    st.markdown(f":material/account_balance: **Loan Default Risk**")
    st.caption("Predictor & Explainability Tool")
    st.divider()
    st.markdown(":material/model_training: **Model**")
    st.markdown(f"`{type(model).__name__}`")
    st.markdown(":material/tune: **Decision threshold**")
    st.markdown(f"`{threshold}`")
    st.divider()
    st.caption(":material/warning: Portfolio / educational project — not financial "
               "advice, not a real lending decision tool.")
    st.divider()
    st.caption(f"Vaishali Singh · {datetime.now().year}")

st.markdown(f'<h1>Loan Default Risk <span class="accent">Predictor</span></h1>', unsafe_allow_html=True)
st.caption("Enter applicant details, get a risk score and a SHAP explanation of why.")
st.warning(
    "**Experimental model on a dataset that appears to be synthetic.** The data appears to be synthetic, and this model was trained with "
    "class weights, so its score overstates the real default rate (average score about 42% against 11.6% actual). "
    "Read it as a relative ranking, not a probability. It is not a lending decision."
)

if dataset_stats:
    st.markdown(f"""
    <div class="stats-banner">
        <div><span class="stats-item">{dataset_stats.get('total_applicants', 'N/A'):,}</span>
             <span class="stats-label"> applicants analyzed</span></div>
        <div><span class="stats-item">{dataset_stats.get('default_rate_pct', 'N/A')}%</span>
             <span class="stats-label"> default rate</span></div>
        <div><span class="stats-item">{dataset_stats.get('num_features', 'N/A')}</span>
             <span class="stats-label"> engineered features</span></div>
        <div><span class="stats-item">{type(model).__name__}</span>
             <span class="stats-label"> best model</span></div>
    </div>
    """, unsafe_allow_html=True)

tab_predict, tab_batch, tab_models, tab_about = st.tabs([
    ":material/search: Predict",
    ":material/folder_open: Batch Scoring",
    ":material/bar_chart: Model Comparison",
    ":material/info: About"
])

with tab_predict:
    st.markdown("**Quick demo**")
    pcol1, pcol2, pcol3 = st.columns([1, 1, 3])
    with pcol1:
        st.button(":material/priority_high: Load High-Risk Example",
                   on_click=apply_preset, args=("high_risk",))
    with pcol2:
        st.button(":material/check_circle: Load Low-Risk Example",
                   on_click=apply_preset, args=("low_risk",))

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 18, 75, 35, key="age")
        income = st.slider("Annual Income ($)", 1000, 300000, 60000, step=1000, key="income")
        loan_amount = st.slider("Loan Amount ($)", 500, 300000, 20000, step=500, key="loan_amount")
        credit_score = st.slider("Credit Score", 300, 850, 650, key="credit_score")

    with col2:
        months_employed = st.slider("Months Employed", 0, 480, 36, key="months_employed")
        num_credit_lines = st.slider("Number of Credit Lines", 0, 20, 3, key="num_credit_lines")
        interest_rate = st.slider("Interest Rate (%)", 1.0, 30.0, 10.0, step=0.1, key="interest_rate")
        loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60], key="loan_term")

    with col3:
        dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, step=0.01, key="dti_ratio")
        education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"], key="education")
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"], key="employment_type")
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], key="marital_status")

    col4, col5, col6 = st.columns(3)
    with col4:
        has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"], key="has_mortgage")
    with col5:
        has_dependents = st.selectbox("Has Dependents", ["Yes", "No"], key="has_dependents")
    with col6:
        has_cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"], key="has_cosigner")

    loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"], key="loan_purpose")

    st.divider()
    predict_clicked = st.button(":material/search: Predict Risk", type="primary")

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

        decision = "HIGH RISK" if proba >= threshold + 0.2 else "ELEVATED RISK" if proba >= threshold else "LOWER RISK"
        st.session_state.history.insert(0, {
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Age": age, "Income": income, "LoanAmount": loan_amount,
            "Score %": round(proba * 100, 1), "Risk band": decision
        })
        st.session_state.history = st.session_state.history[:5]

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)

            gauge_color = TEAL if proba < threshold else (WARN if proba < threshold + 0.2 else DANGER)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba * 100, 1),
                number={'suffix': "%", 'font': {'size': 34, 'color': TEXT}},
                gauge={
                    'axis': {'range': [0, 100], 'tickvals': [0, 25, 50, 75, 100],
                             'tickfont': {'color': MUTED}},
                    'bar': {'color': gauge_color},
                    'bgcolor': "#F0F2F5",
                    'steps': [
                        {'range': [0, threshold * 100], 'color': "#E9F6F5"},
                        {'range': [threshold * 100, 100], 'color': "#FBEFE6"},
                    ],
                    'threshold': {'line': {'color': TEXT, 'width': 2},
                                   'thickness': 0.8, 'value': threshold * 100}
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=40, r=60, t=50, b=20),
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                     font={'color': TEXT, 'size': 14})
            st.plotly_chart(fig_gauge, use_container_width=True)

            if decision == "HIGH RISK":
                st.error(":material/error: HIGH RISK band: a flag for human review, not a decision")
            elif decision == "ELEVATED RISK":
                st.warning(":material/info: ELEVATED RISK band: above the review threshold")
            else:
                st.success(":material/check_circle: LOWER RISK band: below the review threshold")
            st.caption(f"Review threshold on the model score: {threshold}. Bands are display categories, not lending decisions.")

            report = f"""Loan Default Risk Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Applicant Profile:
- Age: {age} | Income: ${income:,} | Loan Amount: ${loan_amount:,}
- Credit Score: {credit_score} | DTI Ratio: {dti_ratio} | Interest Rate: {interest_rate}%
- Employment: {employment_type} ({months_employed} months) | Education: {education}
- Marital Status: {marital_status} | Loan Purpose: {loan_purpose}

Result:
- Model risk score (NOT a calibrated probability): {proba*100:.2f}%
- Review threshold on the score: {threshold}
- Risk band: {decision} (not a lending decision)
"""
            st.download_button(":material/download: Download Report", report, file_name="loan_risk_report.txt")
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
            st.markdown("**Top factors for this applicant**")
            pills_html = ""
            for name, val in pairs:
                cls = "factor-up" if val > 0 else "factor-down"
                arrow = "▲ increases risk" if val > 0 else "▼ decreases risk"
                pills_html += f'<span class="factor-pill {cls}">{name}: {arrow} ({val:+.2f})</span>'
            st.markdown(pills_html, unsafe_allow_html=True)

    if st.session_state.history:
        st.divider()
        st.markdown(":material/history: **Recent predictions (this session)**")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

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
                results["Risk_Score_%"] = np.round(probs * 100, 2)
                results["Risk band"] = np.select(
                    [probs >= threshold + 0.2, probs >= threshold],
                    ["HIGH RISK", "ELEVATED RISK"],
                    default="LOWER RISK"
                )

                st.success(f":material/task_alt: Scored {len(results)} applicants.")
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Lower risk", int((results['Risk band'] == 'LOWER RISK').sum()))
                mcol2.metric("Elevated risk", int((results['Risk band'] == 'ELEVATED RISK').sum()))
                mcol3.metric("High risk", int((results['Risk band'] == 'HIGH RISK').sum()))

                st.dataframe(results, use_container_width=True, hide_index=True)

                fig_hist = go.Figure(go.Histogram(
                    x=results["Risk_Score_%"], nbinsx=30,
                    marker_color=TEAL
                ))
                fig_hist.add_vline(x=threshold * 100, line_dash="dash", line_color=WARN,
                                    annotation_text="Review threshold", annotation_font_color=TEXT)
                fig_hist.update_layout(
                    title="Distribution of model risk scores", height=320,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': TEXT}, xaxis_title="Model risk score (%)", yaxis_title="Applicants"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                csv_out = results.to_csv(index=False).encode("utf-8")
                st.download_button(":material/download: Download scored results (CSV)", csv_out,
                                    file_name="scored_applicants.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not process file: {e}")
    else:
        st.info(":material/upload_file: No file uploaded yet. Try exporting a few rows "
                "from your dataset (without the 'Default' and 'LoanID' columns) to test this.")

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
            marker_color=[NAVY, TEAL, NAVY, TEAL],
            text=comp_df[metric], textposition="outside"
        ))
        fig_bar.update_layout(
            height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={'color': TEXT}, yaxis_title=metric.replace("_", " ").upper()
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption(f":material/emoji_events: Selected model: {type(model).__name__} "
                   f"(chosen by highest validation ROC-AUC)")
    else:
        st.info("Run `python src/train_model.py` first — it writes "
                "`models/all_model_comparison.json` with all 4 models' metrics.")

    st.divider()
    st.markdown("### Best model: ROC curve and confusion matrix")
    roc_conf = load_roc_confusion()
    if roc_conf:
        rc_col1, rc_col2 = st.columns(2)

        with rc_col1:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=roc_conf["roc"]["fpr"], y=roc_conf["roc"]["tpr"],
                mode="lines", line=dict(color=NAVY, width=3), name="ROC Curve"
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="#B0B7C0", dash="dash"), name="Random baseline"
            ))
            fig_roc.update_layout(
                title="ROC Curve", height=380,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={'color': TEXT}, xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate", showlegend=True
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with rc_col2:
            cm = roc_conf["confusion_matrix"]
            labels = roc_conf["labels"]
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=labels, y=labels,
                colorscale=[[0, "#F0F2F5"], [1, TEAL]],
                text=cm, texttemplate="%{text}", showscale=False
            ))
            fig_cm.update_layout(
                title="Confusion Matrix (at decision threshold)", height=380,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={'color': TEXT}, xaxis_title="Predicted", yaxis_title="Actual"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Run the ROC/confusion-matrix export step in `src/train_model.py` "
                "— it writes `models/roc_confusion.json`.")

with tab_about:
    st.markdown("### Loan Default Risk Predictor with Explainable AI")
    st.markdown(
        "This app estimates a loan default risk score and explains every "
        "prediction using SHAP. It is an educational demo, not a lending decision tool."
    )

    st.markdown(":material/route: **Pipeline**")
    pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
    steps = [
        ("1", "EDA", "Explore class imbalance and patterns"),
        ("2", "Feature Eng.", "Ratios, buckets, encoding"),
        ("3", "Modeling", "4 models compared, best selected"),
        ("4", "SHAP + LIME", "Global and local explanations"),
        ("5", "This App", "Live scoring and reports"),
    ]
    for col, (num, title, desc) in zip([pcol1, pcol2, pcol3, pcol4, pcol5], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div style="font-weight:600; color:{TEXT}; margin:4px 0;">{title}</div>
                <div style="font-size:0.75rem; color:{MUTED};">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(":material/construction: **Tech Stack**")
    tech = ["scikit-learn", "XGBoost", "LightGBM", "SHAP", "LIME", "Streamlit", "Plotly", "Pandas"]
    badges_html = "".join(f'<span class="badge">{t}</span>' for t in tech)
    st.markdown(badges_html, unsafe_allow_html=True)

    st.divider()
    st.markdown(":material/person: **Author**")
    st.markdown("**Vaishali Singh** — M.Sc. Computational Science, BHU Varanasi")
    st.markdown(":material/link: [GitHub](https://github.com/geekyvaishali)")

    st.divider()
    st.caption(":material/gpp_maybe: Not financial advice. Educational portfolio project only.")
