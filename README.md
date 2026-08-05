# Loan Default Risk Predictor with Explainable AI

Predicts an applicant's probability of loan default and explains *why* using
SHAP and LIME, wrapped in an interactive Streamlit app.

**Live demo:** _add your Streamlit Cloud link here_
**Author:** Vaishali Singh

## Problem Statement

Lenders need not just a risk score but a *reason* — regulators and applicants
both require explainable decisions, not black-box rejections. This project
builds a loan default classifier and pairs every prediction with a SHAP-based
explanation, mirroring how explainability is used in real credit-risk systems.

## Dataset

- Source: Loan Default dataset (255,347 applicants, 18 columns)
- Target: `Default` (binary) — ~11.6% positive class (imbalanced)
- Features: demographic (age), financial (income, loan amount, credit score,
  DTI ratio, interest rate), employment, and categorical applicant attributes

## Approach

1. **EDA** — univariate/bivariate analysis, class imbalance check
2. **Feature engineering** — ratio features (Loan-to-Income, Interest Burden),
   age/employment/credit-score buckets, one-hot encoding
3. **Modeling** — Logistic Regression baseline vs Random Forest, XGBoost,
   LightGBM; tuned via RandomizedSearchCV; evaluated on ROC-AUC/Precision/Recall/F1
   (not accuracy, due to class imbalance); business-driven threshold instead of 0.5
4. **Explainability** — global SHAP summary (top risk drivers), SHAP dependence
   plots, per-applicant SHAP waterfall, cross-checked against LIME
5. **App** — Streamlit interface with live what-if sliders and inline SHAP plots

## Key Results

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | _fill in_ | | | |
| Random Forest | _fill in_ | | | |
| XGBoost (tuned) | _fill in_ | | | |
| LightGBM | _fill in_ | | | |

_(Run `src/train_model.py` and copy numbers from `models/model_card.md`)_

## SHAP Insight Summary

_Fill in after running `notebooks/02_explainability.py` — top 3-5 global drivers
and whether they match domain intuition (e.g. higher DTI ratio, lower credit
score, higher loan-to-income ratio typically increase default risk)._

## Screenshot / Demo

_Add a screenshot or GIF of the app here._

## Project Structure

```
loan-default-xai/
├── data/
│   ├── raw/               # Loan_default.csv (not committed if large — see .gitignore)
│   └── processed/         # engineered train/val sets (parquet)
├── notebooks/
│   ├── 01_eda.py
│   └── 02_explainability.py
├── src/
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── explain.py
├── app/
│   └── app.py              # Streamlit app
├── models/                 # final_model.pkl, model_card.md, threshold.json
├── requirements.txt
└── README.md
```

## Run Locally

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/feature_engineering.py
python src/train_model.py
python notebooks/02_explainability.py   # optional, for the SHAP/LIME artifacts

streamlit run app/app.py
```

## Disclaimer

This is a portfolio / educational project. It is **not** financial advice and
must not be used for real lending decisions without proper validation,
fairness auditing, and regulatory review.
