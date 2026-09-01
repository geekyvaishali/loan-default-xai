# Loan Default Risk Predictor with Explainable AI

Predicts an applicant's probability of loan default and explains *why* using
SHAP and LIME, wrapped in an interactive Streamlit app.

**Live demo:**https://loan-default-xai-poguzlzskanu6scamkbk3j.streamlit.app/
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
| Logistic Regression | 0.7571 | 0.2284 | 0.6854 | 0.3426 |
| Random Forest | 0.7444 | 0.6268 | 0.0292 | 0.0557 |
| XGBoost | 0.7548 | 0.2350 | 0.6528 | 0.3456 |
| LightGBM | 0.7567 | 0.2314 | 0.6704 | 0.3441 |

**Selected model:** Logistic Regression (highest validation ROC-AUC)
_(Run `src/train_model.py` and copy numbers from `models/model_card.md`)_

## SHAP Insight Summary

The SHAP analysis identified **Age**, **InterestRate**, and **LoanToIncomeRatio**
as the top 3 global drivers of default risk:

- **InterestRate**: Higher interest rates strongly increase predicted default
  risk — consistent with lenders pricing riskier applicants at higher rates.
- **LoanToIncomeRatio** (engineered feature): A higher loan amount relative to
  income increases risk, confirming that this custom ratio adds real predictive
  signal beyond the raw features.
- **Age**: Age has a strong influence on predicted risk, reflecting patterns
  around credit history length and income stability across age groups.

These findings align with standard credit-risk intuition, giving confidence
that the model is learning genuine patterns rather than spurious correlations.
Cross-checking with LIME on individual applicants showed broadly consistent
attributions, with minor differences expected since LIME approximates locally
while SHAP computes exact contributions for tree-based/linear models.

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
