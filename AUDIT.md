# Audit report: Loan Default Risk Predictor

Every number below was produced by running code on the repository you uploaded (255,347 rows, seed 42 split).
Environment limits for this audit: no network, no `xgboost`/`lightgbm`/`shap`/`lime`/`streamlit`/`fastapi`/`pyarrow`, so those parts were read, not run.

## Current architecture
`data/raw/Loan_default.csv` -> `feature_engineering.py` (engineer + `get_dummies`, saves parquet) -> `train_model.py` (4 models, 20% validation split, saves pickle) -> `app/app.py` (re-implements encoding, loads pickle) with `src/explain.py` (SHAP, imports Streamlit).

## What is already good
- Honest scope: README disclaims real-lending use; imbalance is acknowledged; ROC-AUC/F1 used instead of accuracy.
- Stratified split, fixed seeds, several model families compared, threshold chosen from a precision/recall target instead of 0.5.
- The reported AUC is reproducible: the shipped model scores **0.7571** on its own validation split, matching the README.
- Sensible EDA, SHAP + LIME exploration, usable UI structure, presets, batch scoring, download buttons.

## Critical bugs and misleading results (all verified)
| # | Finding | Evidence |
|---|---|---|
| 1 | `app/app.py` is tracked in git but **deleted in the working tree**; the current app was saved as `notepad appapp.py.txt`. A careless `git add -A` would delete the deployed app. | `git status` shows ` D app/app.py`, `?? notepad appapp.py.txt` |
| 2 | **The displayed "default probability" is not a probability.** `class_weight='balanced'` inflates scores. | Shipped model: mean score 0.425 vs real default rate 0.116; Brier 0.199, *worse* than predicting the base rate (0.103) |
| 3 | `model_card.md` reports **another model's metrics**: it prints `results[-1]` (LightGBM's row) while naming LogisticRegression. | 0.7567 / 0.2314 / 0.6704 / 0.3441 equals the LightGBM row in `all_model_comparison.json` |
| 4 | Feature engineering leaks and skews. Caps and medians are computed on whatever batch is passed: on the full data before the split, and a single applicant is never capped. | Income 250,000 stays 250,000 for one row (full-data cap is 148,661); batch results depend on batch composition |
| 5 | One validation split is used for model choice, threshold choice **and** reported metrics; there is no untouched test set. | `train_model.py` |
| 6 | README claims differ from the code: "tuned via RandomizedSearchCV" (tuning only runs if XGBoost wins; Logistic Regression won, so nothing was tuned) and "live what-if sliders" (the app is a plain input form). | `train_model.py` L~95, `app/app.py` |
| 7 | Arbitrary decision bands: `REJECT` at threshold + 0.2, applied to uncalibrated scores. 6.5% of applicants are auto-labelled REJECT. Presenting approve/reject output is also the wrong framing for an educational model. | `app/app.py`, measured on validation split |
| 8 | UI allows inputs far outside training data with no warning (credit lines up to 20 vs max 4 in data; months employed to 480 vs 119; age to 75 vs 69; presets use 6 lines / 180 months). | training ranges from data |
| 9 | `model_card.md` is written without UTF-8 encoding, so characters are corrupted. | file shows `Model Card � Loan Default` |

## ML and statistical weaknesses
- **The dataset looks synthetic.** Numeric columns are almost perfectly uniform (histogram flatness CV: Income 0.004, MonthsEmployed 0.003, InterestRate 0.007, CreditScore 0.007). The ~0.76 AUC ceiling and "Age is the top driver" reflect how the data was generated, not lending behaviour. Say so in interviews before a reviewer finds it.
- No duplicates, no nulls (checked). Differences between candidate models are small (CV AUC 0.7556 / 0.7513 / 0.7469) and, apart from the weaker random forest, mostly within noise.
- Age, marital status and education are used as predictors; many jurisdictions restrict such attributes in credit decisions. No fairness check exists, and none is possible without group-outcome data.
- Unscaled logistic regression with `max_iter=1000` (convergence risk).

## Missing industry-relevant features
Input validation, calibration analysis, PR-AUC, tests, versioned artifacts with metadata, drift checks, API layer, CI, a real model card, a sample input file, unpinned/unused dependencies (`imbalanced-learn` is never imported; `plotly` is listed twice).

## Done (verified by running it)
* **Phase 1, ML core:** `config`, `schema` (validation + data-quality report), `features` (leak-safe transformer), `pipeline` (one raw-in/probability-out pipeline + calibrator), `evaluate`, `train` (60/20/20 split, CV on train only, one-standard-error model selection, calibration + threshold on validation only, single test evaluation), `predict`, `attribution` (exact linear contributions, replacement fallback for other models), versioned `artifacts/` and a generated model card.
* **Phase 2, LoanWise:** customer and officer portals (`app/loanwise_app.py`, `src/loanwise/`), scrypt password hashing, hashed server-side sessions with idle/absolute expiry and lockout, role checks inside the service layer, customer isolation, application status machine with mandatory decision reasons, audit trail, batch CSV assessment with size/row limits, downloadable report, demo seeding.
* 88 automated tests pass (`python -m unittest discover -s tests -t .`), including access-control, session, workflow, persistence and batch tests. A real-model end-to-end run (seed, customer submit, officer decision, restart) also passed.

## Not done / not verified
* **The Streamlit pages have no automated tests** and were not exercised in a browser during development; they call only the tested service layer.
* `xgboost` / `lightgbm` candidates are coded but were not run during development (not installed there).
* No FastAPI layer, Docker setup, live drift monitoring, or fairness assessment (the data has no protected-group labels).
* The GitHub Actions workflow has not been observed running on GitHub.
