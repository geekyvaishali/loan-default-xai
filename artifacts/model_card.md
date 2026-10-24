# Model card: Loan Default Risk (educational)

**Version:** 20261023-193630-1d7556a  |  **Trained (UTC):** 2026-10-23T19:36:30+00:00
**Algorithm:** logistic_regression (+ sigmoid probability calibration)
**Why this model:** best CV roc_auc is logistic_regression (0.7556); one-standard-error rule picked the simplest model within 1 std of it (logistic_regression eligible)

## Data
- 255,347 rows, default rate 11.61%; file `Loan_default.csv` (sha256 `1d7556a9071e...`).
- Split: train 153,207 / validation 51,070 / test 51,070 (stratified, seed 42).
- Duplicate rows: 0; null cells: 0.
- **Synthetic-data warning:** numeric columns are almost perfectly uniform (flatness check: True). This strongly suggests a synthetic dataset, so results do not describe real lending behaviour.

## Test-set results (touched once, after all choices were made)
- ROC-AUC 0.7616 (95% bootstrap CI 0.7557 to 0.7675); PR-AUC 0.3363 (a no-skill model scores about 0.116).
- Brier 0.0903 vs 0.1026 for always predicting the base rate; ECE 0.0036 (before calibration: Brier 0.0903, ECE 0.0035).
- At threshold 0.1421 (highest precision subject to recall >= 0.60 on the validation split): precision 0.258, recall 0.613, F1 0.363, flagged 27.6% of applicants.

- Calibration changed little: the model was already close to calibrated (Brier moved by 0.0000). It is kept because it is cheap, preserves ranking, and matters more for tree models.

## Intended use
Portfolio / education only. Not validated for real credit decisions, not regulatory-compliant.

## Limitations
- Hyperparameters are fixed, regularised values, not extensively tuned; model differences are within noise (see CV std).
- Age, marital status and education are model inputs. Many jurisdictions restrict such attributes in credit decisions; no fairness audit was possible here (no protected-group labels, no outcome data by group).
- Single static snapshot: no time-based validation, no drift history, no real-world outcome feedback.
- Probabilities are calibrated on one validation split of this dataset only.
