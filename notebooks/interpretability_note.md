# Interpretability Note

Top 3 global SHAP drivers: ['Age', 'InterestRate', 'LoanToIncomeRatio']

Talking point for interviews: check whether these line up with domain intuition
(e.g. DTIRatio, InterestRate, CreditScore, LoanToIncomeRatio driving risk UP makes
business sense; if something counter-intuitive shows up, that's worth investigating
and mentioning as a limitation, not hiding).

SHAP vs LIME agreement: open the printed LIME weights above and compare direction
(+/-) and rough ranking against the SHAP waterfall for the same 3 applicants.
Note down here whether they agree or disagree, and why (SHAP is exact/game-theoretic
for tree models; LIME is a local linear approximation — small disagreements are normal,
large disagreements are worth digging into).
