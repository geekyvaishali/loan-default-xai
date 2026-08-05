# EDA Findings — Loan Default Dataset

- Rows: 255,347 | Columns: 20
- No missing values in any column.
- Class imbalance: 88.4% no-default vs 11.6% default.
  -> Must use class_weight='balanced' or SMOTE, and evaluate with ROC-AUC / F1, not accuracy.
- Top 5 numeric features correlated with Default (by |correlation|):
Age               0.167783
InterestRate      0.131273
Income            0.099119
MonthsEmployed    0.097374
LoanAmount        0.086659
- Check the bar charts in eda_plots/default_rate_by_category.png for which
  Education / EmploymentType / MaritalStatus groups default more.
- Check bucket_default_rates.png — typically lower income & higher loan amount
  buckets show higher default rate (confirm this from YOUR plot before writing it in your report).
