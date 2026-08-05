"""
PHASE 1 — EDA (Days 1-2)
Run this file directly, OR paste each `# %%` block into a Jupyter cell
if you're using VS Code's Jupyter extension (recommended — you get inline plots).

How to run in VS Code:
1. Open this file in VS Code
2. If you have the "Jupyter" + "Python" extensions installed, each `# %%`
   becomes a runnable cell (you'll see "Run Cell" above each block)
3. Or just run: python notebooks/01_eda.py   (plots will save to notebooks/eda_plots/)
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

OUT_DIR = os.path.join(os.path.dirname(__file__), "eda_plots")
os.makedirs(OUT_DIR, exist_ok=True)

# %%
# 1. Load data
df = pd.read_csv("data/raw/Loan_default.csv")
print("Shape:", df.shape)
df.head()

# %%
# 2. dtypes + missing value audit
print(df.dtypes)
print("\nMissing values per column:\n", df.isnull().sum())
# This dataset has ZERO missing values — good news, Phase 2 cleaning will be lighter
# than the roadmap assumed (that roadmap was written for Home Credit's messier dataset)

# %%
# 3. Target distribution
target_dist = df["Default"].value_counts(normalize=True) * 100
print(target_dist)
target_dist.plot(kind="bar", color=["#2E86AB", "#E63946"])
plt.title("Default vs Non-Default Distribution (%)")
plt.xticks([0, 1], ["No Default (0)", "Default (1)"], rotation=0)
plt.ylabel("% of applicants")
plt.savefig(f"{OUT_DIR}/target_distribution.png", bbox_inches="tight")
plt.close()
# ~11.6% default rate -> this IS an imbalanced classification problem.
# Note this down for the model card later: accuracy alone will be misleading.

# %%
# 4. Univariate EDA — numeric columns
numeric_cols = ["Age", "Income", "LoanAmount", "CreditScore",
                 "MonthsEmployed", "NumCreditLines", "InterestRate",
                 "LoanTerm", "DTIRatio"]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="#2E86AB")
    ax.set_title(col)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/univariate_numeric.png", bbox_inches="tight")
plt.close()

# %%
# 5. Bivariate EDA — default rate by categorical columns
cat_cols = ["Education", "EmploymentType", "MaritalStatus",
            "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"]

fig, axes = plt.subplots(4, 2, figsize=(14, 18))
for ax, col in zip(axes.flatten(), cat_cols):
    rate = df.groupby(col)["Default"].mean().sort_values(ascending=False) * 100
    rate.plot(kind="bar", ax=ax, color="#E63946")
    ax.set_title(f"Default Rate (%) by {col}")
    ax.set_ylabel("Default rate %")
axes.flatten()[-1].axis("off")  # 7 plots in an 8-slot grid
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/default_rate_by_category.png", bbox_inches="tight")
plt.close()

# %%
# 6. Default rate by income / credit-amount buckets
df["IncomeBucket"] = pd.qcut(df["Income"], 5, labels=["Very Low", "Low", "Mid", "High", "Very High"])
df["LoanAmountBucket"] = pd.qcut(df["LoanAmount"], 5, labels=["Very Low", "Low", "Mid", "High", "Very High"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df.groupby("IncomeBucket")["Default"].mean().plot(kind="bar", ax=axes[0], color="#2E86AB")
axes[0].set_title("Default Rate by Income Bucket")
df.groupby("LoanAmountBucket")["Default"].mean().plot(kind="bar", ax=axes[1], color="#2E86AB")
axes[1].set_title("Default Rate by Loan Amount Bucket")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/bucket_default_rates.png", bbox_inches="tight")
plt.close()

# %%
# 7. Correlation heatmap (numeric features vs target)
corr = df[numeric_cols + ["Default"]].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", bbox_inches="tight")
plt.close()

# %%
# 8. Write EDA findings summary (auto-generated draft — edit in your own words after reading plots)
top_corr = corr["Default"].drop("Default").abs().sort_values(ascending=False)
findings = f"""# EDA Findings — Loan Default Dataset

- Rows: {df.shape[0]:,} | Columns: {df.shape[1]}
- No missing values in any column.
- Class imbalance: {target_dist[0]:.1f}% no-default vs {target_dist[1]:.1f}% default.
  -> Must use class_weight='balanced' or SMOTE, and evaluate with ROC-AUC / F1, not accuracy.
- Top 5 numeric features correlated with Default (by |correlation|):
{top_corr.head(5).to_string()}
- Check the bar charts in eda_plots/default_rate_by_category.png for which
  Education / EmploymentType / MaritalStatus groups default more.
- Check bucket_default_rates.png — typically lower income & higher loan amount
  buckets show higher default rate (confirm this from YOUR plot before writing it in your report).
"""
with open(f"{OUT_DIR}/../eda_findings.md", "w") as f:
    f.write(findings)
print(findings)
print(f"\nAll plots saved to: {OUT_DIR}/")
