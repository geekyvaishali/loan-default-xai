"""
PHASE 4 — Explainability: SHAP + LIME (Days 8-9)

Run:  python notebooks/02_explainability.py
Requires: models/final_model.pkl, data/processed/ (run Phases 2-3 first)

This also writes src/explain.py's core logic (see that file) — this script
is where you EXPLORE, the src/explain.py module is what the app IMPORTS.
"""

# %%
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"
OUT_DIR = os.path.join(os.path.dirname(__file__), "shap_plots")
os.makedirs(OUT_DIR, exist_ok=True)

model = joblib.load(f"{MODELS_DIR}/final_model.pkl")
feature_columns = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")
X_val = pd.read_parquet(f"{PROCESSED_DIR}/X_val.parquet")
X_val = X_val.astype(float)

# %%
# 1. Build a SHAP explainer.
# Auto-detect: tree models (RF/XGBoost/LightGBM) need TreeExplainer,
# linear models (LogisticRegression) need LinearExplainer.
background = shap.sample(X_val, 100, random_state=42)
model_type = type(model).__name__
if model_type == "LogisticRegression":
    explainer = shap.LinearExplainer(model, background)
else:
    explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_val.sample(2000, random_state=42))
X_sample = X_val.sample(2000, random_state=42)

# %%
# 2. Global SHAP summary (beeswarm) — top 10 global risk drivers
shap.summary_plot(shap_values, X_sample, show=False, max_display=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_summary_beeswarm.png", bbox_inches="tight")
plt.close()

# %%
# 3. Get the top 3 features by mean |SHAP value| for dependence plots
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top3_idx = np.argsort(mean_abs_shap)[-3:][::-1]
top3_features = [feature_columns[i] for i in top3_idx]
print("Top 3 global drivers:", top3_features)

for feat in top3_features:
    shap.dependence_plot(feat, shap_values, X_sample, show=False)
    plt.tight_layout()
    safe_name = feat.replace("/", "_")
    plt.savefig(f"{OUT_DIR}/shap_dependence_{safe_name}.png", bbox_inches="tight")
    plt.close()

# %%
# 4. Local explanation for ONE applicant — waterfall plot
idx = 0
row = X_sample.iloc[[idx]]
row_shap = explainer(row)  # new SHAP API, gives an Explanation object
shap.plots.waterfall(row_shap[0], show=False, max_display=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_waterfall_example_applicant.png", bbox_inches="tight")
plt.close()
print("Predicted probability for this applicant:", model.predict_proba(row)[0, 1])

# %%
# 5. Cross-check with LIME on 3-5 individual predictions
from lime.lime_tabular import LimeTabularExplainer

lime_explainer = LimeTabularExplainer(
    training_data=X_val.values,
    feature_names=feature_columns,
    class_names=["No Default", "Default"],
    mode="classification",
    random_state=42
)

for i in range(3):
    row = X_sample.iloc[i]
    exp = lime_explainer.explain_instance(
        row.values, model.predict_proba, num_features=10
    )
    print(f"\n--- LIME explanation, applicant #{i} ---")
    for feat, weight in exp.as_list():
        print(f"  {feat}: {weight:+.4f}")
    exp.save_to_file(f"{OUT_DIR}/lime_applicant_{i}.html")

# %%
# 6. Interpretability note (draft — read the plots and edit in your own words)
note = f"""# Interpretability Note

Top 3 global SHAP drivers: {top3_features}

Talking point for interviews: check whether these line up with domain intuition
(e.g. DTIRatio, InterestRate, CreditScore, LoanToIncomeRatio driving risk UP makes
business sense; if something counter-intuitive shows up, that's worth investigating
and mentioning as a limitation, not hiding).

SHAP vs LIME agreement: open the printed LIME weights above and compare direction
(+/-) and rough ranking against the SHAP waterfall for the same 3 applicants.
Note down here whether they agree or disagree, and why (SHAP is exact/game-theoretic
for tree models; LIME is a local linear approximation — small disagreements are normal,
large disagreements are worth digging into).
"""
with open(f"{OUT_DIR}/../interpretability_note.md", "w") as f:
    f.write(note)
print(note)
print(f"\nAll SHAP/LIME artifacts saved to {OUT_DIR}/")
