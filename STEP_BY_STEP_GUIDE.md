# Loan Default Risk + XAI — VS Code Step-by-Step Guide

Ye guide tumhare roadmap PDF ke hisaab se hai, bas is real dataset (`Loan_default.csv`,
255,347 rows, 18 columns, koi missing value nahi, ~11.6% default rate) ke liye
adjust kiya gaya hai. Home Credit dataset se ye zyada clean hai, isliye Phase 2
thoda halka hai — but explainability aur app wahi depth pe hai.

---

## STEP 0 — Project folder set up karo (5 min)

1. VS Code kholo → `File > Open Folder` → is poore `loan-default-xai` folder ko kholo.
2. VS Code me naya terminal kholo: `` Ctrl+` `` (backtick).
3. Virtual environment banao:

```bash
python -m venv venv
```

4. Activate karo:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

5. VS Code ke bottom-right me Python interpreter select karo → `.venv`/`venv` wala pick karo.

6. Dependencies install karo:

```bash
pip install -r requirements.txt
```

7. Confirm `data/raw/Loan_default.csv` already is present in the folder (maine daal diya hai).

**VS Code Extensions jo helpful honge:** Python (Microsoft), Jupyter (Microsoft) —
inse `# %%` cells ko individually run kar paogi.

---

## STEP 1 — EDA (Phase 1, Days 1-2)

```bash
python notebooks/01_eda.py
```

Ye `notebooks/eda_plots/` folder me saare plots save karega + `notebooks/eda_findings.md`
me summary likhega. VS Code me file explorer se plots kholo, dekh lo patterns, aur
`eda_findings.md` ko apne words me thoda edit kar dena (interview me poochenge
"tumne khud analyze kiya ya bas code chalaya" — so actually look at the plots).

**Tip:** Agar Jupyter-style interactively run karna hai, to same file kholo VS Code me,
har `# %%` ke upar "Run Cell" button dikhega (Jupyter extension ke saath).

---

## STEP 2 — Cleaning + Feature Engineering (Phase 2, Days 3-4)

```bash
python src/feature_engineering.py
```

Isse `data/processed/` me train/val split + engineered features save ho jayenge
(parquet format — pandas + pyarrow use karta hai).

Kya add kiya:
- `LoanToIncomeRatio`, `InterestBurden` — engineered ratio features
- Age / Employment-length / CreditScore buckets
- One-hot encoding sab categoricals ka
- Outlier capping (99th percentile) Income aur LoanAmount pe
- Stratified 80/20 train/val split (class balance preserve hota hai)

`src/feature_engineering.py` ka `engineer_features()` function REUSE hoga
Step 5 (Streamlit app) me bhi — taaki training aur live-prediction ka preprocessing
kabhi mismatch na ho. Ye ek common bug hai jo interviewers puchte hain, so isko
apne project ka strength point bana sakti ho.

---

## STEP 3 — Model Training + Tuning (Phase 3, Days 5-7)

```bash
python src/train_model.py
```

Ye 4 models train karega: Logistic Regression (baseline), Random Forest, XGBoost,
LightGBM — sabka ROC-AUC/Precision/Recall/F1 print karega, best model pick karega
(ROC-AUC se), phir uspe RandomizedSearchCV se hyperparameter tuning karega, 5-fold CV
se stability check karega, aur ek business threshold choose karega (0.5 fixed nahi —
target recall 60% pe precision maximize karta hai).

Output: `models/final_model.pkl`, `models/model_card.md`, `models/feature_columns.pkl`,
`models/threshold.json`, `models/all_model_comparison.json`.

Ye run karne me thoda time lagega (XGBoost tuning ke wajah se, ~5-15 min depending
on tumhare laptop pe). Chai lelo isbeech 🙂

---

## STEP 4 — Explainability: SHAP + LIME (Phase 4, Days 8-9)

```bash
python notebooks/02_explainability.py
```

Ye `notebooks/shap_plots/` me save karega:
- `shap_summary_beeswarm.png` — top 10 global risk drivers
- `shap_dependence_*.png` — top 3 features ke dependence plots
- `shap_waterfall_example_applicant.png` — ek applicant ka local explanation
- `lime_applicant_0/1/2.html` — LIME cross-check same 3 applicants ke liye
- `interpretability_note.md` — draft note (SHAP vs LIME agreement likhna hai)

**Interview ke liye important:** `interpretability_note.md` khud padh ke, apne
words me likho ki top drivers domain-sense me sahi lagte hai ya nahi, aur SHAP/LIME
agree karte hai ya nahi. Ye ek strong talking point hai jo tumhare CV pe already
scripted honest-answer approach ke saath consistent hai.

`src/explain.py` ek separate reusable module hai — ye Step 5 ke app me import
hota hai (SHAP explainer sirf ek baar banta hai, `@st.cache_resource` se, taaki
app slow na ho).

---

## STEP 5 — Streamlit App (Phase 5, Days 10-12)

**Project ke ROOT folder se run karna zaroori hai** (app/ se nahi):

```bash
streamlit run app/app.py
```

Browser me automatically khulega (`localhost:8501`). Features:
- Input form: sab 16 features ke liye (income, loan amount, credit score, etc.)
- What-if sliders: income/loan amount live change karo, prediction re-run hota hai
- Risk probability + Approve/Manual Review/Reject label (threshold-based)
- SHAP waterfall plot us specific applicant ke liye
- Sidebar: model info + disclaimer

Test karna: bahut high income, bahut low income, extreme values daal ke dekho
app crash to nahi karta.

---

## STEP 6 — Deployment + Docs + CV (Phase 6, Days 13-14)

1. **GitHub repo:**
```bash
git init
echo "venv/
data/raw/*.csv
data/processed/
__pycache__/
*.pyc
.DS_Store" > .gitignore
git add .
git commit -m "Initial commit: Loan Default Risk XAI project"
```
   GitHub pe naya repo `loan-default-xai` banao, phir:
```bash
git remote add origin https://github.com/<your-username>/loan-default-xai.git
git branch -M main
git push -u origin main
```
   (Raw CSV bada hai — .gitignore me daal diya hai, README me Kaggle download
   instructions likh dena, ya agar chhota hai to Git LFS use kar sakti ho.)

2. **Streamlit Community Cloud pe deploy:**
   - https://share.streamlit.io pe jao, GitHub se sign in karo
   - "New app" → apna repo select karo → main file path: `app/app.py`
   - Deploy — thoda time lagega pehli baar (dependencies install hongi)
   - **Note:** models/ aur data/processed/ bhi repo me push karne padenge
     (ya app startup pe train_model.py + feature_engineering.py automatically
     chalaane ka setup karna padega — chhote dataset ke liye pkl files push
     karna simpler hai)

3. README.md likho (template neeche diya hai — README.md file already bana di hai,
   usko edit karo apne actual results/screenshots ke saath)

4. Ek 30-60 sec screen recording GIF bana lo app ka (ScreenToGif / LICEcap use kar sakti ho)

5. CV update: project bullets + GitHub link + live app link

6. 60-second verbal pitch practice karo: Problem → Approach → Key Result → Why explainability mattered
   (Ye tumhare interview prep guide ke SHAP/LIME section ke saath directly connect
   hoga — same honest framing use karna jo already scripted hai.)

---

## Common issues

- **"No module named shap/lime/xgboost"** → venv activate hai confirm karo, phir
  `pip install -r requirements.txt` phir se chalao.
- **Parquet read error** → `pip install pyarrow` (already requirements.txt me hai).
- **Streamlit app "file not found: models/final_model.pkl"** → Step 3 pehle
  chalao, aur app ko project ROOT se run karo, `app/` folder ke andar se nahi.
- **SHAP waterfall plot blank/error in app** → check `src/explain.py` ka
  `get_explainer()` — ye `TreeExplainer` use karta hai, agar best model
  LogisticRegression nikla to `shap.LinearExplainer` use karna padega
  (comment already code me hai).
