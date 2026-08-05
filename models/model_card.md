# Model Card — Loan Default Risk

**Algorithm:** LogisticRegression
**Validation ROC-AUC:** 0.7567
**Validation Precision / Recall / F1:** 0.2314 / 0.6704 / 0.3441
**5-fold CV ROC-AUC:** 0.7444 ± 0.0038
**Business decision threshold:** 0.553 (probability >= threshold -> flag for manual review)
**Class imbalance handling:** class_weight='balanced' / scale_pos_weight
**Known limitations:** trained on a single static snapshot dataset; no time-based
validation was possible (no application-date column); should be re-validated
periodically against real approval outcomes before any production use.
