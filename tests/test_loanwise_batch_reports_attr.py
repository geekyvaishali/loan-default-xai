import io
import unittest

import numpy as np
import pandas as pd

from src import attribution
from src.config import RAW_FEATURES
from src.loanwise import reports
from src.loanwise.batch import BatchError, score_csv
from src.loanwise.forms import DEFAULTS, model_record, validate_all
from src.synthetic import make_synthetic_applicants
from tests.helpers import Env, predictor_and_dir


def _csv(df) -> bytes:
    return df.to_csv(index=False).encode()


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred, _ = predictor_and_dir()

    def test_valid_file_scores_every_row(self):
        out, s = score_csv(self.pred, _csv(make_synthetic_applicants(40, seed=1)))
        self.assertEqual((len(out), s["scored"], s["invalid"]), (40, 40, 0))
        self.assertTrue(out["probability"].between(0, 1).all())

    def test_error_cases_have_friendly_messages(self):
        good = make_synthetic_applicants(5, seed=2)
        cases = {"empty": b"", "whitespace": b"  \n ", "no rows": _csv(good.head(0)),
                 "missing cols": _csv(good.drop(columns=["Age", "Income"])), "garbage": b"\x00\x01\x02\xff\xfe"}
        for name, raw in cases.items():
            with self.assertRaises(BatchError, msg=name):
                score_csv(self.pred, raw)
        with self.assertRaises(BatchError) as cm:
            score_csv(self.pred, _csv(good.drop(columns=["Age"])))
        self.assertIn("Age", str(cm.exception))

    def test_size_and_row_limits(self):
        good = _csv(make_synthetic_applicants(30, seed=3))
        with self.assertRaises(BatchError):
            score_csv(self.pred, good, max_bytes=100)
        with self.assertRaises(BatchError):
            score_csv(self.pred, good, max_rows=10)

    def test_invalid_rows_are_flagged_not_fatal(self):
        df = make_synthetic_applicants(10, seed=4)
        df.loc[3, "Education"] = "Wizard"
        df.loc[5, "CreditScore"] = 99999
        out, s = score_csv(self.pred, _csv(df))
        self.assertEqual((s["scored"], s["invalid"]), (8, 2))
        self.assertTrue(out.loc[3, "status"].startswith("invalid"))


class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred, cls.art = predictor_and_dir()
        cls.row = pd.DataFrame([model_record(validate_all(DEFAULTS))])[RAW_FEATURES]

    def test_feature_group_folds_encoded_and_bucket_columns(self):
        g = attribution.feature_group
        self.assertEqual(g("Age"), "Age")
        self.assertEqual(g("AgeBucket_26_35"), "Age")
        self.assertEqual(g("CreditScoreBucket_Poor"), "CreditScore")
        self.assertEqual(g("EmploymentLengthBucket_under_1yr"), "MonthsEmployed")
        self.assertEqual(g("EmploymentType_Full_time"), "EmploymentType")
        self.assertEqual(g("Education_Bachelor_s"), "Education")
        self.assertEqual(g("LoanToIncomeRatio"), "LoanToIncomeRatio")

    def test_linear_contributions_are_additive_and_exact(self):
        """base + sum(contributions) must equal the model's own log-odds for this applicant."""
        if type(self.pred.model.pipeline[-1]).__name__ != "LogisticRegression":
            self.skipTest("selected model is not linear")
        info = self.pred.schema["explain"]
        for seed in range(5):
            rec = make_synthetic_applicants(1, seed=seed + 50).iloc[0].to_dict()
            row = pd.DataFrame([rec])[RAW_FEATURES]
            res = attribution.linear_contributions(self.pred.model, row, info["transformed_means"])
            logit = float(self.pred.model.pipeline[-1].decision_function(self.pred.model.pipeline[:-1].transform(row))[0])
            self.assertAlmostEqual(res["base_logit"] + sum(res["contributions"].values()), logit, places=8)

    def test_replacement_fallback_works_for_non_linear_models(self):
        from src.config import TrainConfig
        from src.pipeline import build_pipeline, CalibratedRiskModel, ProbabilityCalibrator
        d = make_synthetic_applicants(3000, seed=9, with_target=True)
        y = d.pop("Default")
        pipe = build_pipeline("hist_gradient_boosting").fit(d, y)
        model = CalibratedRiskModel(pipe, ProbabilityCalibrator("sigmoid").fit(pipe.predict_proba(d)[:, 1], y), "hgb")
        numeric = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        ref = {**{c: float(d[c].median()) for c in numeric},
               **{c: d[c].mode().iloc[0] for c in d.columns if c not in numeric}}
        out = attribution.explain_row(model, self.row, {"explain": {"reference_values": ref}})
        self.assertFalse(out["additive"])
        self.assertTrue(out["factors"])

    def test_explain_row_output_shape_and_ordering(self):
        out = self.pred.explain(model_record(validate_all(DEFAULTS)))
        mags = [abs(f["contribution"]) for f in out["factors"]]
        self.assertEqual(mags, sorted(mags, reverse=True))
        self.assertTrue(all(f["direction"] in ("raises", "lowers") and f["label"] for f in out["factors"]))

    def test_higher_interest_rate_raises_modelled_risk_for_this_model(self):
        low = self.pred.predict_one(model_record(validate_all({**DEFAULTS, "interest_rate": 3.0})))["probability"]
        high = self.pred.predict_one(model_record(validate_all({**DEFAULTS, "interest_rate": 24.0})))["probability"]
        self.assertGreater(high, low)


class ReportTests(unittest.TestCase):
    def test_report_contains_disclaimer_assumptions_and_escapes_html(self):
        e = Env()
        try:
            app_id = e.submit_for(e.alice)
            detail = e.apps.my_application(e.alice, app_id)
            detail["assessment"]["factors"][0]["label"] = "<script>alert(1)</script>"
            html = reports.customer_report_html(detail)
            self.assertIn("NOT a loan offer", html)
            self.assertIn("assumption", html)
            self.assertNotIn("<script>alert(1)</script>", html)
            self.assertIn(detail["reference"], html)
        finally:
            e.close()


if __name__ == "__main__":
    unittest.main()
