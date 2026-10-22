import re
import unittest

import numpy as np

from src import evaluate as ev
from src.pipeline import ProbabilityCalibrator, build_pipeline, available_models
from src.synthetic import make_synthetic_applicants


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = make_synthetic_applicants(4000, seed=5, with_target=True)
        cls.y = d.pop("Default")
        cls.X = d
        cls.pipe = build_pipeline("logistic_regression").fit(cls.X, cls.y)

    def test_batch_prediction_equals_single_row_prediction(self):
        """Train/serve consistency: the same pipeline gives the same score either way."""
        sample = self.X.head(25)
        batch = self.pipe.predict_proba(sample)[:, 1]
        single = np.array([self.pipe.predict_proba(sample.iloc[[i]])[0, 1] for i in range(25)])
        np.testing.assert_allclose(batch, single, rtol=1e-9)

    def test_probabilities_in_range_and_right_shape(self):
        p = self.pipe.predict_proba(self.X.head(100))
        self.assertEqual(p.shape, (100, 2))
        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_feature_names_safe_for_xgboost_and_lightgbm(self):
        cols = self.pipe[:-1].transform(self.X.head(2)).columns
        self.assertTrue(all(re.fullmatch(r"[0-9A-Za-z_]+", c) for c in cols), list(cols))

    def test_unknown_category_does_not_crash_the_encoder(self):
        row = self.X.head(1).copy()
        row["Education"] = "Unseen"
        self.assertEqual(self.pipe.predict_proba(row).shape, (1, 2))

    def test_unknown_model_name_rejected(self):
        with self.assertRaises(ValueError):
            build_pipeline("nope")

    def test_missing_optional_libs_are_skipped_not_fatal(self):
        usable, skipped = available_models(("logistic_regression", "not_a_model"))
        self.assertEqual(usable, ["logistic_regression"])
        self.assertIn("not_a_model", skipped)

    def test_calibrators_output_valid_monotone_probabilities(self):
        rng = np.random.default_rng(0)
        s = rng.uniform(0.01, 0.99, 3000)
        y = (rng.random(3000) < s ** 2).astype(int)
        for method in ("isotonic", "sigmoid"):
            cal = ProbabilityCalibrator(method).fit(s, y)
            grid = np.linspace(0, 1, 50)
            out = cal.predict(grid)
            self.assertTrue(((out >= 0) & (out <= 1)).all())
            self.assertTrue((np.diff(out) >= -1e-12).all())


class EvaluateTests(unittest.TestCase):
    def test_threshold_metrics_counts(self):
        y = np.array([1, 1, 0, 0, 0, 1])
        p = np.array([0.9, 0.4, 0.6, 0.2, 0.1, 0.8])
        m = ev.threshold_metrics(y, p, 0.5)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 1, 2))
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)

    def test_recall_target_policy_meets_target(self):
        rng = np.random.default_rng(1)
        y = (rng.random(5000) < 0.12).astype(int)
        p = np.clip(0.12 + 0.3 * y + rng.normal(0, 0.15, 5000), 0.001, 0.999)
        t = ev.select_threshold(y, p, "recall_target", target_recall=0.6)
        self.assertGreaterEqual(ev.threshold_metrics(y, p, t["threshold"])["recall"], 0.6)

    def test_cost_policy_prefers_lower_threshold_when_misses_are_costly(self):
        rng = np.random.default_rng(2)
        y = (rng.random(5000) < 0.12).astype(int)
        p = np.clip(0.12 + 0.3 * y + rng.normal(0, 0.15, 5000), 0.001, 0.999)
        cheap = ev.select_threshold(y, p, "cost", cost_fn=1, cost_fp=1)["threshold"]
        costly = ev.select_threshold(y, p, "cost", cost_fn=10, cost_fp=1)["threshold"]
        self.assertLess(costly, cheap)

    def test_calibration_table_covers_all_rows(self):
        rng = np.random.default_rng(3)
        p = rng.random(1000)
        y = (rng.random(1000) < p).astype(int)
        self.assertEqual(sum(r["count"] for r in ev.calibration_table(y, p)), 1000)
        self.assertLess(ev.expected_calibration_error(y, p), 0.08)

    def test_bad_policy_rejected(self):
        with self.assertRaises(ValueError):
            ev.select_threshold(np.array([0, 1]), np.array([0.1, 0.9]), "magic")


if __name__ == "__main__":
    unittest.main()
