"""Integration: train on synthetic data -> save artifacts -> load -> score."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import TrainConfig
from src.predict import LoanRiskPredictor
from src.schema import SchemaError
from src.synthetic import make_synthetic_applicants
from src.train import run_training


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        data = make_synthetic_applicants(6000, seed=7, with_target=True)
        data.insert(0, "LoanID", [f"A{i}" for i in range(len(data))])
        data.to_csv(root / "train.csv", index=False)
        cfg = TrainConfig(models=("logistic_regression", "hist_gradient_boosting"), cv_folds=2)
        cls.result = run_training(cfg, root / "train.csv", root / "art")
        cls.art = root / "art"
        cls.pred = LoanRiskPredictor.load(cls.art)
        cls.batch = make_synthetic_applicants(30, seed=8)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_artifacts_written(self):
        for f in ["model.joblib", "metadata.json", "schema.json", "evaluation.json",
                  "data_quality.json", "reference_stats.json", "history.jsonl", "model_card.md"]:
            self.assertTrue((self.art / f).exists(), f)

    def test_splits_are_disjoint_in_size_and_stratified(self):
        s = self.result["metadata"]["splits"]
        self.assertEqual(s["train"] + s["val"] + s["test"], 6000)
        self.assertAlmostEqual(s["train_default_rate"], s["test_default_rate"], delta=0.01)

    def test_metadata_records_reproducibility_info(self):
        m = json.loads((self.art / "metadata.json").read_text(encoding="utf-8"))
        for k in ["model_version", "selected_model", "threshold", "data", "environment", "config"]:
            self.assertIn(k, m)
        self.assertEqual(len(m["data"]["sha256"]), 64)

    def test_score_shape_and_probability_range(self):
        out = self.pred.score(self.batch)
        self.assertEqual(len(out), 30)
        self.assertTrue(out["probability"].between(0, 1).all())
        self.assertTrue((out["status"] == "ok").all())

    def test_flag_follows_configurable_threshold(self):
        lo = LoanRiskPredictor.load(self.art, threshold=0.0).score(self.batch)["flagged"].astype(bool)
        hi = LoanRiskPredictor.load(self.art, threshold=1.01).score(self.batch)["flagged"].astype(bool)
        self.assertTrue(lo.all())
        self.assertFalse(hi.any())

    def test_invalid_rows_raise_or_are_marked(self):
        bad = self.batch.copy()
        bad.loc[bad.index[2], "Education"] = "Wizard"
        with self.assertRaises(SchemaError):
            self.pred.score(bad)
        out = self.pred.score(bad, on_invalid="mark")
        self.assertTrue(np.isnan(out["probability"].iloc[2]))
        self.assertTrue(out["status"].iloc[2].startswith("invalid"))
        self.assertEqual((out["status"] == "ok").sum(), 29)

    def test_missing_column_raises(self):
        with self.assertRaises(SchemaError):
            self.pred.score(self.batch.drop(columns=["Age"]))

    def test_extra_columns_ignored_and_batch_independent(self):
        extra = self.batch.copy()
        extra["LoanID"] = "x"
        a = self.pred.score(extra)["probability"].to_numpy()
        b = np.array([self.pred.score(self.batch.iloc[[i]])["probability"].iloc[0] for i in range(30)])
        np.testing.assert_allclose(a, b, rtol=1e-9)

    def test_predict_one(self):
        rec = self.batch.iloc[0].to_dict()
        res = self.pred.predict_one(rec)
        self.assertTrue(0 <= res["probability"] <= 1)
        self.assertEqual(res["model_version"], self.pred.version)

    def test_loading_from_missing_dir_gives_helpful_error(self):
        with self.assertRaises(FileNotFoundError):
            LoanRiskPredictor.load(self.art / "nope")


if __name__ == "__main__":
    unittest.main()
