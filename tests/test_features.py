import unittest

import numpy as np
import pandas as pd

from src.features import LoanFeatureEngineer
from src.synthetic import make_synthetic_applicants


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.train = make_synthetic_applicants(2000, seed=3)
        self.fe = LoanFeatureEngineer().fit(self.train)

    def test_caps_are_learned_from_training_data_only(self):
        cap = self.fe.caps_["Income"]
        row = self.train.iloc[[0]].copy()
        row["Income"] = 10 * cap
        out = self.fe.transform(row)
        self.assertAlmostEqual(out["Income"].iloc[0], cap)   # old code: a single row was never capped

    def test_batch_equals_row_by_row(self):
        """Scoring a batch must not depend on the other rows in it."""
        batch = make_synthetic_applicants(40, seed=4)
        whole = self.fe.transform(batch).reset_index(drop=True)
        rows = pd.concat([self.fe.transform(batch.iloc[[i]]) for i in range(len(batch))], ignore_index=True)
        pd.testing.assert_frame_equal(whole, rows)

    def test_ratio_and_burden(self):
        row = self.train.iloc[[0]].copy()
        row[["Income", "LoanAmount", "InterestRate"]] = [50_000, 25_000, 10.0]
        out = self.fe.transform(row)
        self.assertAlmostEqual(out["LoanToIncomeRatio"].iloc[0], 0.5)
        self.assertAlmostEqual(out["InterestBurden"].iloc[0], 2500.0)

    def test_bucket_edges(self):
        row = self.train.iloc[[0]].copy()
        row[["Age", "MonthsEmployed", "CreditScore"]] = [25, 12, 579]
        out = self.fe.transform(row)
        self.assertEqual(out["AgeBucket"].iloc[0], "18_25")
        self.assertEqual(out["EmploymentLengthBucket"].iloc[0], "under_1yr")
        self.assertEqual(out["CreditScoreBucket"].iloc[0], "Poor")

    def test_out_of_range_values_still_get_a_bucket(self):
        row = self.train.iloc[[0]].copy()
        row[["Age", "CreditScore"]] = [95, 850]
        out = self.fe.transform(row)
        self.assertEqual(out["AgeBucket"].iloc[0], "56_plus")
        self.assertNotEqual(out["CreditScoreBucket"].iloc[0], "nan")

    def test_missing_column_raises(self):
        with self.assertRaises(KeyError):
            self.fe.transform(self.train.drop(columns=["Age"]))


if __name__ == "__main__":
    unittest.main()
