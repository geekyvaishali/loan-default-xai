import unittest

import pandas as pd

from src.schema import SchemaError, data_quality_report, validate_frame
from src.synthetic import make_synthetic_applicants


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_applicants(50, seed=1)

    def test_valid_data_passes(self):
        r = validate_frame(self.df)
        self.assertTrue(r.ok)
        self.assertEqual(r.n_rows, 50)

    def test_missing_column_reported(self):
        r = validate_frame(self.df.drop(columns=["Income"]))
        self.assertEqual(r.missing_columns, ["Income"])
        self.assertFalse(r.ok)

    def test_unknown_category_is_an_error(self):
        self.df.loc[3, "Education"] = "Wizard"
        r = validate_frame(self.df)
        self.assertIn(3, r.row_errors)
        self.assertTrue(any("Education" in m for m in r.row_errors[3]))

    def test_null_and_non_numeric_are_errors(self):
        self.df["Age"] = self.df["Age"].astype(object)
        self.df.loc[0, "Age"] = None
        self.df.loc[1, "Age"] = "abc"
        r = validate_frame(self.df)
        self.assertIn(0, r.row_errors)
        self.assertIn(1, r.row_errors)

    def test_hard_bound_error_vs_training_range_warning(self):
        self.df.loc[0, "CreditScore"] = 9999      # impossible -> error
        self.df.loc[1, "NumCreditLines"] = 10     # possible but unseen in training -> warning
        r = validate_frame(self.df)
        self.assertIn(0, r.row_errors)
        self.assertNotIn(1, r.row_errors)
        self.assertIn(1, r.row_warnings)

    def test_validation_does_not_mutate_input(self):
        before = self.df.copy()
        validate_frame(self.df)
        self.assertTrue(before.equals(self.df))

    def test_data_quality_report_counts_duplicates(self):
        d = make_synthetic_applicants(30, seed=2, with_target=True)
        d = pd.concat([d, d.iloc[[0]]], ignore_index=True)
        rep = data_quality_report(d)
        self.assertEqual(rep["n_duplicate_rows_excluding_id"], 1)
        self.assertEqual(rep["n_null_cells"], 0)


if __name__ == "__main__":
    unittest.main()
