import unittest

from src.loanwise import finance, forms


class FinanceTests(unittest.TestCase):
    def test_emi_known_values(self):
        self.assertAlmostEqual(finance.emi(100_000, 12, 12), 8884.88, places=2)
        self.assertAlmostEqual(finance.emi(200_000, 8, 240), 1672.88, places=2)
        self.assertAlmostEqual(finance.emi(12_000, 0, 12), 1000.0)

    def test_schedule_is_consistent(self):
        sched = finance.amortization_schedule(50_000, 9.5, 36)
        self.assertEqual(len(sched), 36)
        self.assertAlmostEqual(sum(r["principal"] for r in sched), 50_000, places=4)
        self.assertAlmostEqual(sched[-1]["balance"], 0.0, places=6)
        s = finance.summary(50_000, 9.5, 36)
        self.assertAlmostEqual(s["total_interest"], sum(r["interest"] for r in sched), places=4)

    def test_max_principal_inverts_emi(self):
        pay = finance.emi(75_000, 11, 48)
        self.assertAlmostEqual(finance.max_principal(pay, 11, 48), 75_000, places=4)

    def test_capacity_never_negative(self):
        self.assertEqual(finance.payment_capacity(12_000, 5_000), 0.0)
        self.assertAlmostEqual(finance.payment_capacity(120_000, 1_000, 0.4), 3_000)

    def test_invalid_inputs_rejected(self):
        for args in [(0, 10, 12), (1000, -1, 12), (1000, 10, 0)]:
            with self.assertRaises(ValueError):
                finance.emi(*args)


class FormTests(unittest.TestCase):
    def test_defaults_are_valid_and_map_to_model_columns(self):
        rec = forms.model_record(forms.validate_all(forms.DEFAULTS))
        self.assertEqual(set(rec), {"Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines",
                                    "InterestRate", "LoanTerm", "DTIRatio", "Education", "EmploymentType",
                                    "MaritalStatus", "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"})

    def test_step_validation_messages(self):
        bad = {**forms.DEFAULTS, "age": 17, "credit_score": "abc", "education": "Wizard"}
        self.assertIn("age", forms.validate_step(1, bad))
        self.assertIn("education", forms.validate_step(1, bad))
        self.assertIn("credit_score", forms.validate_step(2, bad))
        self.assertEqual(forms.validate_step(3, forms.DEFAULTS), {})

    def test_obligations_cannot_exceed_income(self):
        bad = {**forms.DEFAULTS, "annual_income": 12_000, "monthly_obligations": 5_000}
        self.assertIn("monthly_obligations", forms.validate_step(2, bad))

    def test_tenure_limited_to_trained_range(self):
        self.assertIn("loan_term_months", forms.validate_step(3, {**forms.DEFAULTS, "loan_term_months": 84}))

    def test_whole_number_and_missing_values(self):
        self.assertIn("age", forms.validate_step(1, {**forms.DEFAULTS, "age": 30.5}))
        self.assertIn("age", forms.validate_step(1, {**forms.DEFAULTS, "age": None}))

    def test_validate_all_collects_every_error(self):
        with self.assertRaises(forms.ValidationError) as cm:
            forms.validate_all({**forms.DEFAULTS, "age": 5, "loan_amount": 5})
        self.assertTrue({"age", "loan_amount"} <= set(cm.exception.fields))

    def test_dti_formula(self):
        clean = forms.validate_all({**forms.DEFAULTS, "annual_income": 120_000, "monthly_obligations": 1_000})
        d = forms.derived_figures(clean)
        self.assertAlmostEqual(d["dti"], (1_000 + d["emi"]) / 10_000)


if __name__ == "__main__":
    unittest.main()
