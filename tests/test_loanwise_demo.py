import tempfile
import unittest
from pathlib import Path

from src.loanwise import demo
from src.loanwise.applications import STATUSES, ApplicationService
from src.loanwise.auth import AuthService
from src.loanwise.db import Database
from tests.helpers import predictor_and_dir


class DemoSeedTests(unittest.TestCase):
    """The seeder drives the real service layer on a simulated clock spanning ~60 days, so it also
    guards against session idle-timeouts breaking long simulated workflows."""

    @classmethod
    def setUpClass(cls):
        cls.pred, _ = predictor_and_dir()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = Database(Path(cls.tmp.name) / "demo.db")
        cls.out = demo.seed(cls.db, cls.pred, n_applications=30, seed_value=11)
        cls.auth = AuthService(cls.db)
        cls.apps = ApplicationService(cls.db, cls.auth, cls.pred)
        cls.officer = cls.auth.login(demo.DEMO_OFFICER, demo.DEMO_PASSWORD, "officer")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_seed_creates_flagged_synthetic_records_through_the_real_workflow(self):
        rows = self.apps.list_applications(self.officer)
        self.assertEqual(len(rows), 30)
        self.assertTrue(all(r["is_demo"] for r in rows))
        self.assertTrue(self.apps.overview(self.officer)["has_demo_records"])
        self.assertTrue(all(r["status"] in STATUSES and r["status"] != "Draft" for r in rows))

    def test_decided_demo_applications_have_reasons_and_audit_events(self):
        decided = [r for r in self.apps.list_applications(self.officer) if r["status"] in ("Approved", "Rejected")]
        self.assertTrue(decided, "seed should progress some applications to a recorded decision")
        d = self.apps.get_application(self.officer, decided[0]["id"])
        self.assertTrue(d["decision_reason"])
        self.assertIn("decision", [e["event"] for e in d["events"]])

    def test_overview_counts_are_consistent_with_records(self):
        ov = self.apps.overview(self.officer)
        self.assertEqual(sum(ov["by_status"].values()), ov["total"])
        self.assertEqual(sum(ov["bands"].values()), ov["total"])

    def test_demo_customers_can_sign_in_but_not_open_the_officer_portal(self):
        self.assertTrue(self.auth.login(demo.DEMO_CUSTOMERS[0], demo.DEMO_PASSWORD, "customer"))
        from src.loanwise.errors import AuthError
        with self.assertRaises(AuthError):
            self.auth.login(demo.DEMO_CUSTOMERS[0], demo.DEMO_PASSWORD, "officer")


if __name__ == "__main__":
    unittest.main()
