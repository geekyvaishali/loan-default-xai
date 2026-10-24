import json
import sqlite3
from contextlib import closing
import tempfile
import unittest
from pathlib import Path

from src.loanwise.applications import ApplicationService
from src.loanwise.auth import AuthService
from src.loanwise.db import Database
from src.loanwise.errors import AuthError, NotFound, PermissionDenied, ValidationError, WorkflowError
from src.loanwise.forms import DEFAULTS
from tests.helpers import PW, Env


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = Env()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    # ---- customer journey --------------------------------------------------
    def test_submit_stores_model_assessment_and_reference(self):
        e = self.env
        app_id = e.submit_for(e.alice)
        d = e.apps.my_application(e.alice, app_id)
        self.assertEqual(d["status"], "Submitted")
        self.assertRegex(d["reference"], r"^LW-\d{4}-\d{6}$")
        a = d["assessment"]
        self.assertTrue(0 <= a["probability"] <= 1)
        self.assertEqual(a["model_version"], e.predictor.version)
        self.assertEqual(a["threshold"], e.predictor.threshold)
        self.assertIn(a["band"], {"Low", "Elevated", "High"})
        self.assertTrue(a["factors"])
        self.assertIn("Application submitted", " ".join(t["detail"] or "" for t in d["timeline"]))

    def test_invalid_submission_rejected_and_nothing_saved(self):
        e = self.env
        before = len(e.apps.my_applications(e.bob))
        with self.assertRaises(ValidationError) as cm:
            e.apps.submit(e.bob, None, {**DEFAULTS, "age": 3, "credit_score": 5000})
        self.assertIn("age", cm.exception.fields)
        self.assertEqual(len(e.apps.my_applications(e.bob)), before)

    def test_customer_cannot_see_or_touch_another_customers_application(self):
        e = self.env
        app_id = e.submit_for(e.alice)
        with self.assertRaises(NotFound):
            e.apps.my_application(e.bob, app_id)
        with self.assertRaises(NotFound):
            e.apps.save_draft(e.bob, DEFAULTS, app_id)
        self.assertNotIn(app_id, [a["id"] for a in e.apps.my_applications(e.bob)])

    def test_customer_cannot_use_officer_functions(self):
        e = self.env
        app_id = e.submit_for(e.alice)
        calls = [lambda: e.apps.list_applications(e.alice), lambda: e.apps.get_application(e.alice, app_id),
                 lambda: e.apps.mark_under_review(e.alice, app_id),
                 lambda: e.apps.record_decision(e.alice, app_id, "Approved", "I approve myself please"),
                 lambda: e.apps.add_note(e.alice, app_id, "x"), lambda: e.apps.audit_events(e.alice),
                 lambda: e.apps.overview(e.alice), lambda: e.apps.assign(e.alice, app_id, 1)]
        for call in calls:
            with self.assertRaises(PermissionDenied):
                call()

    def test_officer_cannot_use_customer_functions_and_anonymous_is_rejected(self):
        e = self.env
        with self.assertRaises(PermissionDenied):
            e.apps.my_applications(e.officer)
        with self.assertRaises(AuthError):
            e.apps.list_applications(None)
        with self.assertRaises(AuthError):
            e.apps.list_applications("forged")

    # ---- officer workflow --------------------------------------------------
    def test_full_review_cycle_with_audit_trail(self):
        e = self.env
        app_id = e.submit_for(e.alice)
        with self.assertRaises(WorkflowError):           # cannot decide before review starts
            e.apps.record_decision(e.officer, app_id, "Approved", "Looks fine to me, approving.")
        e.apps.mark_under_review(e.officer, app_id)
        with self.assertRaises(WorkflowError):           # reason is mandatory
            e.apps.record_decision(e.officer, app_id, "Rejected", "short")
        e.apps.add_note(e.officer, app_id, "Checked income documents.")
        e.apps.record_decision(e.officer, app_id, "Rejected", "Income not verifiable from documents provided.")
        with self.assertRaises(WorkflowError):           # terminal state
            e.apps.record_decision(e.officer, app_id, "Approved", "Changing my mind after the fact.")
        d = e.apps.get_application(e.officer, app_id)
        self.assertEqual((d["status"], d["decision"]), ("Rejected", "Rejected"))
        transitions = [(ev["from"], ev["to"]) for ev in d["events"] if ev["to"]]
        self.assertEqual(transitions, [(None, "Draft"), ("Draft", "Submitted"), ("Submitted", "Under Review"),
                                       ("Under Review", "Rejected")])
        self.assertTrue(all(ev["ts"] for ev in d["events"]))
        self.assertIn("reviewer_assigned", [ev["event"] for ev in d["events"]])
        # customer sees the decision and reason, but not officer notes or officer names
        c = e.apps.my_application(e.alice, app_id)
        self.assertEqual(c["decision"], "Rejected")
        self.assertNotIn("Checked income", json.dumps(c))
        self.assertNotIn("Olivia", json.dumps(c))

    def test_request_info_then_customer_response_resubmits_and_reassesses(self):
        e = self.env
        app_id = e.submit_for(e.bob)
        e.apps.mark_under_review(e.officer, app_id)
        with self.assertRaises(ValidationError):
            e.apps.request_info(e.officer, app_id, "")
        e.apps.request_info(e.officer, app_id, "Please confirm your income figure.")
        self.assertEqual(e.apps.my_application(e.bob, app_id)["status"], "Additional Information Required")
        e.apps.respond_to_info_request(e.bob, app_id, {**DEFAULTS, "annual_income": 90_000.0}, "Updated income")
        d = e.apps.get_application(e.officer, app_id)
        self.assertEqual(d["status"], "Submitted")
        self.assertEqual(len(d["assessment_history"]), 2)
        self.assertEqual(d["data"]["annual_income"], 90_000.0)

    def test_illegal_transitions_rejected(self):
        e = self.env
        app_id = e.submit_for(e.alice)
        with self.assertRaises(WorkflowError):
            e.apps.respond_to_info_request(e.alice, app_id, DEFAULTS)   # nothing was requested
        e.apps.mark_under_review(e.officer, app_id)
        with self.assertRaises(WorkflowError):
            e.apps.mark_under_review(e.officer, app_id)

    def test_drafts_are_private_and_hidden_from_officers(self):
        e = self.env
        draft = e.apps.save_draft(e.alice, {"age": 30})
        self.assertNotIn(draft, [r["id"] for r in e.apps.list_applications(e.officer)])
        self.assertEqual(e.apps.my_application(e.alice, draft)["status"], "Draft")

    def test_filters_search_and_overview_use_real_records(self):
        e = Env()
        try:
            ids = [e.submit_for(e.alice, loan_purpose="Auto"), e.submit_for(e.bob, loan_purpose="Home")]
            e.apps.mark_under_review(e.officer, ids[0])
            all_rows = e.apps.list_applications(e.officer)
            self.assertEqual(len(all_rows), 2)
            self.assertEqual([r["loan_purpose"] for r in e.apps.list_applications(e.officer, purposes=["Home"])], ["Home"])
            self.assertEqual(len(e.apps.list_applications(e.officer, statuses=["Under Review"])), 1)
            ref = all_rows[0]["reference"]
            self.assertEqual(len(e.apps.list_applications(e.officer, search=ref.lower())), 1)
            self.assertEqual(len(e.apps.list_applications(e.officer, search="CUST-9999")), 0)
            ov = e.apps.overview(e.officer)
            self.assertEqual((ov["total"], ov["pending"], ov["decided"]), (2, 2, 0))
            self.assertEqual(sum(ov["bands"].values()), 2)
            self.assertFalse(ov["has_demo_records"])
        finally:
            e.close()

    def test_assign_and_audit_listing(self):
        e = self.env
        app_id = e.submit_for(e.alice)
        officers = e.apps.list_officers(e.officer)
        e.apps.assign(e.officer, app_id, officers[0]["id"])
        self.assertTrue(any(ev["event"] == "reviewer_assigned" and ev["application_id"] == app_id
                            for ev in e.apps.audit_events(e.officer)))
        with self.assertRaises(ValidationError):
            e.apps.assign(e.officer, app_id, 99999)

    def test_audit_log_never_contains_secrets(self):
        e = self.env
        e.submit_for(e.alice)
        blob = json.dumps(e.apps.audit_events(e.officer))
        self.assertNotIn(PW, blob)
        self.assertNotIn("scrypt$", blob)

    def test_profile_save_validates_and_roundtrips(self):
        e = self.env
        e.apps.save_profile(e.alice, {"age": 41, "annual_income": 70_000.0, "loan_amount": 999})  # step-3 key ignored
        self.assertEqual(e.apps.get_profile(e.alice), {"age": 41, "annual_income": 70_000.0})
        with self.assertRaises(ValidationError):
            e.apps.save_profile(e.alice, {"age": 5})
        self.assertEqual(e.apps.get_profile(e.bob), {})     # profiles are per user

    def test_records_survive_restart(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            path = Path(tmp.name) / "persist.db"
            db1 = Database(path)
            auth1 = AuthService(db1)
            auth1.register_customer("p@mail.test", "Persist Person", PW)
            tok = auth1.login("p@mail.test", PW, "customer")
            apps1 = ApplicationService(db1, auth1, self.env.predictor)
            app_id = apps1.submit(tok, None, DEFAULTS)["application_id"]
            del db1, auth1, apps1                                   # "restart": brand-new objects, same file
            db2 = Database(path)
            auth2 = AuthService(db2)
            apps2 = ApplicationService(db2, auth2, self.env.predictor)
            tok2 = auth2.login("p@mail.test", PW, "customer")
            rows = apps2.my_applications(tok2)
            self.assertEqual([r["id"] for r in rows], [app_id])
            self.assertEqual(apps2.my_application(tok2, app_id)["status"], "Submitted")
        finally:
            tmp.cleanup()

    def test_sql_injection_attempts_are_inert(self):
        e = self.env
        for evil in ["' OR '1'='1", "x'; DROP TABLE users; --"]:
            with self.assertRaises(AuthError):
                e.auth.login(evil, PW, "customer")
            e.apps.list_applications(e.officer, search=evil)
        with closing(sqlite3.connect(e.db.path)) as c:
            self.assertGreater(c.execute("SELECT COUNT(*) FROM users").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
