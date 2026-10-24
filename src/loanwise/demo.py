"""Synthetic demo data. Run:  python -m src.loanwise.seed_demo

Creates demo accounts and ~40 clearly flagged SYNTHETIC applications through the real service layer
(so audit events, assessments and statuses are genuine records of the demo workflow). Demo credentials
below are for local demonstration only and are never used unless this script is run.
"""
from __future__ import annotations

import random

from ..synthetic import make_synthetic_applicants
from .applications import ApplicationService
from .auth import AuthService

DEMO_PASSWORD = "DemoPass-2026"
DEMO_OFFICER = "officer.demo@loanwise.demo"
DEMO_CUSTOMERS = ["customer.demo@loanwise.demo", "asha.demo@loanwise.demo", "ravi.demo@loanwise.demo"]
DAY = 86400.0


class _Clock:
    def __init__(self, t: float):
        self.t = t

    def __call__(self) -> float:
        return self.t


def seed(db, predictor, n_applications: int = 40, seed_value: int = 7, now: float | None = None) -> dict:
    import time
    now = now or time.time()
    clock = _Clock(now - 60 * DAY)
    auth, apps = AuthService(db, clock), None
    apps = ApplicationService(db, auth, predictor, clock)
    rng = random.Random(seed_value)
    auth.create_officer(DEMO_OFFICER, "Demo Officer", DEMO_PASSWORD, is_demo=True)
    for i, email in enumerate(DEMO_CUSTOMERS):
        uid = auth._create_user(email, ["Demo Customer", "Asha Demo", "Ravi Demo"][i], DEMO_PASSWORD, "customer", True)

    frame = make_synthetic_applicants(n_applications, seed=seed_value)
    created = 0
    for i, rec in frame.iterrows():
        clock.t = now - 60 * DAY + (i / max(1, n_applications)) * 58 * DAY
        email = DEMO_CUSTOMERS[i % len(DEMO_CUSTOMERS)]
        ctoken = auth.login(email, DEMO_PASSWORD, "customer")
        income = float(rec["Income"])
        data = {"age": int(rec["Age"]), "education": rec["Education"], "marital_status": rec["MaritalStatus"],
                "employment_type": rec["EmploymentType"], "months_employed": int(rec["MonthsEmployed"]),
                "has_dependents": rec["HasDependents"], "annual_income": income,
                "monthly_obligations": round(income / 12 * rng.uniform(0.02, 0.25), 2),
                "credit_score": int(rec["CreditScore"]), "num_credit_lines": int(rec["NumCreditLines"]),
                "has_mortgage": rec["HasMortgage"], "loan_amount": float(min(rec["LoanAmount"], income * 3)),
                "loan_purpose": rec["LoanPurpose"], "interest_rate": float(rec["InterestRate"]),
                "loan_term_months": int(rec["LoanTerm"]), "has_cosigner": rec["HasCoSigner"]}
        apps.submit(ctoken, None, data)
        created += 1
        app_id = created
        # progress about 60% of applications through the genuine officer workflow
        roll = rng.random()
        if roll < 0.6:
            clock.t += rng.uniform(0.2, 2) * DAY
            otoken = auth.login(DEMO_OFFICER, DEMO_PASSWORD, "officer")
            apps.mark_under_review(otoken, app_id)
            if roll < 0.15:
                apps.request_info(otoken, app_id, "Please confirm your current employment details.")
            elif roll < 0.45:
                clock.t += rng.uniform(0.2, 3) * DAY
                otoken = auth.login(DEMO_OFFICER, DEMO_PASSWORD, "officer")   # earlier session idled out on the simulated clock
                apps.add_note(otoken, app_id, "Demo review note: figures checked against the submitted inputs.")
                decision = "Approved" if rng.random() < 0.6 else "Rejected"
                apps.record_decision(otoken, app_id, decision, f"Demo decision recorded after manual review ({decision.lower()}).")
    return {"applications": created, "officer": DEMO_OFFICER, "customers": DEMO_CUSTOMERS, "password": DEMO_PASSWORD}


def ensure_seeded(db, predictor) -> bool:
    """Seed the synthetic demo data only if the database has no users yet. Returns True if it seeded.

    Used by hosted demos whose disk is wiped on restart, so the app is never empty on first visit.
    """
    with db.connect() as conn:
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0:
            return False
    seed(db, predictor)
    return True
