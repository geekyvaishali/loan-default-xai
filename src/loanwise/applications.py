"""Application workflow: persistence, status state machine, role/ownership enforcement, audit trail.

Every public method takes the caller's session ``token`` and authorises it first.
Customers only ever see their own applications (other IDs raise ``NotFound``).
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Callable

from ..predict import LoanRiskPredictor
from . import assess, forms
from .auth import AuthService, User
from .db import Database
from .errors import NotFound, PermissionDenied, ValidationError, WorkflowError

STATUSES = ["Draft", "Submitted", "Under Review", "Additional Information Required", "Approved", "Rejected"]
TRANSITIONS = {
    "Draft": {"Submitted"},
    "Submitted": {"Under Review", "Additional Information Required"},
    "Under Review": {"Additional Information Required", "Approved", "Rejected"},
    "Additional Information Required": {"Submitted"},
    "Approved": set(), "Rejected": set(),
}
MIN_REASON_CHARS = 10


class ApplicationService:
    def __init__(self, db: Database, auth: AuthService, predictor: LoanRiskPredictor,
                 clock: Callable[[], float] = time.time):
        self.db, self.auth, self.predictor, self.clock = db, auth, predictor, clock

    # ------------------------------------------------------------------ helpers
    def _event(self, conn, app_id, actor: User | None, etype, frm=None, to=None, detail=None, visible=True):
        conn.execute(
            "INSERT INTO application_events (application_id, ts, actor_id, actor_role, event_type, from_status,"
            " to_status, detail, customer_visible) VALUES (?,?,?,?,?,?,?,?,?)",
            (app_id, self.clock(), actor.id if actor else None, actor.role if actor else "system", etype, frm, to,
             detail, int(visible)))

    def _transition(self, conn, row, new_status: str, actor: User, detail: str | None = None, etype="status_change"):
        if new_status not in TRANSITIONS[row["status"]]:
            raise WorkflowError(f"An application cannot move from '{row['status']}' to '{new_status}'.")
        conn.execute("UPDATE applications SET status = ?, updated_at = ? WHERE id = ?",
                     (new_status, self.clock(), row["id"]))
        self._event(conn, row["id"], actor, etype, row["status"], new_status, detail)

    def _store_assessment(self, conn, app_id: int, result: dict) -> None:
        conn.execute(
            "INSERT INTO assessments (application_id, created_at, model_version, probability, threshold, band,"
            " flagged, warnings_json, factors_json, inputs_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (app_id, self.clock(), result["model_version"], result["probability"], result["threshold"],
             result["band"], int(result["flagged"]), json.dumps(result["warnings"]),
             json.dumps({"method": result["explanation_method"], "factors": result["factors"]}),
             json.dumps(result["inputs"])))

    @staticmethod
    def _latest_assessment(conn, app_id: int) -> dict | None:
        r = conn.execute("SELECT * FROM assessments WHERE application_id = ? ORDER BY id DESC LIMIT 1",
                         (app_id,)).fetchone()
        return _assessment_dict(r) if r else None

    def _own_application(self, conn, user: User, app_id: int):
        row = conn.execute("SELECT * FROM applications WHERE id = ? AND customer_id = ?",
                           (app_id, user.id)).fetchone()
        if row is None:
            raise NotFound("Application not found.")
        return row

    @staticmethod
    def _any_application(conn, app_id: int):
        row = conn.execute("SELECT * FROM applications WHERE id = ?", (app_id,)).fetchone()
        if row is None:
            raise NotFound("Application not found.")
        return row

    # ------------------------------------------------------------ customer side
    def save_draft(self, token: str, data: dict, app_id: int | None = None) -> int:
        user = self.auth.require(token, "customer")
        now = self.clock()
        with self.db.connect() as conn:
            if app_id is None:
                cur = conn.execute(
                    "INSERT INTO applications (customer_id, status, data_json, created_at, updated_at, is_demo)"
                    " VALUES (?,?,?,?,?,?)", (user.id, "Draft", json.dumps(data), now, now, int(user.is_demo)))
                new_id = int(cur.lastrowid)
                conn.execute("UPDATE applications SET reference = ? WHERE id = ?", (_reference(new_id, now), new_id))
                self._event(conn, new_id, user, "created", None, "Draft", "Draft saved")
                return new_id
            row = self._own_application(conn, user, app_id)
            if row["status"] != "Draft":
                raise WorkflowError("Only drafts can be edited.")
            conn.execute("UPDATE applications SET data_json = ?, updated_at = ? WHERE id = ?",
                         (json.dumps(data), now, app_id))
            return app_id

    def submit(self, token: str, app_id: int | None, data: dict) -> dict:
        """Validate everything, run the model, and move Draft -> Submitted. Returns the assessment."""
        user = self.auth.require(token, "customer")
        clean = forms.validate_all(data)
        result = assess.run_assessment(self.predictor, clean)
        derived = forms.derived_figures(clean)
        if app_id is None:
            app_id = self.save_draft(token, clean)
        now = self.clock()
        with self.db.connect() as conn:
            row = self._own_application(conn, user, app_id)
            conn.execute("UPDATE applications SET data_json = ?, submitted_at = ? WHERE id = ?",
                         (json.dumps({**clean, "_derived": derived}), now, app_id))
            self._store_assessment(conn, app_id, result)
            self._transition(conn, row, "Submitted", user, "Application submitted for review")
        return {"application_id": app_id, **result, "derived": derived}

    def respond_to_info_request(self, token: str, app_id: int, data: dict, message: str = "") -> dict:
        user = self.auth.require(token, "customer")
        clean = forms.validate_all(data)
        result = assess.run_assessment(self.predictor, clean)
        derived = forms.derived_figures(clean)
        with self.db.connect() as conn:
            row = self._own_application(conn, user, app_id)
            if row["status"] != "Additional Information Required":
                raise WorkflowError("No additional information has been requested for this application.")
            conn.execute("UPDATE applications SET data_json = ?, info_request = NULL WHERE id = ?",
                         (json.dumps({**clean, "_derived": derived}), app_id))
            self._store_assessment(conn, app_id, result)
            self._transition(conn, row, "Submitted", user, (message or "Customer updated the application")[:500])
        return {"application_id": app_id, **result, "derived": derived}

    def my_applications(self, token: str) -> list[dict]:
        user = self.auth.require(token, "customer")
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM applications WHERE customer_id = ? ORDER BY id DESC",
                                (user.id,)).fetchall()
            out = []
            for r in rows:
                data = json.loads(r["data_json"])
                out.append({"id": r["id"], "reference": r["reference"], "status": r["status"],
                            "loan_amount": data.get("loan_amount"), "loan_purpose": data.get("loan_purpose"),
                            "submitted_at": r["submitted_at"], "updated_at": r["updated_at"]})
            return out

    def my_application(self, token: str, app_id: int) -> dict:
        user = self.auth.require(token, "customer")
        with self.db.connect() as conn:
            row = self._own_application(conn, user, app_id)
            events = conn.execute(
                "SELECT * FROM application_events WHERE application_id = ? AND customer_visible = 1 ORDER BY id",
                (app_id,)).fetchall()
            return {
                "id": row["id"], "reference": row["reference"], "status": row["status"],
                "data": json.loads(row["data_json"]), "created_at": row["created_at"],
                "submitted_at": row["submitted_at"], "info_request": row["info_request"],
                "decision": row["decision"], "decision_reason": row["decision_reason"],
                "decided_at": row["decided_at"], "assessment": self._latest_assessment(conn, app_id),
                "timeline": [{"ts": e["ts"], "event": e["event_type"], "from": e["from_status"],
                              "to": e["to_status"], "detail": e["detail"],
                              "by": "You" if e["actor_role"] == "customer" else
                                    ("Bank review team" if e["actor_role"] == "officer" else "System")}
                             for e in events],
            }

    def get_profile(self, token: str) -> dict:
        user = self.auth.require(token, "customer")
        with self.db.connect() as conn:
            r = conn.execute("SELECT data_json FROM customer_profiles WHERE user_id = ?", (user.id,)).fetchone()
            return json.loads(r["data_json"]) if r else {}

    def save_profile(self, token: str, data: dict) -> None:
        user = self.auth.require(token, "customer")
        allowed = {k: v for k, v in data.items() if k in forms.FIELDS and forms.FIELDS[k][0] in (1, 2)}
        errors = {}
        for field in allowed:
            try:
                forms._coerce(field, allowed[field])
            except ValueError as exc:
                errors[field] = str(exc)
        if errors:
            raise ValidationError("Please correct the highlighted fields.", errors)
        with self.db.connect() as conn:
            conn.execute("INSERT INTO customer_profiles (user_id, data_json) VALUES (?, ?) "
                         "ON CONFLICT(user_id) DO UPDATE SET data_json = excluded.data_json",
                         (user.id, json.dumps(allowed)))

    # ------------------------------------------------------------- officer side
    def list_officers(self, token: str) -> list[dict]:
        self.auth.require(token, "officer")
        with self.db.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT id, full_name FROM users WHERE role = 'officer' ORDER BY full_name")]

    def list_applications(self, token: str, statuses=None, bands=None, purposes=None,
                          date_from: float | None = None, date_to: float | None = None,
                          search: str = "") -> list[dict]:
        self.auth.require(token, "officer")
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT a.*, u.full_name AS reviewer_name FROM applications a "
                "LEFT JOIN users u ON u.id = a.reviewer_id WHERE a.status != 'Draft' ORDER BY a.id DESC").fetchall()
            out = []
            for r in rows:
                data = json.loads(r["data_json"])
                latest = self._latest_assessment(conn, r["id"])
                item = {"id": r["id"], "reference": r["reference"], "customer_ref": f"CUST-{r['customer_id']:04d}",
                        "loan_amount": data.get("loan_amount"), "loan_purpose": data.get("loan_purpose"),
                        "submitted_at": r["submitted_at"], "status": r["status"],
                        "probability": latest["probability"] if latest else None,
                        "band": latest["band"] if latest else None,
                        "reviewer": r["reviewer_name"], "is_demo": bool(r["is_demo"])}
                if statuses and item["status"] not in statuses:
                    continue
                if bands and item["band"] not in bands:
                    continue
                if purposes and item["loan_purpose"] not in purposes:
                    continue
                if date_from and (item["submitted_at"] or 0) < date_from:
                    continue
                if date_to and (item["submitted_at"] or 0) > date_to:
                    continue
                q = (search or "").strip().lower()
                if q and q not in item["reference"].lower() and q not in item["customer_ref"].lower() \
                        and q != str(item["id"]):
                    continue
                out.append(item)
            return out

    def get_application(self, token: str, app_id: int) -> dict:
        self.auth.require(token, "officer")
        with self.db.connect() as conn:
            row = self._any_application(conn, app_id)
            cust = conn.execute("SELECT full_name FROM users WHERE id = ?", (row["customer_id"],)).fetchone()
            events = conn.execute(
                "SELECT e.*, u.full_name AS actor_name FROM application_events e "
                "LEFT JOIN users u ON u.id = e.actor_id WHERE e.application_id = ? ORDER BY e.id", (app_id,)).fetchall()
            notes = conn.execute(
                "SELECT n.*, u.full_name AS officer_name FROM notes n JOIN users u ON u.id = n.officer_id "
                "WHERE n.application_id = ? ORDER BY n.id", (app_id,)).fetchall()
            assessments = [_assessment_dict(r) for r in conn.execute(
                "SELECT * FROM assessments WHERE application_id = ? ORDER BY id", (app_id,))]
            reviewer = conn.execute("SELECT full_name FROM users WHERE id = ?", (row["reviewer_id"],)).fetchone() \
                if row["reviewer_id"] else None
            return {
                "id": row["id"], "reference": row["reference"], "customer_ref": f"CUST-{row['customer_id']:04d}",
                "customer_name": cust["full_name"], "status": row["status"], "data": json.loads(row["data_json"]),
                "submitted_at": row["submitted_at"], "reviewer_id": row["reviewer_id"],
                "reviewer": reviewer["full_name"] if reviewer else None, "info_request": row["info_request"],
                "decision": row["decision"], "decision_reason": row["decision_reason"], "decided_at": row["decided_at"],
                "is_demo": bool(row["is_demo"]),
                "assessment": assessments[-1] if assessments else None, "assessment_history": assessments,
                "notes": [{"ts": n["ts"], "officer": n["officer_name"], "text": n["text"]} for n in notes],
                "events": [{"ts": e["ts"], "actor": e["actor_name"] or e["actor_role"], "role": e["actor_role"],
                            "event": e["event_type"], "from": e["from_status"], "to": e["to_status"],
                            "detail": e["detail"]} for e in events],
            }

    def mark_under_review(self, token: str, app_id: int) -> None:
        user = self.auth.require(token, "officer")
        with self.db.connect() as conn:
            row = self._any_application(conn, app_id)
            self._transition(conn, row, "Under Review", user, "Review started")
            if row["reviewer_id"] is None:
                conn.execute("UPDATE applications SET reviewer_id = ? WHERE id = ?", (user.id, app_id))
                self._event(conn, app_id, user, "reviewer_assigned", detail=f"Assigned to {user.full_name}", visible=False)

    def assign(self, token: str, app_id: int, officer_id: int) -> None:
        user = self.auth.require(token, "officer")
        with self.db.connect() as conn:
            row = self._any_application(conn, app_id)
            target = conn.execute("SELECT id, full_name FROM users WHERE id = ? AND role = 'officer'",
                                  (officer_id,)).fetchone()
            if target is None:
                raise ValidationError("Choose a valid officer.")
            if row["status"] in ("Approved", "Rejected", "Draft"):
                raise WorkflowError("This application can no longer be reassigned.")
            conn.execute("UPDATE applications SET reviewer_id = ?, updated_at = ? WHERE id = ?",
                         (target["id"], self.clock(), app_id))
            self._event(conn, app_id, user, "reviewer_assigned", detail=f"Assigned to {target['full_name']}", visible=False)

    def request_info(self, token: str, app_id: int, message: str) -> None:
        user = self.auth.require(token, "officer")
        message = (message or "").strip()
        if len(message) < 5:
            raise ValidationError("Describe what information is needed (at least 5 characters).")
        with self.db.connect() as conn:
            row = self._any_application(conn, app_id)
            self._transition(conn, row, "Additional Information Required", user, message[:500])
            conn.execute("UPDATE applications SET info_request = ? WHERE id = ?", (message[:1000], app_id))

    def record_decision(self, token: str, app_id: int, decision: str, reason: str) -> None:
        """A human officer records the final decision. A reason is mandatory; the model never decides."""
        user = self.auth.require(token, "officer")
        if decision not in ("Approved", "Rejected"):
            raise ValidationError("Decision must be 'Approved' or 'Rejected'.")
        reason = (reason or "").strip()
        if len(reason) < MIN_REASON_CHARS:
            raise WorkflowError(f"A reason of at least {MIN_REASON_CHARS} characters is required.")
        with self.db.connect() as conn:
            row = self._any_application(conn, app_id)
            if row["status"] != "Under Review":
                raise WorkflowError("Mark the application 'Under Review' before recording a decision.")
            self._transition(conn, row, decision, user, reason[:1000], etype="decision")
            conn.execute("UPDATE applications SET decision = ?, decision_reason = ?, decided_at = ?, decided_by = ? "
                         "WHERE id = ?", (decision, reason[:1000], self.clock(), user.id, app_id))

    def add_note(self, token: str, app_id: int, text: str) -> None:
        user = self.auth.require(token, "officer")
        text = (text or "").strip()
        if not text:
            raise ValidationError("Note cannot be empty.")
        with self.db.connect() as conn:
            self._any_application(conn, app_id)
            conn.execute("INSERT INTO notes (application_id, officer_id, ts, text) VALUES (?,?,?,?)",
                         (app_id, user.id, self.clock(), text[:2000]))
            self._event(conn, app_id, user, "note_added", detail="Reviewer note added", visible=False)

    def audit_events(self, token: str, limit: int = 500) -> list[dict]:
        self.auth.require(token, "officer")
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT e.*, a.reference, u.full_name AS actor_name FROM application_events e "
                "JOIN applications a ON a.id = e.application_id LEFT JOIN users u ON u.id = e.actor_id "
                "ORDER BY e.id DESC LIMIT ?", (limit,)).fetchall()
            return [{"ts": e["ts"], "reference": e["reference"], "application_id": e["application_id"],
                     "actor": e["actor_name"] or e["actor_role"], "role": e["actor_role"], "event": e["event_type"],
                     "from": e["from_status"], "to": e["to_status"], "detail": e["detail"]} for e in rows]

    def overview(self, token: str) -> dict:
        """All figures are computed from stored records; nothing is simulated."""
        self.auth.require(token, "officer")
        rows = self.list_applications(token)
        by_status = {s: 0 for s in STATUSES if s != "Draft"}
        for r in rows:
            by_status[r["status"]] += 1
        trend: dict[str, int] = {}
        for r in rows:
            if r["submitted_at"]:
                day = datetime.fromtimestamp(r["submitted_at"]).strftime("%Y-%m-%d")
                trend[day] = trend.get(day, 0) + 1
        return {
            "total": len(rows), "by_status": by_status,
            "pending": by_status["Submitted"] + by_status["Under Review"] + by_status["Additional Information Required"],
            "decided": by_status["Approved"] + by_status["Rejected"],
            "model_high_risk": sum(1 for r in rows if r["band"] == "High"),
            "bands": {b: sum(1 for r in rows if r["band"] == b) for b in ("Low", "Elevated", "High")},
            "probabilities": [r["probability"] for r in rows if r["probability"] is not None],
            "trend": dict(sorted(trend.items())),
            "amount_total": float(sum(r["loan_amount"] or 0 for r in rows)),
            "has_demo_records": any(r["is_demo"] for r in rows),
        }


def _reference(app_id: int, now: float) -> str:
    return f"LW-{datetime.fromtimestamp(now).year}-{app_id:06d}"


def _assessment_dict(r) -> dict:
    return {"id": r["id"], "created_at": r["created_at"], "model_version": r["model_version"],
            "probability": r["probability"], "threshold": r["threshold"], "band": r["band"],
            "flagged": bool(r["flagged"]), "warnings": json.loads(r["warnings_json"]),
            "factors": json.loads(r["factors_json"])["factors"],
            "explanation_method": json.loads(r["factors_json"])["method"],
            "inputs": json.loads(r["inputs_json"])}
