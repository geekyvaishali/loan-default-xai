"""SQLite persistence (stdlib only). All SQL is parametrised. Timestamps are epoch seconds (REAL).

Migration path to PostgreSQL: the schema is plain SQL; swapping this module for SQLAlchemy later keeps
the service layer unchanged.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from ..config import ROOT

DEFAULT_DB_PATH = Path(os.environ.get("LOANWISE_DB", ROOT / "data" / "loanwise.db"))
SCHEMA_VERSION = 1
STATUSES = ("Draft", "Submitted", "Under Review", "Additional Information Required", "Approved", "Rejected")

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('customer','officer')),
    pw_hash TEXT NOT NULL,
    is_demo INTEGER NOT NULL DEFAULT 0,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until REAL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    created_at REAL NOT NULL,
    last_seen REAL NOT NULL,
    expires_at REAL NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS customer_profiles (
    user_id INTEGER PRIMARY KEY REFERENCES users(id),
    data_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reference TEXT UNIQUE,
    customer_id INTEGER NOT NULL REFERENCES users(id),
    status TEXT NOT NULL CHECK (status IN {STATUSES!r}),
    data_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    submitted_at REAL,
    is_demo INTEGER NOT NULL DEFAULT 0,
    reviewer_id INTEGER REFERENCES users(id),
    info_request TEXT,
    decision TEXT,
    decision_reason TEXT,
    decided_at REAL,
    decided_by INTEGER REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_app_customer ON applications(customer_id);
CREATE INDEX IF NOT EXISTS idx_app_status ON applications(status);
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL REFERENCES applications(id),
    created_at REAL NOT NULL,
    model_version TEXT NOT NULL,
    probability REAL NOT NULL,
    threshold REAL NOT NULL,
    band TEXT NOT NULL,
    flagged INTEGER NOT NULL,
    warnings_json TEXT NOT NULL,
    factors_json TEXT NOT NULL,
    inputs_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assess_app ON assessments(application_id);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL REFERENCES applications(id),
    officer_id INTEGER NOT NULL REFERENCES users(id),
    ts REAL NOT NULL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS application_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL REFERENCES applications(id),
    ts REAL NOT NULL,
    actor_id INTEGER REFERENCES users(id),
    actor_role TEXT NOT NULL,
    event_type TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT,
    detail TEXT,
    customer_visible INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_events_app ON application_events(application_id);
"""


class Database:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init(self) -> None:
        with self.connect() as conn:
            conn.executescript(_SCHEMA)
            if conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
                conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
