"""Authentication and server-side sessions. Passwords/tokens: see security.py.

* Login is bound to a portal role; a customer's valid password never opens the officer portal, and
  every failure (unknown email, wrong password, wrong portal) returns the same generic message.
* The browser holds only an opaque token; its SHA-256 is what the database stores.
* Sessions expire after an idle period and after an absolute lifetime; logout and password change revoke.
* Every service call re-validates the token, so role checks never depend on UI state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from . import security
from .db import Database
from .errors import AuthError, PermissionDenied, ValidationError

ABSOLUTE_TIMEOUT_S = 8 * 3600
IDLE_TIMEOUT_S = 60 * 60
MAX_FAILED_ATTEMPTS = 5
LOCK_SECONDS = 15 * 60
GENERIC_LOGIN_ERROR = "Invalid email or password."
SESSION_ENDED = "Your session has ended. Please sign in again."


@dataclass(frozen=True)
class User:
    id: int
    email: str
    full_name: str
    role: str
    is_demo: bool = False

    @property
    def customer_ref(self) -> str:
        return f"CUST-{self.id:04d}"


class AuthService:
    def __init__(self, db: Database, clock: Callable[[], float] = time.time):
        self.db, self.clock = db, clock

    # ---------------------------------------------------------------- accounts
    def _create_user(self, email: str, full_name: str, password: str, role: str, is_demo: bool = False) -> int:
        email = security.validate_email(email)
        full_name = (full_name or "").strip()
        if not full_name:
            raise ValidationError("Full name is required.", {"full_name": "required"})
        security.validate_password_strength(password)
        with self.db.connect() as conn:
            if conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
                raise ValidationError("An account with this email already exists.", {"email": "already registered"})
            cur = conn.execute(
                "INSERT INTO users (email, full_name, role, pw_hash, is_demo, created_at) VALUES (?,?,?,?,?,?)",
                (email, full_name, role, security.hash_password(password), int(is_demo), self.clock()))
            return int(cur.lastrowid)

    def register_customer(self, email: str, full_name: str, password: str) -> int:
        """Public self-registration. Always a customer; officers are created by an administrator only."""
        return self._create_user(email, full_name, password, "customer")

    def create_officer(self, email: str, full_name: str, password: str, is_demo: bool = False) -> int:
        return self._create_user(email, full_name, password, "officer", is_demo)

    # ---------------------------------------------------------------- sessions
    def _open_session(self, conn, user_id: int) -> str:
        token, now = security.new_token(), self.clock()
        conn.execute("INSERT INTO sessions (token_hash, user_id, created_at, last_seen, expires_at) VALUES (?,?,?,?,?)",
                     (security.token_hash(token), user_id, now, now, now + ABSOLUTE_TIMEOUT_S))
        return token

    def login(self, email: str, password: str, portal: str) -> str:
        """Verify credentials for one portal and return a session token."""
        email = (email or "").strip().lower()
        now = self.clock()
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            if row is None:
                security.verify_password(password or "", security.DUMMY_HASH)   # keep timing similar
                raise AuthError(GENERIC_LOGIN_ERROR)
            if row["locked_until"] and row["locked_until"] > now:
                raise AuthError("Too many failed sign-in attempts. Please try again later.")
            if security.verify_password(password or "", row["pw_hash"]) and row["role"] == portal:
                conn.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = ?", (row["id"],))
                return self._open_session(conn, row["id"])
            failed = row["failed_attempts"] + 1
            if failed >= MAX_FAILED_ATTEMPTS:
                conn.execute("UPDATE users SET failed_attempts = 0, locked_until = ? WHERE id = ?",
                             (now + LOCK_SECONDS, row["id"]))
            else:
                conn.execute("UPDATE users SET failed_attempts = ? WHERE id = ?", (failed, row["id"]))
        raise AuthError(GENERIC_LOGIN_ERROR)

    def authenticate(self, token: str | None) -> User:
        """Resolve a token to a user or raise AuthError (missing, forged, revoked, idle or expired)."""
        if not token:
            raise AuthError(SESSION_ENDED)
        now = self.clock()
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT s.token_hash, s.last_seen, s.expires_at, s.revoked, u.id, u.email, u.full_name, u.role, u.is_demo "
                "FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ?",
                (security.token_hash(token),)).fetchone()
            if row is None or row["revoked"]:
                raise AuthError(SESSION_ENDED)
            if now >= row["expires_at"] or now - row["last_seen"] > IDLE_TIMEOUT_S:
                conn.execute("UPDATE sessions SET revoked = 1 WHERE token_hash = ?", (row["token_hash"],))
                raise AuthError(SESSION_ENDED)
            conn.execute("UPDATE sessions SET last_seen = ? WHERE token_hash = ?", (now, row["token_hash"]))
            return User(row["id"], row["email"], row["full_name"], row["role"], bool(row["is_demo"]))

    def logout(self, token: str | None) -> None:
        if token:
            with self.db.connect() as conn:
                conn.execute("UPDATE sessions SET revoked = 1 WHERE token_hash = ?", (security.token_hash(token),))

    def require(self, token_or_user, role: str) -> User:
        """Authenticate (if given a token) and enforce the role. Raises AuthError / PermissionDenied."""
        user = token_or_user if isinstance(token_or_user, User) else self.authenticate(token_or_user)
        if user.role != role:
            raise PermissionDenied(f"This action requires the {role} role.")
        return user

    def change_password(self, token: str, old_password: str, new_password: str) -> str:
        """Change the password, revoke every existing session, and return a fresh token."""
        user = self.authenticate(token)
        with self.db.connect() as conn:
            row = conn.execute("SELECT pw_hash FROM users WHERE id = ?", (user.id,)).fetchone()
            if not security.verify_password(old_password or "", row["pw_hash"]):
                raise AuthError("Your current password is incorrect.")
            security.validate_password_strength(new_password)
            conn.execute("UPDATE users SET pw_hash = ?, failed_attempts = 0, locked_until = NULL WHERE id = ?",
                         (security.hash_password(new_password), user.id))
            conn.execute("UPDATE sessions SET revoked = 1 WHERE user_id = ?", (user.id,))
            return self._open_session(conn, user.id)
