import sqlite3
from contextlib import closing
import unittest

from src.loanwise import security
from src.loanwise.auth import (ABSOLUTE_TIMEOUT_S, IDLE_TIMEOUT_S, LOCK_SECONDS, MAX_FAILED_ATTEMPTS)
from src.loanwise.errors import AuthError, PermissionDenied, ValidationError
from tests.helpers import PW, Env


class Clock:
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t


class SecurityTests(unittest.TestCase):
    def test_hash_is_salted_and_verifiable(self):
        a, b = security.hash_password(PW), security.hash_password(PW)
        self.assertNotEqual(a, b)
        self.assertTrue(security.verify_password(PW, a))
        self.assertFalse(security.verify_password("wrong", a))
        self.assertNotIn(PW, a)

    def test_garbage_hash_does_not_crash(self):
        self.assertFalse(security.verify_password(PW, "not-a-hash"))

    def test_password_and_email_validation(self):
        for bad in ["short1", "onlyletterslong", "1234567890123"]:
            with self.assertRaises(ValidationError):
                security.validate_password_strength(bad)
        self.assertEqual(security.validate_email("  A@B.co "), "a@b.co")
        with self.assertRaises(ValidationError):
            security.validate_email("nope")


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.env = Env(self.clock)

    def tearDown(self):
        self.env.close()

    def test_wrong_password_and_wrong_portal_share_one_generic_error(self):
        msgs = set()
        for email, pw, portal in [("alice@mail.test", "bad", "customer"), ("alice@mail.test", PW, "officer"),
                                  ("nobody@mail.test", PW, "customer")]:
            with self.assertRaises(AuthError) as cm:
                self.env.auth.login(email, pw, portal)
            msgs.add(str(cm.exception))
        self.assertEqual(len(msgs), 1)

    def test_lockout_after_repeated_failures_then_recovery(self):
        for _ in range(MAX_FAILED_ATTEMPTS):
            with self.assertRaises(AuthError):
                self.env.auth.login("bob@mail.test", "bad", "customer")
        with self.assertRaises(AuthError) as cm:   # even the right password is refused while locked
            self.env.auth.login("bob@mail.test", PW, "customer")
        self.assertIn("Too many", str(cm.exception))
        self.clock.t += LOCK_SECONDS + 1
        self.assertTrue(self.env.auth.login("bob@mail.test", PW, "customer"))

    def test_session_idle_and_absolute_expiry(self):
        t = self.env.auth.login("alice@mail.test", PW, "customer")
        self.clock.t += IDLE_TIMEOUT_S - 5
        self.env.auth.authenticate(t)                       # activity refreshes idle timer
        self.clock.t += IDLE_TIMEOUT_S + 5
        with self.assertRaises(AuthError):
            self.env.auth.authenticate(t)
        t2 = self.env.auth.login("alice@mail.test", PW, "customer")
        for _ in range(int(ABSOLUTE_TIMEOUT_S // (IDLE_TIMEOUT_S - 60)) + 1):
            self.clock.t += IDLE_TIMEOUT_S - 60
            try:
                self.env.auth.authenticate(t2)
            except AuthError:
                break
        else:
            self.fail("absolute timeout never applied")

    def test_logout_revokes_token(self):
        t = self.env.auth.login("alice@mail.test", PW, "customer")
        self.env.auth.logout(t)
        with self.assertRaises(AuthError):
            self.env.auth.authenticate(t)

    def test_tokens_stored_hashed_not_plain(self):
        t = self.env.auth.login("alice@mail.test", PW, "customer")
        with closing(sqlite3.connect(self.env.db.path)) as c:
            stored = [r[0] for r in c.execute("SELECT token_hash FROM sessions")]
            pw_hashes = [r[0] for r in c.execute("SELECT pw_hash FROM users")]
        self.assertNotIn(t, stored)
        self.assertTrue(all(h.startswith("scrypt$") for h in pw_hashes))

    def test_unknown_or_missing_token_rejected(self):
        for bad in [None, "", "forged-token"]:
            with self.assertRaises(AuthError):
                self.env.auth.authenticate(bad)

    def test_role_enforcement(self):
        with self.assertRaises(PermissionDenied):
            self.env.auth.require(self.env.alice, "officer")
        with self.assertRaises(PermissionDenied):
            self.env.auth.require(self.env.officer, "customer")

    def test_self_registration_only_creates_customers_and_rejects_duplicates(self):
        self.env.auth.register_customer("new@mail.test", "New Person", PW)
        with self.assertRaises(ValidationError):
            self.env.auth.register_customer("NEW@mail.test", "Other Person", PW)
        with self.assertRaises(AuthError):                  # a customer account cannot open the officer portal
            self.env.auth.login("new@mail.test", PW, "officer")

    def test_change_password_revokes_old_sessions(self):
        old = self.env.alice
        new = self.env.auth.change_password(old, PW, "An0ther-Passw0rd")
        with self.assertRaises(AuthError):
            self.env.auth.authenticate(old)
        self.env.auth.authenticate(new)
        with self.assertRaises(AuthError):
            self.env.auth.login("alice@mail.test", PW, "customer")
        with self.assertRaises(AuthError):
            self.env.auth.change_password(new, "wrong-old-password1", "Yet-An0ther-Pass")


if __name__ == "__main__":
    unittest.main()
