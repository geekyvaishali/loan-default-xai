"""Password hashing and session tokens (standard library only)."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets

from .errors import ValidationError

_N, _R, _P = 2 ** 14, 8, 1
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def hash_password(password: str) -> str:
    """Salted scrypt hash, formatted ``scrypt$N$r$p$salt$hash``."""
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt),
                            n=int(n), r=int(r), p=int(p), dklen=len(bytes.fromhex(digest)))
        return hmac.compare_digest(dk, bytes.fromhex(digest))
    except (ValueError, TypeError):
        return False


DUMMY_HASH = hash_password("not-a-real-password")  # equalises timing for unknown emails


def validate_password_strength(password: str) -> None:
    problems = []
    if len(password) < 10:
        problems.append("at least 10 characters")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        problems.append("letters and numbers")
    if problems:
        raise ValidationError("Password needs " + " and ".join(problems) + ".", {"password": "too weak"})


def validate_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not _EMAIL.match(email) or len(email) > 254:
        raise ValidationError("Enter a valid email address.", {"email": "invalid"})
    return email


def new_token() -> str:
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    """Only this hash is stored, so a database leak does not reveal usable session tokens."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
