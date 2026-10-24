"""Expected, user-facing errors. Anything else is a bug and should surface as a real exception."""
from __future__ import annotations


class LoanWiseError(Exception):
    """Base class: the message is safe to show to the user."""


class AuthError(LoanWiseError):
    """Bad credentials, locked account, or a missing/expired session."""


class PermissionDenied(LoanWiseError):
    """The signed-in user's role does not allow this action."""


class NotFound(LoanWiseError):
    """Record missing OR not visible to this user (deliberately indistinguishable)."""


class WorkflowError(LoanWiseError):
    """The action is not allowed in the application's current state."""


class ValidationError(LoanWiseError):
    """Invalid input. ``fields`` maps field name -> message for inline display."""

    def __init__(self, message: str, fields: dict[str, str] | None = None):
        super().__init__(message)
        self.fields = dict(fields or {})
