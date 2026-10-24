"""CLI: python -m src.loanwise.create_officer  - officers are created by an administrator, never self-registered."""
import getpass

from .auth import AuthService
from .db import Database


def main() -> None:
    email = input("Officer email: ").strip()
    name = input("Full name: ").strip()
    password = getpass.getpass("Password (min 10 chars, letters+numbers): ")
    if password != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords do not match.")
    AuthService(Database()).create_officer(email, name, password)
    print("Officer account created.")


if __name__ == "__main__":
    main()
