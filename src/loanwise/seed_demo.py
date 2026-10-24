"""CLI: python -m src.loanwise.seed_demo  (creates demo accounts + synthetic applications)."""
from ..predict import LoanRiskPredictor
from .db import Database
from .demo import seed


def main() -> None:
    db = Database()
    with db.connect() as conn:
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]:
            raise SystemExit("Database already has users; delete data/loanwise.db first to re-seed.")
    out = seed(db, LoanRiskPredictor.load())
    print(f"Seeded {out['applications']} SYNTHETIC applications.")
    print(f"Officer login : {out['officer']} / {out['password']}")
    print(f"Customer login: {out['customers'][0]} / {out['password']}  (demo credentials, local use only)")


if __name__ == "__main__":
    main()
