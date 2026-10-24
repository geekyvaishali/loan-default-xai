"""Shared test fixtures. The model used here is trained on SYNTHETIC data only (never the real dataset)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.config import TrainConfig
from src.loanwise.applications import ApplicationService
from src.loanwise.auth import AuthService
from src.loanwise.db import Database
from src.loanwise.forms import DEFAULTS
from src.predict import LoanRiskPredictor
from src.synthetic import make_synthetic_applicants
from src.train import run_training

PW = "Test-Passw0rd-2026"
_CACHE: dict = {}


def predictor_and_dir():
    """Train a small model once per test run and return (LoanRiskPredictor, artifact_dir)."""
    if "pred" not in _CACHE:
        root = Path(tempfile.mkdtemp(prefix="lw_model_"))
        data = make_synthetic_applicants(6000, seed=7, with_target=True)
        data.insert(0, "LoanID", [f"A{i}" for i in range(len(data))])
        data.to_csv(root / "train.csv", index=False)
        run_training(TrainConfig(models=("logistic_regression",), cv_folds=2), root / "train.csv", root / "art")
        _CACHE["pred"] = (LoanRiskPredictor.load(root / "art"), root / "art")
    return _CACHE["pred"]


class Env:
    """A throwaway database with one officer and two customers. alice/bob/officer are session tokens."""

    def __init__(self, clock=None):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self._tmp.name) / "test.db")
        self.predictor, _ = predictor_and_dir()
        self.auth = AuthService(self.db, clock) if clock else AuthService(self.db)
        self.apps = (ApplicationService(self.db, self.auth, self.predictor, clock) if clock
                     else ApplicationService(self.db, self.auth, self.predictor))
        self.auth.create_officer("officer@mail.test", "Olivia Officer", PW)
        self.auth.register_customer("alice@mail.test", "Alice Customer", PW)
        self.auth.register_customer("bob@mail.test", "Bob Customer", PW)
        self.officer = self.auth.login("officer@mail.test", PW, "officer")
        self.alice = self.auth.login("alice@mail.test", PW, "customer")
        self.bob = self.auth.login("bob@mail.test", PW, "customer")

    def submit_for(self, token: str, **overrides) -> int:
        return self.apps.submit(token, None, {**DEFAULTS, **overrides})["application_id"]

    def close(self) -> None:
        self._tmp.cleanup()
