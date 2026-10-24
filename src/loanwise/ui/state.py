"""Service bootstrap and session handling. Authorisation is never decided from UI state:
the role always comes from a token validated by the service layer on every run."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from ...config import ARTIFACT_DIR
from ...predict import LoanRiskPredictor
from ..applications import ApplicationService
from ..auth import AuthService, User
from ..db import DEFAULT_DB_PATH, Database
from ..errors import AuthError
from . import components as ui


@dataclass
class Services:
    db: Database
    auth: AuthService
    apps: ApplicationService
    predictor: LoanRiskPredictor
    artifact_dir: Path


@st.cache_resource(show_spinner=False)
def get_services() -> Services:
    artifact_dir = Path(os.environ.get("LOANWISE_ARTIFACTS", ARTIFACT_DIR))
    db = Database(os.environ.get("LOANWISE_DB", DEFAULT_DB_PATH))
    auth = AuthService(db)
    predictor = LoanRiskPredictor.load(artifact_dir)
    return Services(db, auth, ApplicationService(db, auth, predictor), predictor, artifact_dir)


def current_user(svc: Services) -> User | None:
    token = st.session_state.get("token")
    if not token:
        return None
    try:
        return svc.auth.authenticate(token)
    except AuthError as exc:
        st.session_state.clear()
        ui.set_flash("warning", str(exc))
        return None


def sign_in(svc: Services, email: str, password: str, portal: str) -> None:
    token = svc.auth.login(email, password, portal)
    st.session_state.clear()
    st.session_state["token"] = token


def sign_out(svc: Services) -> None:
    svc.auth.logout(st.session_state.get("token"))
    st.session_state.clear()


def go(key: str, page: str) -> None:
    """Button callback: change page before the next render (widget state may not be set afterwards)."""
    st.session_state[key] = page
