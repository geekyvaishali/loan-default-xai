"""LoanWise entry point.   Run from the project root:   streamlit run app/loanwise_app.py"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="LoanWise", page_icon=":material/account_balance:", layout="wide",
                   initial_sidebar_state="auto")

from src.loanwise.ui import customer, landing, officer, state  # noqa: E402


def main() -> None:
    try:
        svc = state.get_services()
    except FileNotFoundError as exc:
        st.error(f"{exc}")
        st.stop()
    user = state.current_user(svc)
    if user is None:
        portal = st.session_state.get("portal")
        if portal in ("customer", "officer"):
            landing.render_login(svc, portal)
        else:
            landing.render_landing()
    elif user.role == "customer":      # role comes from the server-validated session, never from UI state
        customer.render(svc, user)
    elif user.role == "officer":
        officer.render(svc, user)
    else:
        st.error("Unknown role.")


main()
