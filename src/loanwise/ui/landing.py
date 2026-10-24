"""Public landing page and the two portal sign-in screens."""
from __future__ import annotations

import os

import streamlit as st

from ..errors import LoanWiseError
from . import components as ui
from . import state, theme
from ..demo import DEMO_CUSTOMERS, DEMO_OFFICER, DEMO_PASSWORD


def _choose(portal: str) -> None:
    st.session_state["portal"] = portal


def render_landing() -> None:
    theme.inject("landing")
    ui.show_flash()
    ui.html('<div class="lw-land-brand">Loan<span>Wise</span></div>'
            '<div class="lw-land-h">Smarter Loan Decisions.<br>Greater Financial Clarity.</div>'
            '<div class="lw-land-sub">An end-to-end loan eligibility and credit-risk workflow — '
            'a portfolio project, not a bank.</div>')
    left, right = st.columns(2, gap="large")
    with left:
        ui.html('<div class="lw-portal cust"><div class="tag">For applicants</div>'
                '<h3>Check Your Loan Eligibility</h3>'
                '<p>Explore your estimated loan eligibility, calculate monthly payments, and understand the factors '
                'that may influence a lender\'s decision.</p></div>')
        st.button("Continue as Customer", key="go_customer", on_click=_choose, args=("customer",), width="stretch")
    with right:
        ui.html('<div class="lw-portal off"><div class="tag">For bank staff</div>'
                '<h3>Loan Risk &amp; Application Management</h3>'
                '<p>Review applications, analyze model-predicted risk, examine explanations, and manage the lending '
                'review workflow.</p></div>')
        st.button("Continue as Bank Officer", key="go_officer", on_click=_choose, args=("officer",), width="stretch")
    st.write("")
    ui.note("<b>Portfolio demonstration.</b> LoanWise is not a bank and cannot approve or fund loans. "
            "Risk scores come from an experimental model trained on a dataset that appears to be synthetic; "
            "they must not be used for real credit decisions.")


def render_login(svc: state.Services, portal: str) -> None:
    theme.inject("landing")
    ui.show_flash()
    label = "Customer" if portal == "customer" else "Bank Officer"
    st.button("← Back", key="back_landing", on_click=_choose, args=(None,))
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        with st.container(border=True):
            st.subheader(f"{label} sign in")
            if portal == "customer":
                tab_in, tab_new = st.tabs(["Sign in", "Create account"])
            else:
                tab_in, tab_new = st.container(), None
                st.caption("Officer accounts are issued by an administrator and cannot be created here.")
            with tab_in:
                with st.form(f"login_{portal}"):
                    email = st.text_input("Email")
                    password = st.text_input("Password", type="password")
                    submitted = st.form_submit_button("Sign in", type="primary")
                if submitted:
                    try:
                        state.sign_in(svc, email, password, portal)
                    except LoanWiseError as exc:
                        ui.show_error(exc)
                    else:
                        st.rerun()
            if tab_new is not None:
                with tab_new:
                    with st.form("register"):
                        name = st.text_input("Full name")
                        email2 = st.text_input("Email address")
                        pw1 = st.text_input("Password (10+ characters, letters and numbers)", type="password")
                        pw2 = st.text_input("Repeat password", type="password")
                        created = st.form_submit_button("Create account", type="primary")
                    if created:
                        try:
                            if pw1 != pw2:
                                st.error("The two passwords do not match.")
                            else:
                                svc.auth.register_customer(email2, name, pw1)
                                ui.set_flash("success", "Account created. Please sign in.")
                                st.rerun()
                        except LoanWiseError as exc:
                            ui.show_error(exc)
            if os.environ.get("LOANWISE_DEMO_MODE") == "1":
                who = DEMO_CUSTOMERS[0] if portal == "customer" else DEMO_OFFICER
                ui.note(f"<b>Demo mode.</b> Synthetic demo account (local use only): <code>{who}</code> / "
                        f"<code>{DEMO_PASSWORD}</code>")
