"""Customer portal: light, guided, plain-language."""
from __future__ import annotations

import json
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import finance, forms, reports
from ..errors import LoanWiseError
from . import components as ui
from . import state, theme

PAGES = ["Dashboard", "Check eligibility", "EMI simulator", "My applications", "Profile"]
NAV_KEY = "cust_page"
HINT = {
    "age": "Model training data covers ages 18-69.", "months_employed": "Months with your current employer.",
    "annual_income": "Gross yearly income.", "monthly_obligations": "Loan, card and other debt payments per month.",
    "credit_score": "Self-reported. This demo does not verify it.", "num_credit_lines": "Training data covers 1-4.",
    "interest_rate": "An assumption you control. It is not a lender's offer.",
    "loan_term_months": "The model was trained on 12-60 month loans, so longer tenures are not assessed.",
}


def render(svc: state.Services, user) -> None:
    theme.inject("customer")
    token = st.session_state["token"]
    st.session_state.setdefault(NAV_KEY, PAGES[0])
    top = st.columns([1.1, 4.4, 1.0])
    with top[0]:
        ui.html('<div class="lw-logo">Loan<span>Wise</span></div>')
    with top[1]:
        st.radio("Navigate", PAGES, key=NAV_KEY, horizontal=True, label_visibility="collapsed")
    with top[2]:
        if st.button("Sign out", key="signout_c"):
            state.sign_out(svc)
            st.rerun()
    st.divider()
    ui.show_flash()
    page = st.session_state[NAV_KEY]
    {"Dashboard": dashboard, "Check eligibility": eligibility, "EMI simulator": emi_simulator,
     "My applications": my_applications, "Profile": profile}[page](svc, token, user)


# --------------------------------------------------------------------------- dashboard
def dashboard(svc, token, user) -> None:
    apps = svc.apps.my_applications(token)
    profile_data = svc.apps.get_profile(token)
    st.markdown(f"## Welcome, {user.full_name.split()[0]}")
    st.caption("Here is where your applications stand and what you can do next.")
    submitted = [a for a in apps if a["status"] != "Draft"]
    latest = svc.apps.my_application(token, submitted[0]["id"]) if submitted else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications", len(submitted))
    c2.metric("Latest status", latest["status"] if latest else "None yet")
    if latest:
        d = latest["data"]
        c3.metric("Estimated EMI", ui.money2(d.get("_derived", {}).get("emi")))
        a = latest["assessment"]
        c4.metric("Model default estimate", f"{a['probability']:.1%}", help="Experimental model score, not a decision.")
    else:
        c3.metric("Estimated EMI", "—")
        c4.metric("Model default estimate", "—")

    left, right = st.columns([1.6, 1])
    with left:
        with st.container(border=True):
            st.markdown("#### Your applications")
            if apps:
                ui.table(pd.DataFrame([{"Reference": a["reference"], "Status": a["status"],
                                        "Amount": ui.money(a["loan_amount"]), "Purpose": a["loan_purpose"],
                                        "Submitted": ui.day(a["submitted_at"])} for a in apps[:6]]))
                st.button("View all applications", on_click=state.go, args=(NAV_KEY, "My applications"))
            else:
                st.info("You have no applications yet. Start with the eligibility check.")
    with right:
        with st.container(border=True):
            st.markdown("#### Quick actions")
            st.button("Check eligibility", type="primary", on_click=state.go, args=(NAV_KEY, "Check eligibility"),
                      key="qa_elig", width="stretch")
            st.button("Open EMI simulator", on_click=state.go, args=(NAV_KEY, "EMI simulator"),
                      key="qa_emi", width="stretch")
        source = (latest or {}).get("data") or profile_data
        if source.get("annual_income"):
            with st.container(border=True):
                st.markdown("#### Illustrative payment capacity")
                rate, months, cap = 10.0, 36, 0.40
                room = finance.payment_capacity(source["annual_income"], source.get("monthly_obligations", 0), cap)
                st.metric("Up to (illustration)", ui.money(finance.max_principal(room, rate, months)))
                st.caption(f"Simple arithmetic: {int(cap * 100)}% of monthly income minus existing payments, at an "
                           f"assumed {rate}% over {months} months. The {int(cap * 100)}% cap is an assumption, not "
                           "lender policy. Adjust it in the EMI simulator.")


# --------------------------------------------------------------------------- eligibility flow
def _form_state(svc, token) -> dict:
    if "elig" not in st.session_state:
        base = {**forms.DEFAULTS, **svc.apps.get_profile(token)}
        st.session_state["elig"] = {"step": 0, "form": base, "app_id": None, "result_id": None}
    return st.session_state["elig"]


def _step_fields(step: int, form: dict, prefix: str = "f") -> None:
    k = lambda f: f"{prefix}_{f}"
    h = lambda f: HINT.get(f)
    if step == 0:
        a, b = st.columns(2)
        form["age"] = a.number_input("Age", 18, 100, int(form["age"]), 1, key=k("age"), help=h("age"))
        form["months_employed"] = b.number_input("Months in current employment", 0, 720, int(form["months_employed"]), 1,
                                                 key=k("months"), help=h("months_employed"))
        a, b = st.columns(2)
        form["education"] = a.selectbox("Education", forms.FIELDS["education"][3],
                                        forms.FIELDS["education"][3].index(form["education"]), key=k("edu"))
        form["employment_type"] = b.selectbox("Employment type", forms.FIELDS["employment_type"][3],
                                              forms.FIELDS["employment_type"][3].index(form["employment_type"]), key=k("emp"))
        a, b = st.columns(2)
        form["marital_status"] = a.selectbox("Marital status", forms.FIELDS["marital_status"][3],
                                             forms.FIELDS["marital_status"][3].index(form["marital_status"]), key=k("mar"),
                                             help="Used by the demo model because it was in the training data.")
        form["has_dependents"] = b.radio("Financial dependents", ["No", "Yes"], index=["No", "Yes"].index(form["has_dependents"]),
                                         horizontal=True, key=k("dep"))
    elif step == 1:
        a, b = st.columns(2)
        form["annual_income"] = a.number_input("Annual income ($)", 1000.0, 10_000_000.0, float(form["annual_income"]),
                                               1000.0, format="%.0f", key=k("inc"), help=h("annual_income"))
        form["monthly_obligations"] = b.number_input("Existing monthly debt payments ($)", 0.0, 1_000_000.0,
                                                     float(form["monthly_obligations"]), 50.0, format="%.0f",
                                                     key=k("obl"), help=h("monthly_obligations"))
        form["credit_score"] = st.slider("Credit score (self-reported)", 300, 850, int(form["credit_score"]),
                                         key=k("cs"), help=h("credit_score"))
        a, b = st.columns(2)
        form["num_credit_lines"] = a.number_input("Open credit lines", 0, 50, int(form["num_credit_lines"]), 1,
                                                  key=k("cl"), help=h("num_credit_lines"))
        form["has_mortgage"] = b.radio("Has a mortgage", ["No", "Yes"], index=["No", "Yes"].index(form["has_mortgage"]),
                                       horizontal=True, key=k("mort"))
    elif step == 2:
        a, b = st.columns(2)
        form["loan_amount"] = a.number_input("Loan amount ($)", 1000.0, 10_000_000.0, float(form["loan_amount"]),
                                             500.0, format="%.0f", key=k("amt"))
        form["loan_purpose"] = b.selectbox("Loan purpose", forms.LOAN_PURPOSES,
                                           forms.LOAN_PURPOSES.index(form["loan_purpose"]), key=k("pur"))
        a, b = st.columns(2)
        form["interest_rate"] = a.slider("Assumed annual interest rate (%)", 1.0, 40.0, float(form["interest_rate"]),
                                         0.1, key=k("rate"), help=h("interest_rate"))
        form["loan_term_months"] = b.selectbox("Tenure (months)", forms.TENURE_OPTIONS,
                                               forms.TENURE_OPTIONS.index(form["loan_term_months"]), key=k("term"),
                                               help=h("loan_term_months"))
        form["has_cosigner"] = st.radio("Has a co-signer", ["No", "Yes"], index=["No", "Yes"].index(form["has_cosigner"]),
                                        horizontal=True, key=k("cos"))
        pay = finance.emi(form["loan_amount"], form["interest_rate"], form["loan_term_months"])
        st.caption(f"Estimated EMI at the assumed rate: **{ui.money2(pay)}** per month.")


def eligibility(svc, token, user) -> None:
    elig = _form_state(svc, token)
    if elig["result_id"]:
        return _results(svc, token, elig["result_id"], elig)
    step, form = elig["step"], elig["form"]
    st.markdown("## Loan eligibility check")
    st.caption("Four short steps. We only ask for what the demonstration model needs: no passwords, bank "
               "credentials, or government ID numbers.")
    ui.stepper(forms.STEP_TITLES, step)
    with st.container(border=True):
        st.markdown(f"#### {forms.STEP_TITLES[step]}")
        if step < 3:
            _step_fields(step, form)
        else:
            _review(svc, form)
    errors = st.session_state.pop("elig_errors", {})
    for msg in errors.values():
        st.error(msg)

    nav = st.columns([1, 1, 1, 3])
    if step > 0 and nav[0].button("Back", key="elig_back"):
        elig["step"] -= 1
        st.rerun()
    if step < 3:
        if nav[1].button("Next", type="primary", key="elig_next"):
            problems = forms.validate_step(step + 1, form)
            if problems:
                st.session_state["elig_errors"] = problems
            else:
                elig["step"] += 1
            st.rerun()
    else:
        if nav[1].button("Save draft", key="elig_draft"):
            try:
                elig["app_id"] = svc.apps.save_draft(token, form, elig["app_id"])
                ui.set_flash("success", "Draft saved. Find it under My applications.")
            except LoanWiseError as exc:
                ui.set_flash("error", str(exc))
            st.rerun()
        agreed = st.session_state.get("elig_agree", False)
        if nav[2].button("Submit", type="primary", key="elig_submit", disabled=not agreed):
            try:
                with st.spinner("Running the assessment…"):
                    out = svc.apps.submit(token, elig["app_id"], form)
                elig["result_id"] = out["application_id"]
                ui.set_flash("success", "Application submitted and assessed.")
            except LoanWiseError as exc:
                ui.set_flash("error", str(exc))
            st.rerun()


def _review(svc, form: dict) -> None:
    labels = {f: forms.FIELDS[f][1] for f in forms.FIELDS}
    rows = [{"Item": labels[f], "Your entry": form[f]} for f in forms.FIELDS]
    ui.table(pd.DataFrame(rows).astype(str))
    try:
        clean = forms.validate_all(form)
    except LoanWiseError as exc:
        ui.show_error(exc)
        return
    derived = forms.derived_figures(clean)
    a, b = st.columns(2)
    a.metric("Estimated EMI", ui.money2(derived["emi"]))
    b.metric("Debt-to-income (derived)", f"{derived['dti']:.2f}",
             help="(existing monthly payments + estimated EMI) / monthly income")
    report = svc.predictor.validate(pd.DataFrame([forms.model_record(clean)]))
    for msgs in report.row_warnings.values():
        ui.note("<b>Outside the data the model learned from:</b> " + "; ".join(escape(m) for m in msgs) +
                ". The estimate will be less reliable.", warn=True)
    ui.note(ui.MODEL_DISCLAIMER)
    st.checkbox("I confirm this information is accurate to the best of my knowledge, and I understand the result "
                "is a model assessment, not a lender's decision.", key="elig_agree")


def _factor_chart(factors: list[dict]) -> go.Figure:
    fs = list(reversed(factors))
    fig = go.Figure(go.Bar(
        x=[f["contribution"] for f in fs], y=[f["label"] for f in fs], orientation="h",
        marker_color=[theme.DANGER if f["contribution"] > 0 else theme.TEAL for f in fs],
        hovertemplate="%{y}: %{x:+.2f} log-odds<extra></extra>"))
    fig.update_layout(height=60 + 34 * len(fs), margin=dict(l=10, r=10, t=10, b=30), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", xaxis_title="← lowers model score | raises model score →",
                      font={"color": theme.INK})
    return fig


def _results(svc, token, app_id: int, elig: dict | None = None) -> None:
    d = svc.apps.my_application(token, app_id)
    a, data = d["assessment"], d["data"]
    st.markdown(f"## Your assessment · {d['reference']}")
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        with st.container(border=True):
            ui.html(f'<div class="lw-muted">Estimated default probability</div><div class="lw-big">{a["probability"]:.1%}</div>')
            above = a["probability"] >= a["threshold"]
            st.markdown(f"This is **{'above' if above else 'below'}** the review threshold of {a['threshold']:.1%} "
                        "used by this experimental model.")
            st.caption(f"For comparison, about {svc.predictor.metadata['data']['default_rate']:.1%} of applicants in "
                       f"the model's training data defaulted. Model version {a['model_version']}.")
            ui.html(f'Application status: {ui.chip(d["status"])}')
        ui.note("<b>Model output ≠ lender decision.</b> This score estimates default risk. It does not approve or "
                "reject anything, and a real lender would apply its own policies and verification.")
        for w in a["warnings"]:
            ui.note(escape(w), warn=True)
    with right:
        with st.container(border=True):
            st.markdown("#### What influenced the score")
            ui.chart(_factor_chart(a["factors"]))
            ups = [f["label"] for f in a["factors"] if f["contribution"] > 0][:3]
            downs = [f["label"] for f in a["factors"] if f["contribution"] < 0][:3]
            st.markdown(f"**Raising the score most:** {', '.join(ups) or 'none'}  \n"
                        f"**Lowering the score most:** {', '.join(downs) or 'none'}")
            st.caption(f"Method: {a['explanation_method']}. These describe how the model behaves; they do not show "
                       "that changing an input would change a real-world outcome.")

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("#### Assumptions used")
        derived = data.get("_derived", {})
        st.markdown(f"- Interest rate **{data['interest_rate']}%** (your assumption, not an offer)\n"
                    f"- Tenure **{data['loan_term_months']} months**, amount **{ui.money(data['loan_amount'])}**\n"
                    f"- Estimated EMI **{ui.money2(derived.get('emi'))}**\n"
                    f"- Debt-to-income **{derived.get('dti', 0):.2f}** (computed from your figures)\n"
                    "- Income and credit score are self-reported and unverified")
    with c2, st.container(border=True):
        st.markdown("#### General readiness suggestions")
        tips = []
        if derived.get("dti", 0) > 0.40:
            tips.append("Your payments would take a large share of income; a smaller amount or lower existing "
                        "payments reduces that burden.")
        if data["loan_term_months"] <= 24:
            tips.append("A longer tenure lowers the monthly payment, though total interest rises.")
        if data["credit_score"] < 670:
            tips.append("Paying on time and keeping card balances low tends to improve credit scores over time.")
        tips.append("Compare offers from several lenders and read the full cost of credit.")
        st.markdown("\n".join(f"- {t}" for t in tips))
        st.caption("General guidance, not generated by the model.")
    st.download_button("Download assessment report (HTML)", reports.customer_report_html(d),
                       file_name=f"{d['reference']}_assessment.html", mime="text/html")
    if elig is not None:
        b1, b2 = st.columns([1, 1])
        if b1.button("Start another check", key="elig_again"):
            st.session_state.pop("elig", None)
            st.session_state.pop("elig_agree", None)
            st.rerun()
        b2.button("Track this application", type="primary", key="elig_track", on_click=state.go,
                  args=(NAV_KEY, "My applications"))


# --------------------------------------------------------------------------- EMI simulator
def emi_simulator(svc, token, user) -> None:
    st.markdown("## EMI & loan simulator")
    ui.note("The interest rate is <b>your assumption</b>, not a lender's offer. Real offers depend on the lender, "
            "fees and your credit profile.")
    c1, c2, c3 = st.columns(3)
    principal = c1.number_input("Loan principal ($)", 1000.0, 5_000_000.0, 25_000.0, 1000.0, format="%.0f", key="sim_p")
    rate = c2.slider("Annual interest rate (%)", 0.0, 40.0, 10.0, 0.1, key="sim_r")
    months = c3.slider("Tenure (months)", 6, 360, 48, 6, key="sim_n")
    s = finance.summary(principal, rate, months)
    m1, m2, m3 = st.columns(3)
    m1.metric("Monthly EMI", ui.money2(s["emi"]))
    m2.metric("Total repayment", ui.money2(s["total_repayment"]))
    m3.metric("Total interest", ui.money2(s["total_interest"]))
    sched = pd.DataFrame(finance.amortization_schedule(principal, rate, months))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sched["month"], y=sched["balance"], name="Remaining balance",
                             line=dict(color=theme.NAVY, width=3)))
    fig.add_trace(go.Bar(x=sched["month"], y=sched["interest"], name="Interest paid that month",
                         marker_color=theme.TEAL, opacity=0.6))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Month", legend=dict(orientation="h"))
    ui.chart(fig)
    with st.expander("Amortization schedule"):
        shown = sched.rename(columns={"month": "Month", "payment": "Payment", "interest": "Interest",
                                      "principal": "Principal", "balance": "Balance"}).round(2)
        ui.table(shown)
        st.download_button("Download schedule (CSV)", shown.to_csv(index=False).encode(),
                           file_name="amortization_schedule.csv", mime="text/csv")
    with st.expander("How much could a payment support? (illustration)"):
        a, b, c = st.columns(3)
        income = a.number_input("Annual income ($)", 0.0, 10_000_000.0, 60_000.0, 1000.0, format="%.0f", key="cap_i")
        debts = b.number_input("Existing monthly payments ($)", 0.0, 1_000_000.0, 500.0, 50.0, format="%.0f", key="cap_d")
        cap = c.slider("Payment-to-income cap (assumption)", 0.10, 0.60, 0.40, 0.05, key="cap_c")
        room = finance.payment_capacity(income, debts, cap)
        st.metric("Monthly room for a new payment", ui.money2(room))
        st.metric("Loan this could service at the rate and tenure above", ui.money(finance.max_principal(room, rate, months)))
        st.caption("The cap is a rule of thumb you can change. It is not lender policy or a validated eligibility rule.")


# --------------------------------------------------------------------------- tracking
def my_applications(svc, token, user) -> None:
    st.markdown("## My applications")
    apps = svc.apps.my_applications(token)
    if not apps:
        st.info("No applications yet.")
        st.button("Check eligibility", type="primary", on_click=state.go, args=(NAV_KEY, "Check eligibility"))
        return
    ui.table(pd.DataFrame([{"Reference": a["reference"], "Status": a["status"], "Amount": ui.money(a["loan_amount"]),
                            "Purpose": a["loan_purpose"], "Submitted": ui.day(a["submitted_at"]),
                            "Last update": ui.stamp(a["updated_at"])} for a in apps]))
    refs = {a["reference"]: a["id"] for a in apps}
    choice = st.selectbox("Open an application", list(refs), key="track_sel")
    d = svc.apps.my_application(token, refs[choice])
    left, right = st.columns([1.3, 1], gap="large")
    with left, st.container(border=True):
        ui.html(f'<h4>{escape(d["reference"])} &nbsp; {ui.chip(d["status"])}</h4>')
        st.markdown("##### Status timeline")
        ui.timeline(d["timeline"])
    with right, st.container(border=True):
        if d["decision"]:
            st.markdown(f"**Decision recorded: {d['decision']}** on {ui.day(d['decided_at'])}")
            st.write(d["decision_reason"])
            st.caption("Recorded by a human reviewer. The model score alone does not decide applications.")
        elif d["status"] == "Additional Information Required":
            st.warning(f"The review team asked: {d['info_request']}")
        elif d["status"] == "Draft":
            st.markdown("This draft has not been submitted.")
        else:
            st.markdown("Your application is with the review team. No decision has been recorded yet.")
        if d["assessment"]:
            st.metric("Model default estimate", f"{d['assessment']['probability']:.1%}")
    if d["assessment"]:
        with st.expander("View assessment details"):
            _results(svc, token, d["id"])
    if d["status"] == "Draft":
        if st.button("Continue this draft", key="draft_continue"):
            base = {**forms.DEFAULTS, **d["data"]}
            st.session_state["elig"] = {"step": 0, "form": {k: base[k] for k in forms.FIELDS},
                                        "app_id": d["id"], "result_id": None}
            st.session_state[NAV_KEY] = "Check eligibility"
            st.rerun()
    if d["status"] == "Additional Information Required":
        st.markdown("##### Update your information")
        form = {**forms.DEFAULTS, **{k: v for k, v in d["data"].items() if k in forms.FIELDS}}
        for i in range(3):
            with st.expander(forms.STEP_TITLES[i], expanded=(i == 0)):
                _step_fields(i, form, prefix=f"upd{d['id']}")
        msg = st.text_input("Message to the review team (optional)", key="upd_msg")
        if st.button("Resubmit", type="primary", key="upd_submit"):
            try:
                svc.apps.respond_to_info_request(token, d["id"], form, msg)
                ui.set_flash("success", "Thank you. Your updated application was resubmitted.")
            except LoanWiseError as exc:
                ui.set_flash("error", str(exc))
            st.rerun()


# --------------------------------------------------------------------------- profile
def profile(svc, token, user) -> None:
    st.markdown("## Profile")
    data = {**forms.DEFAULTS, **svc.apps.get_profile(token)}
    with st.container(border=True):
        st.markdown("#### Account")
        st.write(f"**{user.full_name}** · {user.email} · reference {user.customer_ref}")
    with st.container(border=True):
        st.markdown("#### Details used to pre-fill the eligibility check")
        st.caption("Stored privately for you. Bank-facing analytics show only a reference number, never this page.")
        form = dict(data)
        _step_fields(0, form, prefix="prof")
        _step_fields(1, form, prefix="prof")
        if st.button("Save profile", type="primary", key="prof_save"):
            try:
                svc.apps.save_profile(token, form)
                ui.set_flash("success", "Profile saved.")
            except LoanWiseError as exc:
                ui.set_flash("error", str(exc))
            st.rerun()
    with st.container(border=True):
        st.markdown("#### Change password")
        with st.form("pw_change"):
            old = st.text_input("Current password", type="password")
            new = st.text_input("New password (10+ characters, letters and numbers)", type="password")
            go_ = st.form_submit_button("Update password", type="primary")
        if go_:
            try:
                st.session_state["token"] = svc.auth.change_password(token, old, new)
                ui.set_flash("success", "Password updated. Other sessions were signed out.")
            except LoanWiseError as exc:
                ui.set_flash("error", str(exc))
            st.rerun()
