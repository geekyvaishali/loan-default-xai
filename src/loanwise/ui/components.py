"""Small shared UI helpers (kept free of business logic)."""
from __future__ import annotations

from datetime import datetime
from html import escape

import streamlit as st

from ..errors import LoanWiseError, ValidationError

STATUS_STYLE = {
    "Draft": ("#5B6472", "#EEF1F5"), "Submitted": ("#1D4ED8", "#E6EEFF"),
    "Under Review": ("#92400E", "#FEF3C7"), "Additional Information Required": ("#9A3412", "#FFEDD5"),
    "Approved": ("#0B6564", "#DDF3F1"), "Rejected": ("#991B1B", "#FDE8E8"),
}
BAND_STYLE = {"Low": ("#0B6564", "#DDF3F1"), "Elevated": ("#92400E", "#FEF3C7"), "High": ("#991B1B", "#FDE8E8")}
MODEL_DISCLAIMER = ("Experimental model assessment from a portfolio project. It estimates a statistical default-risk "
                    "score; it is not a loan offer, an eligibility decision, or financial advice.")


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def money(x) -> str:
    return "—" if x is None else f"${x:,.0f}"


def money2(x) -> str:
    return "—" if x is None else f"${x:,.2f}"


def stamp(t) -> str:
    return datetime.fromtimestamp(t).strftime("%d %b %Y, %H:%M") if t else "—"


def day(t) -> str:
    return datetime.fromtimestamp(t).strftime("%d %b %Y") if t else "—"


def chip(label: str, style_map=STATUS_STYLE) -> str:
    fg, bg = style_map.get(label, ("#5B6472", "#EEF1F5"))
    return f'<span class="lw-chip" style="color:{fg};background:{bg}">{escape(str(label))}</span>'


def table(data) -> None:
    st.dataframe(data, width="stretch", hide_index=True)


def chart(fig) -> None:
    st.plotly_chart(fig, width="stretch")


def set_flash(kind: str, message: str) -> None:
    st.session_state["flash"] = (kind, message)


def show_flash() -> None:
    flash = st.session_state.pop("flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])


def show_error(exc: Exception) -> None:
    """Friendly message for expected errors; unexpected ones are not swallowed."""
    if not isinstance(exc, LoanWiseError):
        raise exc
    st.error(str(exc))
    if isinstance(exc, ValidationError):
        for msg in exc.fields.values():
            if msg and msg not in str(exc):
                st.caption(f"• {msg}")


def stepper(labels: list[str], current: int) -> None:
    cells = "".join(
        f'<div class="{"now" if i == current else "done" if i < current else ""}">{i + 1}. {escape(l)}</div>'
        for i, l in enumerate(labels))
    html(f'<div class="lw-step">{cells}</div>')


def timeline(items: list[dict], by_key: str = "by") -> None:
    rows = ""
    for e in items:
        title = f'{escape(e["from"] or "—")} → {escape(e["to"])}' if e.get("to") else escape(e["event"].replace("_", " ").title())
        detail = f'<div class="lw-muted">{escape(e["detail"])}</div>' if e.get("detail") else ""
        rows += (f'<div class="item"><b>{title}</b> <span class="lw-muted">· {stamp(e["ts"])} · '
                 f'{escape(str(e.get(by_key, "")))}</span>{detail}</div>')
    html(f'<div class="lw-tl">{rows or "<span class=lw-muted>No activity yet.</span>"}</div>')


def note(text: str, warn: bool = False) -> None:
    html(f'<div class="{"lw-warn" if warn else "lw-note"}">{text}</div>')
