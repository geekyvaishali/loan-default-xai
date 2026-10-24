"""Visual design: two distinct skins (customer = light/airy, officer = dark sidebar/dense)."""
from __future__ import annotations

import streamlit as st

NAVY, NAVY_DEEP, TEAL, TEAL_DARK = "#0B2545", "#071A33", "#0E7C7B", "#0A5F5E"
INK, MUTED, LINE, PAPER, MIST = "#14213D", "#5B6472", "#DDE3EA", "#FFFFFF", "#F4F7FA"
DANGER, WARN = "#B42318", "#B54708"

_FONTS = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');"

_BASE = f"""
{_FONTS}
html, body, [class*="css"], .stApp {{ font-family: 'Inter','Segoe UI',Roboto,Arial,sans-serif; color:{INK}; }}
h1,h2,h3,h4 {{ font-family:'Manrope','Inter','Segoe UI',sans-serif; color:{NAVY}; letter-spacing:-0.02em; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display:none !important; }}
header[data-testid="stHeader"] {{ background:transparent; height:0; }}
.block-container {{ padding-top:1.2rem; padding-bottom:3rem; max-width:1240px; }}
.lw-chip {{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:600; }}
.lw-note {{ border-left:4px solid {TEAL}; background:#EEF7F7; padding:12px 16px; border-radius:6px; font-size:.92rem; }}
.lw-warn {{ border-left:4px solid {WARN}; background:#FFF6E8; padding:12px 16px; border-radius:6px; font-size:.92rem; }}
.lw-muted {{ color:{MUTED}; font-size:.88rem; }}
.lw-step {{ display:flex; gap:10px; margin:6px 0 22px; flex-wrap:wrap; }}
.lw-step div {{ flex:1; min-width:120px; padding:10px 12px; border-top:4px solid {LINE}; font-size:.82rem; color:{MUTED}; }}
.lw-step div.done {{ border-color:{TEAL}; color:{TEAL_DARK}; }}
.lw-step div.now {{ border-color:{NAVY}; color:{NAVY}; font-weight:700; }}
.lw-tl {{ border-left:2px solid {LINE}; margin-left:6px; padding-left:16px; }}
.lw-tl .item {{ position:relative; margin-bottom:14px; }}
.lw-tl .item:before {{ content:''; position:absolute; left:-23px; top:5px; width:10px; height:10px; border-radius:50%; background:{TEAL}; }}
"""

_CUSTOMER = f"""
.stApp {{ background:{MIST}; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {{ display:none !important; }}
.lw-topbar {{ display:flex; align-items:center; gap:14px; padding:6px 0 14px; border-bottom:1px solid {LINE}; margin-bottom:18px; }}
.lw-logo {{ font-family:'Manrope',sans-serif; font-weight:800; font-size:1.35rem; color:{NAVY}; }}
.lw-logo span {{ color:{TEAL}; }}
[data-testid="stRadio"] [role="radiogroup"] {{ gap:6px; flex-wrap:wrap; }}
[data-testid="stRadio"] label {{ background:{PAPER}; border:1px solid {LINE}; border-radius:999px; padding:6px 16px; cursor:pointer; }}
[data-testid="stRadio"] label:has(input:checked) {{ background:{NAVY}; border-color:{NAVY}; }}
[data-testid="stRadio"] label:has(input:checked) p {{ color:#fff !important; }}
[data-testid="stRadio"] label > div:first-child {{ display:none; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:{PAPER}; border-radius:16px; border:1px solid {LINE}; box-shadow:0 2px 10px rgba(11,37,69,.05); }}
div[data-testid="stMetric"] {{ background:{PAPER}; border:1px solid {LINE}; border-radius:14px; padding:16px 18px; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] {{ background:{TEAL}; border:0; border-radius:10px; font-weight:600; padding:.6rem 1.2rem; }}
.stButton > button[kind="primary"]:hover {{ background:{TEAL_DARK}; }}
.stButton > button {{ border-radius:10px; }}
.lw-hero {{ font-family:'Manrope',sans-serif; font-weight:800; font-size:2.1rem; color:{NAVY}; line-height:1.15; }}
.lw-big {{ font-family:'Manrope',sans-serif; font-weight:800; font-size:3rem; color:{NAVY}; line-height:1; }}
"""

_OFFICER = f"""
.stApp {{ background:#EEF1F5; }}
.block-container {{ padding-top:.8rem; max-width:1500px; font-size:.93rem; }}
[data-testid="stSidebar"] {{ background:{NAVY_DEEP}; min-width:250px; }}
[data-testid="stSidebar"] * {{ color:#D7E2F0 !important; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{ padding:7px 12px; border-radius:4px; width:100%; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{ background:#12335F; border-left:3px solid #2BB3B1; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {{ display:none; }}
[data-testid="stSidebar"] .stButton > button {{ background:transparent; border:1px solid #2C4A73; border-radius:4px; }}
.lw-bar {{ display:flex; justify-content:space-between; align-items:center; background:{NAVY}; color:#fff; padding:10px 16px; border-radius:4px; margin-bottom:14px; }}
.lw-bar b {{ font-family:'Manrope',sans-serif; letter-spacing:.02em; }}
div[data-testid="stMetric"] {{ background:{PAPER}; border:1px solid {LINE}; border-top:3px solid {TEAL}; border-radius:2px; padding:10px 14px; }}
div[data-testid="stMetricValue"] {{ font-size:1.6rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:{PAPER}; border-radius:3px; border:1px solid {LINE}; }}
.stButton > button[kind="primary"] {{ background:{NAVY}; border:0; border-radius:3px; }}
.stButton > button {{ border-radius:3px; }}
.lw-brand {{ font-family:'Manrope',sans-serif; font-weight:800; font-size:1.25rem; color:#fff !important; }}
"""

_LANDING = f"""
.stApp {{ background:linear-gradient(180deg,{NAVY_DEEP} 0,{NAVY} 340px,{MIST} 340px); }}
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {{ display:none !important; }}
.lw-land-brand {{ font-family:'Manrope',sans-serif; font-weight:800; color:#fff; font-size:1.4rem; }}
.lw-land-brand span {{ color:#2BB3B1; }}
.lw-land-h {{ font-family:'Manrope',sans-serif; font-weight:800; color:#fff; font-size:2.6rem; line-height:1.1; margin:34px 0 8px; }}
.lw-land-sub {{ color:#B8C7DB; font-size:1.1rem; margin-bottom:44px; }}
.lw-portal {{ border-radius:18px; padding:28px; min-height:230px; border:1px solid {LINE}; }}
.lw-portal.cust {{ background:#fff; }}
.lw-portal.off {{ background:{NAVY_DEEP}; border-color:#1F3A63; }}
.lw-portal.off h3, .lw-portal.off p, .lw-portal.off .tag {{ color:#E6EEF9 !important; }}
.lw-portal .tag {{ font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; color:{TEAL}; font-weight:700; }}
.st-key-go_customer button {{ background:{TEAL}; color:#fff; border:0; border-radius:10px; font-weight:700; padding:.7rem; }}
.st-key-go_officer button {{ background:{NAVY}; color:#fff; border:0; border-radius:10px; font-weight:700; padding:.7rem; }}
.st-key-go_officer button:hover {{ background:#16365F; color:#fff; }}
"""


def inject(skin: str) -> None:
    css = {"customer": _CUSTOMER, "officer": _OFFICER, "landing": _LANDING}[skin]
    st.markdown(f"<style>{_BASE}{css}</style>", unsafe_allow_html=True)
