"""Customer-facing downloadable report (self-contained HTML). Every dynamic value is HTML-escaped."""
from __future__ import annotations

from datetime import datetime
from html import escape

DISCLAIMER = (
    "This is an experimental model assessment from an educational portfolio project. It is NOT a loan offer, "
    "NOT an approval or rejection, and NOT a validated credit decision. A real lender applies its own policy, "
    "verifies your information and makes its own decision."
)


def _when(ts) -> str:
    return datetime.fromtimestamp(ts).strftime("%d %b %Y %H:%M") if ts else "-"


def _row(label: str, value) -> str:
    return f"<tr><th>{escape(str(label))}</th><td>{escape(str(value))}</td></tr>"


def customer_report_html(detail: dict) -> str:
    a, data = detail.get("assessment"), detail.get("data", {})
    derived = data.get("_derived", {})
    inputs = "".join(_row(k.replace("_", " ").capitalize(), v) for k, v in data.items() if not k.startswith("_"))
    parts = [
        f"<h1>LoanWise eligibility assessment (experimental)</h1>",
        f"<p class='meta'>Reference <b>{escape(str(detail['reference']))}</b> &middot; status "
        f"<b>{escape(str(detail['status']))}</b> &middot; submitted {escape(_when(detail.get('submitted_at')))}</p>",
        f"<p class='box'><b>Please read:</b> {escape(DISCLAIMER)}</p>",
    ]
    if a:
        parts += [
            "<h2>What the model estimated</h2><table>",
            _row("Estimated default probability", f"{a['probability']:.1%}"),
            _row("Model review threshold", f"{a['threshold']:.1%}"),
            _row("Risk band (relative to that threshold)", a["band"]),
            _row("Model version", a["model_version"]), "</table>",
        ]
        if a.get("warnings"):
            parts.append("<h3>Input notes</h3><ul>" + "".join(f"<li>{escape(str(w))}</li>" for w in a["warnings"]) + "</ul>")
        if a.get("factors"):
            parts.append("<h2>Inputs that influenced the model most</h2><p>These describe how the model behaves; "
                         "they are not causes and do not show how to change an outcome.</p><ul>")
            parts += [f"<li>{escape(str(f['label']))}: {'raises' if f['contribution'] > 0 else 'lowers'} "
                      f"the modelled risk</li>" for f in a["factors"]]
            parts.append("</ul>")
            parts.append(f"<p class='meta'>Method: {escape(str(a.get('explanation_method', '')))}</p>")
    parts += ["<h2>Assumptions and information used</h2>",
              "<p>The interest rate below is an assumption you chose, not a lender's offer. All figures are "
              "self-reported and unverified.</p><table>"]
    if derived:
        parts += [_row("Estimated monthly instalment (EMI)", f"{derived['emi']:,.2f}"),
                  _row("Debt-to-income used by the model", f"{derived['dti']:.2f}")]
    parts += [inputs, "</table>"]
    body = "".join(parts)
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:760px;margin:2rem auto;color:#0B2545;line-height:1.5}"
           "h1,h2{color:#0B2545}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #d7dee8;"
           "padding:6px 8px;text-align:left}th{width:45%;font-weight:600}.meta{color:#4a5a73}"
           ".box{background:#eaf6f6;border-left:4px solid #13808A;padding:10px 14px}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>LoanWise assessment</title><style>{css}</style></head><body>{body}</body></html>"
