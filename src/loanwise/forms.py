"""Customer application form: field definitions, step validation and model-input mapping."""
from __future__ import annotations

from ..schema import ALLOWED_CATEGORIES, TRAINING_RANGE
from . import finance
from .errors import ValidationError

STEP_TITLES = ["About you", "Income & obligations", "Loan details", "Review"]
TENURE_OPTIONS = [12, 24, 36, 48, 60]   # the model was trained on 12-60 month loans only
LOAN_PURPOSES = ALLOWED_CATEGORIES["LoanPurpose"]

# field -> (step, label, kind, constraint)
FIELDS: dict[str, tuple[int, str, str, object]] = {
    "age": (1, "Age", "int", (18, 100)),
    "education": (1, "Education", "choice", ALLOWED_CATEGORIES["Education"]),
    "marital_status": (1, "Marital status", "choice", ALLOWED_CATEGORIES["MaritalStatus"]),
    "employment_type": (1, "Employment type", "choice", ALLOWED_CATEGORIES["EmploymentType"]),
    "months_employed": (1, "Months in current employment", "int", (0, 720)),
    "has_dependents": (1, "Financial dependents", "choice", ALLOWED_CATEGORIES["HasDependents"]),
    "annual_income": (2, "Annual income", "float", (1_000, 10_000_000)),
    "monthly_obligations": (2, "Existing monthly debt payments", "float", (0, 1_000_000)),
    "credit_score": (2, "Credit score (self-reported)", "int", (300, 850)),
    "num_credit_lines": (2, "Open credit lines", "int", (0, 50)),
    "has_mortgage": (2, "Has a mortgage", "choice", ALLOWED_CATEGORIES["HasMortgage"]),
    "loan_amount": (3, "Loan amount", "float", (1_000, 10_000_000)),
    "loan_purpose": (3, "Loan purpose", "choice", LOAN_PURPOSES),
    "interest_rate": (3, "Assumed annual interest rate (%)", "float", (1.0, 40.0)),
    "loan_term_months": (3, "Tenure (months)", "choice", TENURE_OPTIONS),
    "has_cosigner": (3, "Has a co-signer", "choice", ALLOWED_CATEGORIES["HasCoSigner"]),
}

DEFAULTS = {
    "age": 35, "education": "Bachelor's", "marital_status": "Single", "employment_type": "Full-time",
    "months_employed": 36, "has_dependents": "No", "annual_income": 60_000.0, "monthly_obligations": 500.0,
    "credit_score": 650, "num_credit_lines": 3, "has_mortgage": "No", "loan_amount": 20_000.0,
    "loan_purpose": "Other", "interest_rate": 10.0, "loan_term_months": 36, "has_cosigner": "No",
}


def _coerce(field: str, value):
    _, label, kind, constraint = FIELDS[field]
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{label} is required.")
    if kind == "choice":
        if value not in constraint:
            raise ValueError(f"{label}: choose one of {', '.join(map(str, constraint))}.")
        return value
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.")
    if kind == "int":
        if num != int(num):
            raise ValueError(f"{label} must be a whole number.")
        num = int(num)
    lo, hi = constraint
    if not lo <= num <= hi:
        raise ValueError(f"{label} must be between {lo:,} and {hi:,}.")
    return num


def validate_step(step: int, data: dict) -> dict[str, str]:
    """Return {field: message} for one step (empty dict when valid), including cross-field rules."""
    errors: dict[str, str] = {}
    for field, (fstep, *_rest) in FIELDS.items():
        if fstep != step:
            continue
        try:
            _coerce(field, data.get(field))
        except ValueError as exc:
            errors[field] = str(exc)
    if step == 2 and not errors:
        if _coerce("monthly_obligations", data["monthly_obligations"]) > _coerce("annual_income", data["annual_income"]) / 12:
            errors["monthly_obligations"] = "Monthly debt payments exceed your monthly income; please check both values."
    return errors


def validate_all(data: dict) -> dict:
    """Validate steps 1-3 and return cleaned values. Raises ValidationError listing every problem."""
    errors: dict[str, str] = {}
    for step in (1, 2, 3):
        errors.update(validate_step(step, data))
    if errors:
        raise ValidationError("Please correct the highlighted fields.", errors)
    return {f: _coerce(f, data[f]) for f in FIELDS}


def derived_figures(clean: dict) -> dict:
    """EMI and debt-to-income derived from the customer's inputs."""
    pay = finance.emi(clean["loan_amount"], clean["interest_rate"], clean["loan_term_months"])
    dti = finance.debt_to_income(clean["annual_income"], clean["monthly_obligations"], pay)
    return {"emi": pay, "dti": dti}


def model_record(clean: dict) -> dict:
    """Map the cleaned customer inputs to the model's raw feature columns (same names as training)."""
    derived = derived_figures(clean)
    return {
        "Age": clean["age"], "Income": clean["annual_income"], "LoanAmount": clean["loan_amount"],
        "CreditScore": clean["credit_score"], "MonthsEmployed": clean["months_employed"],
        "NumCreditLines": clean["num_credit_lines"], "InterestRate": clean["interest_rate"],
        "LoanTerm": clean["loan_term_months"], "DTIRatio": round(derived["dti"], 4),
        "Education": clean["education"], "EmploymentType": clean["employment_type"],
        "MaritalStatus": clean["marital_status"], "HasMortgage": clean["has_mortgage"],
        "HasDependents": clean["has_dependents"], "LoanPurpose": clean["loan_purpose"],
        "HasCoSigner": clean["has_cosigner"],
    }


assert set(TRAINING_RANGE) <= set(model_record(validate_all(DEFAULTS)))
