"""Loan arithmetic. Standard reducing-balance (amortising) formulas."""
from __future__ import annotations


def emi(principal: float, annual_rate_pct: float, months: int) -> float:
    """Monthly instalment: P*r*(1+r)^n / ((1+r)^n - 1), r = annual rate / 1200."""
    if principal <= 0 or months <= 0:
        raise ValueError("principal and months must be positive")
    if annual_rate_pct < 0:
        raise ValueError("interest rate cannot be negative")
    r = annual_rate_pct / 1200.0
    if r == 0:
        return principal / months
    growth = (1 + r) ** months
    return principal * r * growth / (growth - 1)


def amortization_schedule(principal: float, annual_rate_pct: float, months: int) -> list[dict]:
    """Month-by-month split of each payment into interest and principal."""
    payment = emi(principal, annual_rate_pct, months)
    r = annual_rate_pct / 1200.0
    balance, rows = principal, []
    for m in range(1, months + 1):
        interest = balance * r
        principal_part = payment - interest
        balance = max(0.0, balance - principal_part)
        rows.append({"month": m, "payment": payment, "interest": interest,
                     "principal": principal_part, "balance": balance if m < months else 0.0})
    return rows


def summary(principal: float, annual_rate_pct: float, months: int) -> dict:
    payment = emi(principal, annual_rate_pct, months)
    total = payment * months
    return {"emi": payment, "total_repayment": total, "total_interest": total - principal}


def max_principal(monthly_payment: float, annual_rate_pct: float, months: int) -> float:
    """Inverse of ``emi``: the largest loan a given monthly payment can service."""
    if monthly_payment <= 0 or months <= 0:
        return 0.0
    r = annual_rate_pct / 1200.0
    if r == 0:
        return monthly_payment * months
    growth = (1 + r) ** months
    return monthly_payment * (growth - 1) / (r * growth)


def payment_capacity(annual_income: float, monthly_obligations: float, cap_ratio: float = 0.40) -> float:
    """Illustrative monthly room for a new payment: cap x monthly income - existing obligations.

    ``cap_ratio`` is an adjustable *assumption*, not a lender's policy."""
    return max(0.0, annual_income / 12.0 * cap_ratio - monthly_obligations)


def debt_to_income(annual_income: float, monthly_obligations: float, new_emi: float) -> float:
    """(existing monthly obligations + proposed EMI) / monthly income."""
    return (monthly_obligations + new_emi) / (annual_income / 12.0)
