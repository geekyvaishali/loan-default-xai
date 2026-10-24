"""Turns validated application data into a model assessment (uses the shared predictor)."""
from __future__ import annotations

from ..predict import LoanRiskPredictor
from ..schema import SchemaError
from . import forms
from .errors import ValidationError


def risk_band(probability: float, threshold: float) -> str:
    """Project convention (not lending policy): bands are multiples of the review threshold."""
    if probability < threshold:
        return "Low"
    return "Elevated" if probability < 2 * threshold else "High"


def run_assessment(predictor: LoanRiskPredictor, clean: dict) -> dict:
    record = forms.model_record(clean)
    try:
        res = predictor.predict_one(record)
    except SchemaError as exc:
        raise ValidationError(f"These inputs cannot be assessed: {exc}") from exc
    explanation = predictor.explain(record)
    return {
        "probability": res["probability"], "threshold": res["threshold"], "flagged": res["flagged"],
        "band": risk_band(res["probability"], res["threshold"]), "model_version": res["model_version"],
        "warnings": res["warnings"], "factors": explanation["factors"],
        "explanation_method": explanation["method"], "inputs": record,
    }
