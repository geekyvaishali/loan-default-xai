"""The single inference interface used by the dashboard, the API and tests."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .attribution import explain_row
from .config import ARTIFACT_DIR, RAW_FEATURES
from .schema import SchemaError, ValidationReport, validate_frame

CHUNK_SIZE = 50_000


class LoanRiskPredictor:
    """Loads versioned artifacts and scores raw applicant rows.

    The model is a risk *score* (calibrated on the training data's own default rate).
    ``flagged`` is a separate, configurable business rule: probability >= threshold.
    """

    def __init__(self, model, metadata: dict, schema: dict, threshold: float | None = None):
        self.model = model
        self.metadata = metadata
        self.schema = schema
        self.threshold = float(threshold if threshold is not None else metadata["threshold"]["threshold"])

    @classmethod
    def load(cls, artifact_dir: Path | str = ARTIFACT_DIR, threshold: float | None = None) -> "LoanRiskPredictor":
        d = Path(artifact_dir)
        needed = ["model.joblib", "metadata.json", "schema.json"]
        missing = [f for f in needed if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing artifact files in {d}: {missing}. Run `python -m src.train` first.")
        return cls(joblib.load(d / "model.joblib"),
                   json.loads((d / "metadata.json").read_text(encoding="utf-8")),
                   json.loads((d / "schema.json").read_text(encoding="utf-8")), threshold)

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        return validate_frame(df)

    def score(self, df: pd.DataFrame, on_invalid: str = "raise") -> pd.DataFrame:
        """Score raw applicant rows.

        Returns a frame with the same index and columns: probability, flagged, status, warnings.
        ``on_invalid='raise'`` rejects the whole batch; ``'mark'`` scores valid rows only.
        """
        if on_invalid not in {"raise", "mark"}:
            raise ValueError("on_invalid must be 'raise' or 'mark'")
        report = validate_frame(df)
        if report.missing_columns:
            raise SchemaError(f"Missing required columns: {report.missing_columns}")
        if report.row_errors and on_invalid == "raise":
            pos = report.invalid_positions[0]
            raise SchemaError(f"{len(report.row_errors)} invalid row(s); first (row {pos}): {report.row_errors[pos]}")

        invalid = np.zeros(len(df), dtype=bool)
        invalid[report.invalid_positions] = True
        proba = np.full(len(df), np.nan)
        valid_idx = np.flatnonzero(~invalid)
        for start in range(0, len(valid_idx), CHUNK_SIZE):
            chunk = valid_idx[start:start + CHUNK_SIZE]
            proba[chunk] = self.model.predict_proba(df.iloc[chunk][RAW_FEATURES])[:, 1]

        out = pd.DataFrame({"probability": proba}, index=df.index)
        out["flagged"] = pd.array(np.where(invalid, pd.NA, proba >= self.threshold), dtype="boolean")
        out["status"] = [("invalid: " + "; ".join(report.row_errors[i])) if invalid[i] else "ok" for i in range(len(df))]
        out["warnings"] = ["; ".join(report.row_warnings.get(i, [])) for i in range(len(df))]
        return out

    def predict_one(self, record: dict) -> dict:
        row = pd.DataFrame([record])
        res = self.score(row).iloc[0]
        return {"probability": float(res["probability"]), "flagged": bool(res["flagged"]),
                "threshold": self.threshold, "model_version": self.version,
                "warnings": [w for w in str(res["warnings"]).split("; ") if w]}

    def explain(self, record: dict, top_n: int = 8) -> dict:
        """Per-applicant attribution of the model score (see src/attribution.py for the method used)."""
        row = pd.DataFrame([record])[RAW_FEATURES]
        return explain_row(self.model, row, self.schema, top_n)
