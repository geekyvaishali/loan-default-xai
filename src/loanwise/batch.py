"""Officer batch assessment: safe CSV ingestion + scoring."""
from __future__ import annotations

import io

import pandas as pd

from ..config import RAW_FEATURES
from ..predict import LoanRiskPredictor
from ..schema import validate_frame

MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000


class BatchError(ValueError):
    """User-fixable problem with an uploaded file (message is safe to show)."""


def score_csv(predictor: LoanRiskPredictor, raw: bytes, max_bytes: int = MAX_BYTES,
              max_rows: int = MAX_ROWS) -> tuple[pd.DataFrame, dict]:
    if not raw or not raw.strip():
        raise BatchError("The file is empty.")
    if len(raw) > max_bytes:
        raise BatchError(f"The file is {len(raw) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise BatchError(f"The file could not be read as CSV ({type(exc).__name__}).") from exc
    if df.empty:
        raise BatchError("The file has a header but no rows.")
    if len(df) > max_rows:
        raise BatchError(f"The file has {len(df):,} rows; the limit is {max_rows:,}.")
    report = validate_frame(df)
    if report.missing_columns:
        raise BatchError("Missing required columns: " + ", ".join(report.missing_columns))
    scored = predictor.score(df, on_invalid="mark")
    out = pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    summary = {"rows": len(out), "scored": int((scored["status"] == "ok").sum()),
               "invalid": int((scored["status"] != "ok").sum()),
               "flagged": int(scored["flagged"].fillna(False).astype(bool).sum()),
               "mean_probability": float(scored["probability"].mean()),
               "threshold": predictor.threshold, "model_version": predictor.version,
               "required_columns": RAW_FEATURES}
    return out, summary
