"""Artifact loading and inference.

The model is loaded once at startup and held in memory. That single sentence is the
whole operational difference from LLM serving: a 30 KiB artifact, a fixed feature
vector in, one float out, deterministic, no KV cache, no batching scheduler.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, Field

from llmops_core.errors import NotFoundError

# Repo-relative by default, but overridable: a non-editable install moves __file__
# into site-packages and the relative walk breaks.
ARTIFACT_DIR = Path(
    os.getenv("LLMOPS_MODEL_DIR")
    or Path(__file__).resolve().parents[4] / "models" / "invoice-late-payment"
)


class Invoice(BaseModel):
    """Fixed schema, validated at the edge. An LLM takes free text; this takes six
    numbers, and a wrong type is a 422 rather than a plausible-looking answer."""

    amount_eur: float = Field(gt=0, le=10_000_000)
    days_to_due: int = Field(ge=0, le=365)
    customer_prior_invoices: int = Field(ge=0, le=100_000)
    customer_prior_late_ratio: float = Field(ge=0.0, le=1.0)
    has_purchase_order: bool
    is_new_customer: bool

    def to_vector(self) -> list[float]:
        return [
            self.amount_eur,
            float(self.days_to_due),
            float(self.customer_prior_invoices),
            self.customer_prior_late_ratio,
            float(self.has_purchase_order),
            float(self.is_new_customer),
        ]


class Prediction(BaseModel):
    paid_late: bool
    probability: float
    threshold: float
    model_name: str
    model_version: str
    request_id: str | None
    latency_ms: float


@dataclass
class LoadedModel:
    name: str
    version: str
    estimator: Any
    metadata: dict[str, Any]
    # 0.5 is the default, not a chosen operating point. At this threshold recall is
    # 0.28: the model flags few late invoices. Picking a threshold needs the business
    # cost of a missed late payment versus a false alarm, which this lab does not have.
    threshold: float = 0.5

    def predict(self, invoices: list[Invoice]) -> tuple[list[bool], list[float]]:
        x = np.asarray([i.to_vector() for i in invoices], dtype=np.float64)
        proba = self.estimator.predict_proba(x)[:, 1]
        return [bool(p >= self.threshold) for p in proba], [round(float(p), 6) for p in proba]


def load_model(directory: Path = ARTIFACT_DIR) -> LoadedModel:
    pointer = directory / "current.txt"
    if not pointer.exists():
        raise NotFoundError(
            f"no model artifact in {directory}; run `uv run python -m classical_ml.train`"
        )
    version = pointer.read_text().strip()
    meta = json.loads((directory / f"model-{version}.json").read_text())
    return LoadedModel(
        name=meta["name"],
        version=version,
        estimator=joblib.load(directory / f"model-{version}.joblib"),
        metadata=meta,
    )
