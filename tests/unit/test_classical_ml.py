from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from classical_ml.model import ARTIFACT_DIR, Invoice, load_model
from llmops_core.config import Settings
from llmops_core.context import REQUEST_ID_HEADER

INVOICE = {
    "amount_eur": 24500.0,
    "days_to_due": 30,
    "customer_prior_invoices": 12,
    "customer_prior_late_ratio": 0.25,
    "has_purchase_order": True,
    "is_new_customer": False,
}

pytestmark = pytest.mark.skipif(
    not (ARTIFACT_DIR / "current.txt").exists(),
    reason="no trained artifact; run `uv run python -m classical_ml.train`",
)


@pytest.fixture
def ml_client(test_settings: Settings) -> Iterator[TestClient]:
    from classical_ml.main import create_app

    with TestClient(create_app(test_settings)) as c:
        yield c


def test_predict_returns_probability_and_version(ml_client: TestClient) -> None:
    r = ml_client.post("/predict", json=INVOICE, headers={REQUEST_ID_HEADER: "ml-1"})
    assert r.status_code == 200
    b = r.json()
    assert 0.0 <= b["probability"] <= 1.0
    assert b["paid_late"] is (b["probability"] >= b["threshold"])
    assert b["model_name"] == "invoice-late-payment"
    assert b["model_version"] == "1.0.0"
    assert b["request_id"] == "ml-1"


def test_prediction_is_deterministic(ml_client: TestClient) -> None:
    """The headline difference from /chat: identical input, byte-identical output,
    every time. No temperature, no sampling, no seed to pin."""
    first = ml_client.post("/predict", json=INVOICE).json()["probability"]
    for _ in range(5):
        assert ml_client.post("/predict", json=INVOICE).json()["probability"] == first


@pytest.mark.parametrize(
    "bad",
    [
        {"amount_eur": -1},
        {"amount_eur": "twenty thousand"},
        {"days_to_due": 400},
        {"customer_prior_late_ratio": 1.5},
        {"has_purchase_order": "maybe"},
    ],
)
def test_schema_rejects_bad_input(ml_client: TestClient, bad: dict[str, object]) -> None:
    """A fixed schema can refuse. An LLM given "amount: twenty thousand" answers anyway."""
    r = ml_client.post("/predict", json={**INVOICE, **bad})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_missing_field_is_rejected(ml_client: TestClient) -> None:
    body = {k: v for k, v in INVOICE.items() if k != "days_to_due"}
    assert ml_client.post("/predict", json=body).status_code == 422


def test_batch_matches_single(ml_client: TestClient) -> None:
    single = ml_client.post("/predict", json=INVOICE).json()["probability"]
    batch = ml_client.post("/predict/batch", json={"invoices": [INVOICE] * 3}).json()
    assert batch["count"] == 3
    assert {p["probability"] for p in batch["predictions"]} == {single}


def test_batch_size_is_capped(ml_client: TestClient) -> None:
    r = ml_client.post("/predict/batch", json={"invoices": [INVOICE] * 1001})
    assert r.status_code == 422


def test_model_card_exposes_provenance(ml_client: TestClient) -> None:
    card = ml_client.get("/model").json()
    assert card["features"] == [
        "amount_eur",
        "days_to_due",
        "customer_prior_invoices",
        "customer_prior_late_ratio",
        "has_purchase_order",
        "is_new_customer",
    ]
    assert len(card["artifact_sha256"]) == 64
    assert card["data"] == "synthetic"
    assert card["metrics"]["roc_auc"] > 0.5


def test_accuracy_is_reported_against_a_baseline() -> None:
    """Accuracy alone is a lie on an imbalanced target: this model scores 0.765 while
    always predicting "on time" scores 0.726. The artifact must carry both."""
    meta = json.loads((ARTIFACT_DIR / "model-1.0.0.json").read_text())
    m = meta["metrics"]
    assert m["accuracy"] > m["baseline_accuracy"]
    assert m["roc_auc"] > 0.5
    assert 0.0 < m["recall"] <= 1.0


def test_feature_order_is_the_contract() -> None:
    """to_vector() must match the order the model was fitted on, or predictions are
    silently wrong rather than erroring."""
    meta = json.loads((ARTIFACT_DIR / "model-1.0.0.json").read_text())
    vector = Invoice(**INVOICE).to_vector()  # type: ignore[arg-type]
    assert len(vector) == len(meta["features"])
    assert vector[0] == INVOICE["amount_eur"]
    assert vector[3] == INVOICE["customer_prior_late_ratio"]


def test_missing_artifact_is_a_clean_error(tmp_path: Path) -> None:
    with pytest.raises(Exception) as exc:
        load_model(tmp_path)
    assert exc.value.code == "not_found"  # type: ignore[attr-defined]
    assert "classical_ml.train" in str(exc.value)


def test_artifact_is_small_enough_to_load_instantly() -> None:
    """~1 KiB versus 4-15 GB of LLM weights: the number that explains why one service
    needs a startupProbe and the other does not."""
    meta = json.loads((ARTIFACT_DIR / "model-1.0.0.json").read_text())
    assert meta["artifact_bytes"] < 100_000
