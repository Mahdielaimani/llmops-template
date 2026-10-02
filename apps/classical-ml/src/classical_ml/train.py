"""Train the invoice late-payment classifier and write a versioned artifact.

Run: uv run python -m classical_ml.train

The artifact is the point of this phase. It carries its own metadata — feature
order, metrics, training data hash — because a `.joblib` with no provenance is
the classical-ML equivalent of an unversioned prompt (ADR-015, artifact class 3).
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "1.0.0"
SEED = 20260926
ARTIFACT_DIR = Path(__file__).resolve().parents[4] / "models" / "invoice-late-payment"

FEATURES = [
    "amount_eur",
    "days_to_due",
    "customer_prior_invoices",
    "customer_prior_late_ratio",
    "has_purchase_order",
    "is_new_customer",
]


def synthesize(n: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic, because the lab has no real invoices (docs/business-requirements.md §6).

    The signal is deliberately learnable but not trivial: prior late ratio dominates,
    amount and tight due dates contribute, and 5% label noise stops the model from
    reaching an implausible 100%. Signal strength was tuned until ROC AUC landed in
    the 0.80-0.90 band — a first pass produced 0.61, which on a 76% majority class
    is no better than always predicting "on time".
    """
    rng = np.random.default_rng(SEED)
    amount = rng.lognormal(mean=8.0, sigma=1.1, size=n)
    days_to_due = rng.integers(7, 91, size=n).astype(float)
    prior = rng.integers(0, 60, size=n).astype(float)
    late_ratio = np.clip(rng.beta(2, 8, size=n), 0, 1)
    has_po = rng.binomial(1, 0.7, size=n).astype(float)
    is_new = (prior < 3).astype(float)

    # Centred and scaled so each term contributes real variance rather than a
    # near-constant offset: log1p(lognormal(8, 1.1)) is ~8 +/- 1, so the raw term
    # was effectively a bias.
    logit = (
        6.0 * (late_ratio - 0.2)
        + 0.9 * (np.log1p(amount) - 8.0)
        - 0.035 * (days_to_due - 48)
        - 0.8 * has_po
        + 0.7 * is_new
        - 1.0
    )
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    flip = rng.random(n) < 0.05
    y = np.where(flip, 1 - y, y)

    x = np.column_stack([amount, days_to_due, prior, late_ratio, has_po, is_new])
    return x, y


def train() -> Path:
    x, y = synthesize()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = Pipeline(
        [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, C=1.0))]
    )
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    # Majority-class accuracy, reported alongside: on an imbalanced target, accuracy
    # on its own hides a model that has learned nothing.
    baseline = max(float(y_test.mean()), 1 - float(y_test.mean()))
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "baseline_accuracy": round(baseline, 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "positive_rate": round(float(y.mean()), 4),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / f"model-{MODEL_VERSION}.joblib"
    joblib.dump(model, model_path, compress=3)

    meta = {
        "name": "invoice-late-payment",
        "version": MODEL_VERSION,
        "algorithm": "LogisticRegression + StandardScaler",
        "features": FEATURES,
        "target": "paid_late",
        "seed": SEED,
        "data": "synthetic",
        "data_sha256": hashlib.sha256(x.tobytes()).hexdigest()[:16],
        "artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "artifact_bytes": model_path.stat().st_size,
        "metrics": metrics,
        "trained_at": datetime.now(UTC).isoformat(),
        "sklearn": __import__("sklearn").__version__,
        "python": platform.python_version(),
    }
    (ARTIFACT_DIR / f"model-{MODEL_VERSION}.json").write_text(json.dumps(meta, indent=2))
    (ARTIFACT_DIR / "current.txt").write_text(MODEL_VERSION)
    return model_path


if __name__ == "__main__":
    path = train()
    meta = json.loads(path.with_suffix(".json").read_text())
    print(f"wrote {path} ({meta['artifact_bytes'] / 1024:.1f} KiB)")
    print(json.dumps(meta["metrics"], indent=2))
