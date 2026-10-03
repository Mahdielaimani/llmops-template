"""Embeddings via fastembed (ONNX).

fastembed rather than sentence-transformers: same model, ~67 MB against ~1 GB with
torch, and disk is this machine's binding constraint (ADR-020). The asymmetry
between document and query embedding matters for retrieval quality — some models
expect a query prefix — so the two calls are separate methods rather than one.
"""

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

from llmops_core.logging import get_logger

log = get_logger("embedding")

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained with this instruction on the query side only. Omitting it
# costs a few points of recall; applying it to documents too also costs recall.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))
        log.info("embedder_ready", model=model_name, dim=self.dim)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return list(self._model.embed(texts, batch_size=self.batch_size))

    def embed_query(self, text: str) -> np.ndarray:
        return next(iter(self._model.embed([QUERY_PREFIX + text])))
