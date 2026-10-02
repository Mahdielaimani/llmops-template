"""Shared helpers for the Phase 4 experiments.

THIS IS A SIMPLIFICATION. These scripts implement attention and a forward pass in
numpy with random weights, to make the shapes and the arithmetic visible. They are
not an inference engine: no KV cache (Phase 5), no fused kernels, no batching
scheduler, no GPU (Phase 6). Generated text is therefore noise — the point is the
mechanics, not the output quality.

The tokenizer, by contrast, is the real GPT-2 tokenizer, because tokenization is
where this project's cost model starts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

# The Windows console is cp1252; GPT-2 tokens contain U+0120 and would raise
# UnicodeEncodeError on print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CACHE = Path(os.getenv("LLMOPS_TOKENIZER_CACHE") or Path.home() / ".cache" / "llmops")

# GPT-2 small's real geometry, so the numbers in docs/inference.md are not invented.
VOCAB = 50257
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 12
D_HEAD = D_MODEL // N_HEADS
D_FF = 4 * D_MODEL
MAX_POS = 1024

SEED = 20261002


def load_tokenizer(name: str = "gpt2") -> Tokenizer:
    """Real GPT-2 BPE. Cached locally so repeat runs need no network."""
    local = CACHE / f"{name}-tokenizer.json"
    if local.exists():
        return Tokenizer.from_file(str(local))
    tok = Tokenizer.from_pretrained(name)
    CACHE.mkdir(parents=True, exist_ok=True)
    tok.save(str(local))
    return tok


def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def init_weights(d_in: int, d_out: int, r: np.random.Generator) -> np.ndarray:
    """Xavier-ish scaling. Without it, attention logits blow up and softmax saturates
    to one-hot, which hides the behaviour these scripts exist to show."""
    return r.normal(0.0, (2.0 / (d_in + d_out)) ** 0.5, size=(d_in, d_out))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Max-subtracted: exp(800) overflows float64, and real logits reach that range."""
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def causal_mask(n: int) -> np.ndarray:
    """-inf above the diagonal: position i must not see i+1, or the model would be
    trained to predict a token it already has."""
    m = np.zeros((n, n))
    m[np.triu_indices(n, k=1)] = -np.inf
    return m


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def kv_bytes_per_token(
    n_layers: int = N_LAYERS, n_kv_heads: int = N_HEADS, d_head: int = D_HEAD, dtype_bytes: int = 2
) -> int:
    """2 (K and V) * layers * kv_heads * head_dim * bytes. The formula behind the
    concurrency ceiling in docs/metrics-and-capacity.md §5."""
    return 2 * n_layers * n_kv_heads * d_head * dtype_bytes
