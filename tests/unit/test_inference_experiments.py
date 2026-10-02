"""The experiments are teaching material, but the claims in docs/inference.md are
derived from them, so the claims get tests. A silently wrong reference document is
worse than no document.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[2] / "inference" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from _shared import (  # noqa: E402
    D_HEAD,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    VOCAB,
    causal_mask,
    gelu,
    init_weights,
    kv_bytes_per_token,
    layer_norm,
    load_tokenizer,
    rng,
    softmax,
)


def test_softmax_rows_sum_to_one_and_survive_large_logits() -> None:
    p = softmax(np.array([800.0, 799.0, -800.0]))
    assert np.isclose(p.sum(), 1.0)
    assert not np.isnan(p).any()  # naive exp() would overflow to inf/nan here


def test_softmax_is_shift_invariant() -> None:
    logits = rng().normal(0, 4, size=1000)
    assert np.allclose(softmax(logits), softmax(logits + 1000.0))


def test_layer_norm_normalises_and_does_not_shrink() -> None:
    """docs/inference.md §2 claims the norm rises to ~sqrt(d_model). Pin it."""
    x = rng().normal(0, 0.02, size=(8, D_MODEL))
    xn = layer_norm(x)
    assert abs(float(xn.mean())) < 1e-10
    assert 0.98 < float(xn.std()) < 1.02
    assert np.linalg.norm(xn[0]) > np.linalg.norm(x[0])
    assert np.isclose(np.linalg.norm(xn[0]), D_MODEL**0.5, rtol=0.05)


def test_causal_mask_blocks_only_the_future() -> None:
    n = 6
    m = causal_mask(n)
    assert np.all(np.isneginf(m[np.triu_indices(n, k=1)]))
    assert np.all(m[np.tril_indices(n)] == 0.0)


def test_masked_attention_cannot_see_forward() -> None:
    """The claim that makes a KV cache valid: zero weight on future positions."""
    n = 7
    r = rng()
    x = layer_norm(r.normal(0, 1, size=(n, D_MODEL)))
    q, k = x @ init_weights(D_MODEL, D_MODEL, r), x @ init_weights(D_MODEL, D_MODEL, r)
    w = softmax(q @ k.T / np.sqrt(D_HEAD) + causal_mask(n))
    assert np.allclose(w.sum(axis=-1), 1.0)
    assert np.allclose(w[np.triu_indices(n, k=1)], 0.0)
    assert w[0, 0] == pytest.approx(1.0)  # first token attends only to itself


def test_scaling_keeps_softmax_soft() -> None:
    """§3: unscaled scores saturate the softmax. Asserted, not just described."""
    r = rng()
    x = layer_norm(r.normal(0, 1, size=(16, D_MODEL)))
    q, k = x @ init_weights(D_MODEL, D_MODEL, r), x @ init_weights(D_MODEL, D_MODEL, r)
    raw = q @ k.T
    assert softmax(raw[-1]).max() > softmax((raw / np.sqrt(D_HEAD))[-1]).max()


def test_attention_without_positions_is_permutation_invariant() -> None:
    r = rng()
    wte = r.normal(0, 0.02, size=(100, D_MODEL))
    ids = [5, 17, 42, 8]
    assert np.allclose(wte[ids].sum(axis=0), wte[list(reversed(ids))].sum(axis=0))


def test_head_split_is_a_reshape_not_a_loss() -> None:
    n = 5
    x = rng().normal(0, 1, size=(n, D_MODEL))
    heads = x.reshape(n, N_HEADS, D_HEAD).transpose(1, 0, 2)
    assert heads.shape == (N_HEADS, n, D_HEAD)
    assert np.allclose(heads.transpose(1, 0, 2).reshape(n, D_MODEL), x)


def test_kv_bytes_matches_the_documented_formula() -> None:
    """36 KiB/token for GPT-2 geometry; 56 KiB for Qwen2.5-7B's GQA. Both appear in
    docs/inference.md §3 and docs/metrics-and-capacity.md §5."""
    assert kv_bytes_per_token() == 2 * N_LAYERS * N_HEADS * D_HEAD * 2 == 36_864
    qwen = kv_bytes_per_token(n_layers=28, n_kv_heads=4, d_head=128, dtype_bytes=2)
    assert qwen == 57_344
    assert round(qwen / 1024) == 56


def test_gqa_is_what_makes_the_cache_affordable() -> None:
    """Qwen2.5-7B has 28 query heads but only 4 KV heads."""
    mha = kv_bytes_per_token(n_layers=28, n_kv_heads=28, d_head=128)
    gqa = kv_bytes_per_token(n_layers=28, n_kv_heads=4, d_head=128)
    assert mha / gqa == 7.0


def test_gelu_is_smooth_and_passes_through_zero() -> None:
    assert gelu(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-6)
    assert gelu(np.array([10.0]))[0] == pytest.approx(10.0, rel=1e-3)
    assert -0.2 < gelu(np.array([-2.0]))[0] < 0.0  # unlike ReLU, not clipped to 0


def test_parameter_count_matches_gpt2_small() -> None:
    per_block = 4 * D_MODEL * D_MODEL + 2 * D_MODEL * (4 * D_MODEL)
    total = VOCAB * D_MODEL + 1024 * D_MODEL + N_LAYERS * per_block
    assert 120e6 < total < 128e6  # published ~124M


def test_financial_text_tokenizes_worse_than_prose() -> None:
    """The §1 finding that changes the capacity model. If this ever stops holding,
    docs/inference.md and the chunk-size assumptions need revisiting."""
    tok = load_tokenizer()
    prose = "The company reported strong growth in the second quarter."
    finance = "Q2 FY2026 revenue was EUR 1,234,567.89, up 12.4% against plan."
    prose_ratio = len(prose) / len(tok.encode(prose).ids)
    finance_ratio = len(finance) / len(tok.encode(finance).ids)
    assert prose_ratio > 5.0
    assert finance_ratio < 3.0
    assert prose_ratio / finance_ratio > 1.8


def test_whitespace_and_case_change_token_ids() -> None:
    tok = load_tokenizer()
    ids = {v: tuple(tok.encode(v).ids) for v in ["revenue", " revenue", "Revenue", "REVENUE"]}
    assert len(set(ids.values())) == 4
    assert len(ids[" revenue"]) == 1  # leading space is inside the token
    assert len(ids["revenue"]) == 2


def test_tokenizer_round_trip_is_lossless() -> None:
    tok = load_tokenizer()
    text = "Invoice INV-2026-0004471: EUR 1,234,567.89 due 2026-11-30."
    assert tok.decode(tok.encode(text).ids) == text
