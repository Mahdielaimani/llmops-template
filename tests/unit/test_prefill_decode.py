"""Phase 5 claims, asserted. The important one is that the KV cache is exact:
if cached and uncached generation ever diverge, the mask or the position offset
is wrong and docs/prefill-decode.md is lying.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[2] / "inference" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))


def _load(stem: str):
    """The experiment modules start with digits, so they are not importable by name."""
    path = EXPERIMENTS / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.lstrip("0123456789_"), path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves cls.__module__ through sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pd = _load("06_prefill_decode")
kv = _load("07_kv_cache")

from _shared import N_HEADS, rng  # noqa: E402

PROMPT_IDS = [464, 1664, 198, 1913, 1349, 287, 262, 1218, 1860, 13]  # all < TEST_VOCAB


# Small vocabulary: the real table is 294 MiB of float64 and these tests assert
# shapes and byte formulas, neither of which depends on vocabulary size.
TEST_VOCAB = 2048


@pytest.fixture(scope="module")
def weights():
    return pd.build(rng(), vocab=TEST_VOCAB)


def test_kv_cache_is_exact_not_an_approximation(weights) -> None:
    """The headline claim of docs/prefill-decode.md §2."""
    naive, _ = pd.generate_no_cache(weights, PROMPT_IDS, 12)
    cached, _, _, _ = pd.generate_with_cache(weights, PROMPT_IDS, 12)
    assert naive == cached


def test_a_wrong_position_offset_breaks_equality(weights) -> None:
    """Proves the previous test can fail — a test that cannot fail proves nothing.
    Feeding the wrong offset shifts the positional embedding and the output diverges."""
    caches = [pd.LayerCache() for _ in weights.blocks]
    pd.forward(weights, PROMPT_IDS, caches, offset=0)
    good = pd.forward(weights, [PROMPT_IDS[-1]], caches, offset=caches[0].n_tokens)
    caches2 = [pd.LayerCache() for _ in weights.blocks]
    pd.forward(weights, PROMPT_IDS, caches2, offset=0)
    bad = pd.forward(weights, [PROMPT_IDS[-1]], caches2, offset=0)
    assert int(np.argmax(good[-1])) != int(np.argmax(bad[-1]))


def test_cached_decode_does_constant_work_per_step(weights) -> None:
    """The real claim behind §4, asserted deterministically rather than by wall clock.

    A timing ratio on a laptop running five containers flakes; the structural fact
    does not. Uncached, step t projects K and V for all t tokens; cached, it projects
    exactly one, whatever t is. Timing numbers live in docs/prefill-decode.md §4,
    where they are labelled as measured and noisy.
    """
    caches = [pd.LayerCache() for _ in weights.blocks]
    pd.forward(weights, PROMPT_IDS, caches, offset=0)

    widths = []
    for _ in range(6):
        before = caches[0].n_tokens
        pd.forward(weights, [0], caches, offset=before)
        widths.append(caches[0].n_tokens - before)

    assert widths == [1] * 6  # constant work per step regardless of sequence length

    # The uncached path, by contrast, re-projects the whole sequence every step.
    naive_widths = [len(PROMPT_IDS) + i for i in range(6)]
    assert naive_widths[-1] > naive_widths[0]
    assert sum(naive_widths) > sum(widths) * 10


def test_cache_grows_by_exactly_one_token_per_decode_step(weights) -> None:
    caches = [pd.LayerCache() for _ in weights.blocks]
    pd.forward(weights, PROMPT_IDS, caches, offset=0)
    assert caches[0].n_tokens == len(PROMPT_IDS)
    for expected in range(len(PROMPT_IDS) + 1, len(PROMPT_IDS) + 5):
        pd.forward(weights, [0], caches, offset=caches[0].n_tokens)
        assert caches[0].n_tokens == expected


def test_measured_kv_bytes_equal_the_formula(weights) -> None:
    """§5: the allocation is predicted exactly, not approximately."""
    n_new = 10
    _, _, _, measured = pd.generate_with_cache(weights, PROMPT_IDS, n_new)
    tokens = len(PROMPT_IDS) + n_new - 1
    expected = 2 * pd.N_LAYERS_TOY * N_HEADS * pd.D_HEAD * 8 * tokens  # float64
    assert measured == expected


def test_decode_step_shapes(weights) -> None:
    """One new token in, one row of logits out — not P rows."""
    caches = [pd.LayerCache() for _ in weights.blocks]
    prefill_logits = pd.forward(weights, PROMPT_IDS, caches, offset=0)
    assert prefill_logits.shape[0] == len(PROMPT_IDS)  # prefill: all positions
    step_logits = pd.forward(weights, [0], caches, offset=caches[0].n_tokens)
    assert step_logits.shape[0] == 1  # decode: one


# --------------------------------------------------------------------------- KV budget


def test_gqa_ratio_drives_cache_size_not_parameter_count() -> None:
    """docs/kv-cache.md §2: only kv_heads matters. Same params, 7x the cache."""
    qwen = next(m for m in kv.MODELS if "7B-Instruct-AWQ" in m.name)
    mha = kv.Model("hypothetical MHA", qwen.params_b, qwen.layers, 28, 28, qwen.d_head, 0.5, 32768)
    assert mha.kv_bytes_per_token() / qwen.kv_bytes_per_token() == 7.0
    assert round(qwen.kv_bytes_per_token() / 1024) == 56
    assert round(mha.kv_bytes_per_token() / 1024) == 392


def test_quantizing_weights_does_not_shrink_the_cache() -> None:
    """A common misconception worth pinning: int4 buys room for a cache, it does not
    make the cache smaller."""
    fp16 = next(m for m in kv.MODELS if m.name == "Qwen2.5-7B-Instruct (fp16)")
    int4 = next(m for m in kv.MODELS if "AWQ" in m.name and "Qwen" in m.name)
    assert int4.weight_bytes() < fp16.weight_bytes() / 3
    assert int4.kv_bytes_per_token() == fp16.kv_bytes_per_token()


def test_fp16_7b_does_not_fit_on_this_card() -> None:
    m = next(x for x in kv.MODELS if x.name == "Qwen2.5-7B-Instruct (fp16)")
    budget = kv.VRAM_TOTAL - m.weight_bytes() - kv.ENGINE_OVERHEAD - kv.ACTIVATIONS
    assert budget < 0


def test_concurrency_ceiling_matches_the_capacity_doc() -> None:
    """Independent derivation of metrics-and-capacity.md §5: ~11 at 4k, ~5 at 8k."""
    m = next(x for x in kv.MODELS if "7B-Instruct-AWQ" in x.name)
    budget = kv.VRAM_TOTAL - m.weight_bytes() - kv.ENGINE_OVERHEAD - kv.ACTIVATIONS
    per_token = m.kv_bytes_per_token()
    assert int(budget // (per_token * 4096)) == 11
    assert int(budget // (per_token * 8192)) == 5


def test_context_and_concurrency_trade_linearly() -> None:
    """Halving max_model_len doubles max_num_seqs."""
    m = next(x for x in kv.MODELS if "7B-Instruct-AWQ" in x.name)
    budget = kv.VRAM_TOTAL - m.weight_bytes() - kv.ENGINE_OVERHEAD - kv.ACTIVATIONS
    at_4k = int(budget // (m.kv_bytes_per_token() * 4096))
    at_2k = int(budget // (m.kv_bytes_per_token() * 2048))
    assert at_2k == pytest.approx(at_4k * 2, abs=1)


def test_paged_allocation_beats_reservation_for_short_answers() -> None:
    """§5: with a 512-token answer against a 4096 reservation, ~8x more sequences."""
    m = next(x for x in kv.MODELS if "7B-Instruct-AWQ" in x.name)
    budget = kv.VRAM_TOTAL - m.weight_bytes() - kv.ENGINE_OVERHEAD - kv.ACTIVATIONS
    per_token = m.kv_bytes_per_token()
    naive = int(budget // (per_token * 4096))
    paged = int(budget // (per_token * 512))
    assert paged / naive == pytest.approx(8.0, abs=0.5)
