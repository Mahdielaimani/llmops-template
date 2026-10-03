"""Prefill vs decode, and the KV cache earning its keep.

    uv run python inference/experiments/06_prefill_decode.py

Generates the same tokens twice — once recomputing everything each step, once
reusing cached K and V. The outputs must be identical; the work is not. That
equality is the proof that caching is not an approximation.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    D_FF,
    D_HEAD,
    D_MODEL,
    MAX_POS,
    N_HEADS,
    VOCAB,
    banner,
    causal_mask,
    gelu,
    init_weights,
    kv_bytes_per_token,
    layer_norm,
    load_tokenizer,
    rng,
    softmax,
)

N_LAYERS_TOY = 2  # same shapes as 12, a sixth of the RAM
PROMPT = "Q2 FY2026 revenue was EUR 1,234,567.89 against a plan of"
N_NEW = 80  # long enough that the sequence length spans 5x, or O(t) growth hides


@dataclass
class LayerCache:
    k: np.ndarray | None = None
    v: np.ndarray | None = None

    def append(self, k_new: np.ndarray, v_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """k_new/v_new are (heads, n_new, d_head). Concatenating along the token axis
        is the cache 'growing' — in a real engine this is a block allocation."""
        self.k = k_new if self.k is None else np.concatenate([self.k, k_new], axis=1)
        self.v = v_new if self.v is None else np.concatenate([self.v, v_new], axis=1)
        return self.k, self.v

    @property
    def n_tokens(self) -> int:
        return 0 if self.k is None else int(self.k.shape[1])

    @property
    def nbytes(self) -> int:
        return 0 if self.k is None else int(self.k.nbytes + self.v.nbytes)


@dataclass
class Weights:
    wte: np.ndarray
    wpe: np.ndarray
    blocks: list[dict[str, np.ndarray]] = field(default_factory=list)


def build(r: np.random.Generator, vocab: int = VOCAB) -> Weights:
    """`vocab` is overridable so tests can allocate a small table: the full
    50257 x 768 float64 embedding matrix is 294 MiB, and this laptop has 15 GB
    shared with Docker (docs/environment.md). The shapes under test do not
    depend on vocabulary size.
    """
    w = Weights(
        wte=r.normal(0, 0.02, size=(vocab, D_MODEL)),
        wpe=r.normal(0, 0.01, size=(MAX_POS, D_MODEL)),
    )
    for _ in range(N_LAYERS_TOY):
        w.blocks.append(
            {
                "wq": init_weights(D_MODEL, D_MODEL, r),
                "wk": init_weights(D_MODEL, D_MODEL, r),
                "wv": init_weights(D_MODEL, D_MODEL, r),
                "wo": init_weights(D_MODEL, D_MODEL, r),
                "w_in": init_weights(D_MODEL, D_FF, r),
                "w_out": init_weights(D_FF, D_MODEL, r),
            }
        )
    return w


def _heads(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    return x.reshape(n, N_HEADS, D_HEAD).transpose(1, 0, 2)


def block(
    x: np.ndarray, p: dict[str, np.ndarray], cache: LayerCache | None, offset: int
) -> np.ndarray:
    """One block. `cache` None means recompute-everything mode. `offset` is how many
    tokens precede x, needed for the mask when x is a single new token."""
    n = x.shape[0]
    h = layer_norm(x)
    q = _heads(h @ p["wq"])
    k_new, v_new = _heads(h @ p["wk"]), _heads(h @ p["wv"])

    if cache is None:
        k, v = k_new, v_new
        mask = causal_mask(n)
    else:
        k, v = cache.append(k_new, v_new)
        total = k.shape[1]
        if n == 1:
            # The single new token may attend to everything cached: no masking needed,
            # which is exactly why decode is a thin matmul and not a triangular one.
            mask = np.zeros((1, total))
        else:
            mask = np.full((n, total), -np.inf)
            full = causal_mask(n)
            mask[:, offset : offset + n] = full
            mask[:, :offset] = 0.0

    w = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(D_HEAD) + mask)
    attn = (w @ v).transpose(1, 0, 2).reshape(n, D_MODEL) @ p["wo"]
    x = x + attn
    return x + gelu(layer_norm(x) @ p["w_in"]) @ p["w_out"]


def forward(
    w: Weights, ids: list[int], caches: list[LayerCache] | None, offset: int = 0
) -> np.ndarray:
    pos = np.arange(offset, offset + len(ids))
    x = w.wte[ids] + w.wpe[pos]
    for i, p in enumerate(w.blocks):
        x = block(x, p, None if caches is None else caches[i], offset)
    return layer_norm(x) @ w.wte.T


def generate_no_cache(w: Weights, prompt: list[int], n_new: int) -> tuple[list[int], list[float]]:
    """Recompute the whole sequence every step. O(t) work at step t, O(n^2) overall."""
    ids = list(prompt)
    steps: list[float] = []
    for _ in range(n_new):
        t0 = time.perf_counter()
        logits = forward(w, ids, None)
        ids.append(int(np.argmax(logits[-1])))
        steps.append((time.perf_counter() - t0) * 1000)
    return ids[len(prompt) :], steps


def generate_with_cache(
    w: Weights, prompt: list[int], n_new: int
) -> tuple[list[int], float, list[float], int]:
    """Prefill once, then one token at a time against the cache."""
    caches = [LayerCache() for _ in w.blocks]

    t0 = time.perf_counter()
    logits = forward(w, prompt, caches, offset=0)
    prefill_ms = (time.perf_counter() - t0) * 1000

    out = [int(np.argmax(logits[-1]))]
    steps: list[float] = []
    for _ in range(n_new - 1):
        t0 = time.perf_counter()
        logits = forward(w, [out[-1]], caches, offset=caches[0].n_tokens)
        out.append(int(np.argmax(logits[-1])))
        steps.append((time.perf_counter() - t0) * 1000)

    return out, prefill_ms, steps, sum(c.nbytes for c in caches)


def main() -> None:
    tok = load_tokenizer()
    w = build(rng())
    prompt = tok.encode(PROMPT).ids
    p = len(prompt)

    banner("1. The two phases")
    print(f"  prompt      {PROMPT!r}")
    print(f"  P           {p} tokens")
    print(f"  generating  {N_NEW} tokens")
    print()
    print("  PREFILL  one forward pass over all P tokens at once.")
    print("           Compute-bound: a big matmul. Ends at the first token -> TTFT.")
    print("  DECODE   N passes over ONE token each, attending to everything before it.")
    print("           Memory-bound: reads all the weights to produce one token.")

    banner("2. Same tokens, with and without a cache")
    generate_no_cache(w, prompt, 2)  # warm BLAS: the first matmul pays thread setup
    naive_ids, naive_steps = generate_no_cache(w, prompt, N_NEW)
    cached_ids, prefill_ms, decode_steps, kv_bytes = generate_with_cache(w, prompt, N_NEW)

    print(f"  no cache    {naive_ids[:8]} ...")
    print(f"  with cache  {cached_ids[:8]} ...")
    print(f"  IDENTICAL   {naive_ids == cached_ids}")
    print()
    print("  Greedy decoding is deterministic, so identical output proves the cache is")
    print("  an exact optimisation, not an approximation. If this line ever prints")
    print("  False the mask or the position offset is wrong.")
    print(f"\n  decoded: {tok.decode(cached_ids)!r}")
    print("  (random weights, so the text is noise - THIS IS A SIMPLIFICATION)")

    banner("3. Cost")
    naive_total = sum(naive_steps)
    cached_total = prefill_ms + sum(decode_steps)
    print(f"  {'':<22}{'total ms':>10}{'per token':>12}")
    print(f"  {'no cache':<22}{naive_total:>10.1f}{naive_total / N_NEW:>12.2f}")
    print(f"  {'prefill + cache':<22}{cached_total:>10.1f}{cached_total / N_NEW:>12.2f}")
    print(f"  {'speedup':<22}{naive_total / cached_total:>9.1f}x")
    print()
    print(f"  prefill            {prefill_ms:>8.1f} ms   <- this is TTFT")
    print(f"  decode, mean step  {np.mean(decode_steps):>8.2f} ms   <- this is TPOT / ITL")
    print(f"  decode, first/last {decode_steps[0]:>8.2f} / {decode_steps[-1]:.2f} ms")
    print(f"  output throughput  {1000 / np.mean(decode_steps):>8.1f} tok/s")

    banner("4. Why the naive version degrades and the cached one does not")
    print(f"  {'step':>6}{'no cache ms':>14}{'cached ms':>12}{'seq len':>10}")
    for i in (0, 5, 11, 17, N_NEW - 2):
        cached = decode_steps[i] if i < len(decode_steps) else float("nan")
        print(f"  {i:>6}{naive_steps[i]:>14.2f}{cached:>12.2f}{p + i:>10}")
    first_half = float(np.mean(naive_steps[: N_NEW // 2]))
    second_half = float(np.mean(naive_steps[N_NEW // 2 :]))
    print(
        f"\n  no cache: first half {first_half:.2f} ms -> second half {second_half:.2f} ms "
        f"({second_half / first_half:.2f}x)"
    )
    c_first = float(np.mean(decode_steps[: len(decode_steps) // 2]))
    c_second = float(np.mean(decode_steps[len(decode_steps) // 2 :]))
    print(
        f"  cached:   first half {c_first:.2f} ms -> second half {c_second:.2f} ms "
        f"({c_second / c_first:.2f}x)"
    )
    print()
    print("  Without a cache, step t re-projects K and V for all t tokens, so cost")
    print("  grows with t and the sequence total is O(n^2). With a cache, step t")
    print("  projects one token: flat in t, O(n) overall. The cached curve is not")
    print("  perfectly flat because attention still reads a cache that keeps growing.")

    banner("5. What the cache costs in memory")
    per_token_toy = kv_bytes_per_token(n_layers=N_LAYERS_TOY, dtype_bytes=8)
    print(f"  measured  {kv_bytes:>12,} B for {p + N_NEW - 1} tokens x {N_LAYERS_TOY} layers")
    print(
        f"  formula   {per_token_toy * (p + N_NEW - 1):>12,} B   "
        f"(2 * {N_LAYERS_TOY} * {N_HEADS} * {D_HEAD} * 8 B float64)"
    )
    print(f"  match     {kv_bytes == per_token_toy * (p + N_NEW - 1)}")
    print()
    print("  numpy is float64; a real engine is fp16, and 12 layers not 2. For")
    print("  GPT-2 small that is 36 KiB per token per sequence, and that number")
    print("  multiplied by context and concurrency is the concurrency ceiling.")
    print("  Experiment 07 turns it into a VRAM budget.")


if __name__ == "__main__":
    main()
