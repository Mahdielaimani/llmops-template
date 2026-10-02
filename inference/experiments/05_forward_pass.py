"""The whole path, end to end, with shapes and a FLOP count checked against the
formula in docs/metrics-and-capacity.md §2.

    uv run python inference/experiments/05_forward_pass.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    D_FF,
    D_HEAD,
    D_MODEL,
    MAX_POS,
    N_HEADS,
    N_LAYERS,
    VOCAB,
    banner,
    causal_mask,
    gelu,
    init_weights,
    layer_norm,
    load_tokenizer,
    rng,
    softmax,
)

TEXT = "Q2 FY2026 revenue was EUR 1,234,567.89, up 12.4% against plan."


class Block:
    """One transformer block. Pre-norm, as GPT-2 does it."""

    def __init__(self, r: np.random.Generator) -> None:
        self.wq, self.wk, self.wv, self.wo = (init_weights(D_MODEL, D_MODEL, r) for _ in range(4))
        self.w_in = init_weights(D_MODEL, D_FF, r)
        self.w_out = init_weights(D_FF, D_MODEL, r)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        h = layer_norm(x)
        q = (h @ self.wq).reshape(n, N_HEADS, D_HEAD).transpose(1, 0, 2)
        k = (h @ self.wk).reshape(n, N_HEADS, D_HEAD).transpose(1, 0, 2)
        v = (h @ self.wv).reshape(n, N_HEADS, D_HEAD).transpose(1, 0, 2)
        w = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(D_HEAD) + causal_mask(n))
        attn = (w @ v).transpose(1, 0, 2).reshape(n, D_MODEL) @ self.wo
        x = x + attn  # residual
        x = x + gelu(layer_norm(x) @ self.w_in) @ self.w_out
        return x


def params() -> dict[str, int]:
    per_block = 4 * D_MODEL * D_MODEL + 2 * D_MODEL * D_FF
    return {
        "embeddings": VOCAB * D_MODEL + MAX_POS * D_MODEL,
        "blocks": N_LAYERS * per_block,
        "per_block": per_block,
    }


def flops_prefill(p: int, n_params: int) -> dict[str, float]:
    dense = 2.0 * n_params * p
    attn = 2.0 * N_LAYERS * p * p * (N_HEADS * D_HEAD)
    return {"dense": dense, "attention": attn, "total": dense + attn}


def main() -> None:
    tok = load_tokenizer()
    r = rng()
    ids = tok.encode(TEXT).ids
    n = len(ids)

    banner("1. Parameter budget, from the geometry")
    pc = params()
    total = pc["embeddings"] + pc["blocks"]
    print(f"  embeddings (wte + wpe)        {pc['embeddings']:>12,}")
    print(f"  {N_LAYERS} blocks x {pc['per_block']:,}      {pc['blocks']:>12,}")
    print(f"  total                         {total:>12,}  (~{total / 1e6:.0f}M)")
    print(f"  fp16 weights                  {total * 2 / 1e6:>12.0f} MB")
    print(f"  int4 weights                  {total * 0.5 / 1e6:>12.0f} MB")
    print(
        "\n  GPT-2 small is ~124M; the published figure ties the embedding matrix to\n"
        "  the output layer, which this toy counts once."
    )

    banner("2. Full forward pass, shapes at every step")
    wte = r.normal(0, 0.02, size=(VOCAB, D_MODEL))
    wpe = r.normal(0, 0.01, size=(MAX_POS, D_MODEL))
    blocks = [Block(r) for _ in range(2)]  # 2, not 12: identical shapes, less RAM

    print(f"  text                '{TEXT}'")
    print(f"  ids                 ({n},)")
    x = wte[ids] + wpe[np.arange(n)]
    print(f"  + embeddings        {x.shape}")
    for i, blk in enumerate(blocks):
        x = blk(x)
        print(f"  after block {i}       {x.shape}   <- shape is invariant, blocks stack")
    x = layer_norm(x)
    print(f"  final LayerNorm     {x.shape}")
    logits = x @ wte.T  # weight tying: reuse the embedding matrix
    print(f"  @ wte.T (tied)      {logits.shape}   (n_tokens, vocab)")
    print(f"  take last position   ({VOCAB},)   <- only this row predicts the next token")
    next_probs = softmax(logits[-1])
    print(f"  softmax -> p         sums to {next_probs.sum():.6f}")

    banner("3. Prefill computes every position; decode needs only the last")
    print(f"  prefill produced logits for all {n} positions: {logits.shape}")
    print(f"  {n - 1} of those rows are discarded at inference time.")
    print(
        "\n  Training needs all of them (next-token loss at every position). Inference\n"
        "  needs one. That waste is why prefill is a single big batched matmul and\n"
        "  decode is a thin one — and why the two phases have completely different\n"
        "  performance profiles (Phase 5)."
    )

    banner("4. FLOPs: formula vs the geometry")
    for p in (n, 512, 3000):
        f = flops_prefill(p, total)
        share = f["attention"] / f["total"]
        print(
            f"  P={p:>5}  dense {f['dense'] / 1e9:>9.2f} GFLOP"
            f"   attention {f['attention'] / 1e9:>8.2f} GFLOP"
            f"   attention share {share:>5.1%}"
        )
    print(
        "\n  The attention term is quadratic in P while the dense term is linear, so\n"
        "  its share grows with context. On a 7B model the dense term dominates far\n"
        "  longer, which is why 3k context is still prefill-bound there\n"
        "  (docs/metrics-and-capacity.md §2)."
    )

    banner("5. Measured wall time (numpy, CPU, 2 blocks)")

    def run(p: int) -> None:
        h = wte[np.zeros(p, dtype=int)] + wpe[np.arange(p)]
        for blk in blocks:
            h = blk(h)
        _ = layer_norm(h) @ wte.T

    run(64)  # warm BLAS: the first call pays thread-pool setup and is not comparable

    print(f"  {'P':>6} {'best of 3':>12} {'ms/token':>10} {'normalised':>11}")
    per_token_at_128 = None
    for p in (128, 256, 512, 1024):
        timings = []
        for _ in range(3):
            t0 = time.perf_counter()
            run(p)
            timings.append((time.perf_counter() - t0) * 1000)
        best = min(timings)
        per_token = best / p
        per_token_at_128 = per_token_at_128 or per_token
        print(f"  {p:>6} {best:>9.1f} ms {per_token:>9.3f} {per_token / per_token_at_128:>10.2f}x")

    print()
    print("  Roughly LINEAR in P, and that is the correct result. An earlier version")
    print("  of this script read three unwarmed samples and claimed superlinear")
    print("  scaling; the data did not support it. Section 4 explains why it cannot:")
    print("  below P=512 the attention term is under 4% of total FLOPs, so the")
    print("  quadratic component is invisible. It only dominates at context lengths")
    print("  this toy cannot reach.")
    print()
    print("  THIS IS A SIMPLIFICATION: numpy on CPU, 2 blocks, no fused kernels, no")
    print("  KV cache. The scaling shape is real; the absolute numbers are not.")
    print("  Phase 6 measures vLLM on the GPU.")


if __name__ == "__main__":
    main()
