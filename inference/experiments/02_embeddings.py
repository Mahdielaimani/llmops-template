"""Token ids -> vectors. Where the discrete becomes continuous.

uv run python inference/experiments/02_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    D_MODEL,
    MAX_POS,
    VOCAB,
    banner,
    layer_norm,
    load_tokenizer,
    rng,
)

TEXT = "Q2 revenue was EUR 1,234,567.89"


def main() -> None:
    tok = load_tokenizer()
    r = rng()
    ids = tok.encode(TEXT).ids
    n = len(ids)

    banner("1. The embedding table is a lookup, not a computation")
    wte = r.normal(0, 0.02, size=(VOCAB, D_MODEL))  # token embeddings
    wpe = r.normal(0, 0.01, size=(MAX_POS, D_MODEL))  # learned positional embeddings
    print(f"  token embedding table  wte  {wte.shape}  = {wte.size:,} parameters")
    print(f"  position embedding     wpe  {wpe.shape}  = {wpe.size:,} parameters")
    print(f"  wte in fp16: {wte.size * 2 / 1e6:.0f} MB, ~{wte.size / 124e6:.0%} of GPT-2 small")
    print("\n  Embedding is row selection: wte[id]. No matmul, no FLOPs worth counting.")

    banner("2. ids -> hidden states")
    tok_emb = wte[ids]
    pos_emb = wpe[np.arange(n)]
    x = tok_emb + pos_emb
    print(f"  ids            {ids}")
    print(f"  tok_emb        {tok_emb.shape}   (n_tokens, d_model)")
    print(f"  pos_emb        {pos_emb.shape}")
    print(f"  x = sum        {x.shape}")
    print(
        "\n  Position is ADDED, not concatenated. Without wpe the model is a bag of\n"
        "  tokens: attention is permutation-invariant, so 'revenue was 100' and\n"
        "  '100 was revenue' would be identical inputs."
    )

    banner("3. Proving attention is permutation-invariant without positions")
    shuffled = list(reversed(ids))
    no_pos_a = wte[ids].sum(axis=0)
    no_pos_b = wte[shuffled].sum(axis=0)
    with_pos_a = (wte[ids] + pos_emb).sum(axis=0)
    with_pos_b = (wte[shuffled] + pos_emb).sum(axis=0)
    print(f"  without positions: reversed same sum : {np.allclose(no_pos_a, no_pos_b)}")
    print(f"  with positions:    reversed same sum : {np.allclose(with_pos_a, with_pos_b)}")

    banner("4. Hidden state: one vector per token, carried through every layer")
    print(f"  shape at every layer boundary: {x.shape}")
    print(f"  per-token vector norm before LayerNorm: {np.linalg.norm(x[0]):.3f}")
    xn = layer_norm(x)
    print(f"  after LayerNorm:                       {np.linalg.norm(xn[0]):.3f}")
    print(f"  LayerNorm output mean ~0: {xn.mean():.2e}, std ~1: {xn.std():.4f}")
    print(
        "\n  LayerNorm renormalises each token vector independently. Without it the\n"
        "  residual stream grows layer by layer and the softmax in attention\n"
        "  saturates — the practical reason it sits before every sub-block."
    )

    banner("5. Memory, per token")
    print(f"  one hidden state  {D_MODEL} floats = {D_MODEL * 2} B in fp16")
    print(f"  {n} tokens          {n * D_MODEL * 2:,} B")
    print(
        "\n  Activations are transient: they exist for one forward pass and are freed.\n"
        "  The K and V projections are NOT freed during generation — that is the KV\n"
        "  cache, and it is what actually limits concurrency (Phase 5)."
    )


if __name__ == "__main__":
    main()
