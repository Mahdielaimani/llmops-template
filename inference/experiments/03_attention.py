"""Attention, one matmul at a time. Q, K, V, the mask, and why K/V are the ones
worth caching.

    uv run python inference/experiments/03_attention.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (
    D_HEAD,
    D_MODEL,
    N_HEADS,
    banner,
    causal_mask,
    init_weights,
    kv_bytes_per_token,
    layer_norm,
    load_tokenizer,
    rng,
    softmax,
)

TEXT = "Q2 revenue was EUR 1,234,567.89 against plan"


def main() -> None:
    tok = load_tokenizer()
    r = rng()
    enc = tok.encode(TEXT)
    tokens, ids = enc.tokens, enc.ids
    n = len(ids)

    x = layer_norm(r.normal(0, 1, size=(n, D_MODEL)))

    banner("1. Three projections of the same input")
    wq, wk, wv = (init_weights(D_MODEL, D_MODEL, r) for _ in range(3))
    q, k, v = x @ wq, x @ wk, x @ wv
    print(f"  x  {x.shape}   @ Wq {wq.shape} -> Q {q.shape}")
    print(f"  x  {x.shape}   @ Wk {wk.shape} -> K {k.shape}")
    print(f"  x  {x.shape}   @ Wv {wv.shape} -> V {v.shape}")
    print(
        "\n  Q: 'what am I looking for'   (belongs to the CURRENT token)\n"
        "  K: 'what can I be found by'   (belongs to EVERY PAST token)\n"
        "  V: 'what I contribute'        (belongs to EVERY PAST token)\n"
        "\n  That asymmetry is the whole reason for the KV cache: Q is recomputed for\n"
        "  the one new token each step, while K and V for all previous tokens are\n"
        "  identical to last step and can be reused."
    )

    banner("2. Split into heads")
    qh = q.reshape(n, N_HEADS, D_HEAD).transpose(1, 0, 2)
    kh = k.reshape(n, N_HEADS, D_HEAD).transpose(1, 0, 2)
    vh = v.reshape(n, N_HEADS, D_HEAD).transpose(1, 0, 2)
    print(f"  {D_MODEL} dims -> {N_HEADS} heads x {D_HEAD} dims")
    print(f"  Q per head {qh.shape}  (heads, n_tokens, d_head)")
    print("\n  Heads are a reshape, not extra parameters. Each attends independently.")

    banner("3. Scores, scaling, and why the sqrt matters")
    scores = qh @ kh.transpose(0, 2, 1)
    scaled = scores / np.sqrt(D_HEAD)
    print(f"  scores {scores.shape}  (heads, n_query, n_key)  = n^2 per head")
    print(f"  raw   std {scores.std():.3f}  max |.| {np.abs(scores).max():.3f}")
    print(f"  /sqrt({D_HEAD}) std {scaled.std():.3f}  max |.| {np.abs(scaled).max():.3f}")
    sharp_raw = softmax(scores[0, -1]).max()
    sharp_scaled = softmax(scaled[0, -1]).max()
    print(f"\n  softmax peak without scaling: {sharp_raw:.4f}")
    print(f"  softmax peak with    scaling: {sharp_scaled:.4f}")
    print(
        "  Unscaled, the dot product grows with d_head, softmax saturates toward\n"
        "  one-hot, and the model can only copy one token. The sqrt keeps it soft."
    )
    print(f"\n  The n^2 term: {n} tokens -> {n * n} scores per head, {n * n * N_HEADS:,} total.")
    print("  Double the context and this quadruples — the P^2 term in prefill cost.")

    banner("4. The causal mask")
    mask = causal_mask(n)
    masked = scaled + mask
    weights = softmax(masked)
    print(f"  mask[0, :5] = {mask[0, :5]}   (row 0 can only see itself)")
    print(f"  mask[-1, :5] = {mask[-1, :5]}  (last row sees everything before it)")
    print(f"\n  each row sums to 1.0 : {np.allclose(weights.sum(axis=-1), 1.0)}")
    print(
        f"  upper triangle is exactly 0 : {np.allclose(weights[0][np.triu_indices(n, k=1)], 0.0)}"
    )
    print(
        "\n  Without the mask the model sees the future during training and learns\n"
        "  nothing useful. At inference the mask is also what makes the cache valid:\n"
        "  a past token's attention output never changes when a new token arrives."
    )

    banner("5. A real attention row (head 0, last token)")
    row = weights[0, -1]
    print(f"  query token: {tokens[-1]!r}")
    order = np.argsort(row)[::-1]
    for i in order[:6]:
        bar = "#" * int(row[i] * 60)
        print(f"    {tokens[i]!r:<14} {row[i]:.4f} {bar}")
    print(
        "\n  Weights are random here, so the pattern is noise — THIS IS A"
        " SIMPLIFICATION.\n  The shape of the computation is what is real."
    )

    banner("6. Output: a weighted sum of V")
    out_h = weights @ vh
    out = out_h.transpose(1, 0, 2).reshape(n, D_MODEL)
    wo = init_weights(D_MODEL, D_MODEL, r)
    out = out @ wo
    print(f"  weights {weights.shape} @ V {vh.shape} -> {out_h.shape}")
    print(f"  merge heads -> {out.transpose().shape[::-1]}  then @ Wo -> {out.shape}")
    print("  Same shape in, same shape out: that is what lets blocks stack.")

    banner("7. What caching K and V actually saves")
    per_token = kv_bytes_per_token()
    print(f"  KV per token per sequence = 2 * {12} layers * {N_HEADS} heads * {D_HEAD} dim * 2 B")
    print(f"                            = {per_token:,} B = {per_token / 1024:.0f} KiB")
    for ctx in (512, 2048, 4096):
        print(f"  at {ctx:>5} tokens: {per_token * ctx / 1e6:>7.1f} MB per sequence")
    print(
        "\n  Without a cache, generating token t recomputes K and V for all t-1\n"
        "  previous tokens: O(t) work per step, O(t^2) for the sequence. With it,\n"
        "  O(1) projections per step and the cost moves from compute to memory.\n"
        "  Measured in Phase 5; the VRAM consequence is metrics-and-capacity.md §5."
    )


if __name__ == "__main__":
    main()
