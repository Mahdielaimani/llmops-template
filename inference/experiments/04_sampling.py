"""Logits -> a token. Greedy, temperature, top-k, top-p, and what each does to
reproducibility.

    uv run python inference/experiments/04_sampling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _shared import D_MODEL, VOCAB, banner, init_weights, load_tokenizer, rng, softmax


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(logits, -np.inf)
    idx = np.argpartition(logits, -k)[-k:]
    out[idx] = logits[idx]
    return out


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: keep the smallest set of tokens whose mass reaches p.
    Unlike top-k the cut adapts — a confident step keeps 1 token, an uncertain
    step keeps hundreds."""
    order = np.argsort(logits)[::-1]
    cumulative = np.cumsum(softmax(logits)[order])
    cut = int(np.searchsorted(cumulative, p) + 1)
    out = np.full_like(logits, -np.inf)
    out[order[:cut]] = logits[order[:cut]]
    return out


def main() -> None:
    tok = load_tokenizer()
    r = rng()

    banner("1. The unembedding: hidden state -> one score per vocabulary entry")
    h = r.normal(0, 1, size=D_MODEL)
    w_unembed = init_weights(D_MODEL, VOCAB, r)
    # Xavier-initialised random weights give logits with std ~0.18, which is almost
    # uniform over 50k tokens — every temperature then looks identical and the demo
    # below would teach the opposite of the truth. A trained GPT-2 produces a final
    # logit std nearer 4. Rescaled to that, and labelled rather than hidden.
    raw = h @ w_unembed
    logits = raw / raw.std() * 4.0
    print(f"  hidden {h.shape} @ W_unembed {w_unembed.shape} -> logits {logits.shape}")
    print(f"  that is {VOCAB:,} scores for ONE position — the widest matmul in the model")
    print(f"  raw (random weights)  std {raw.std():.3f}  -> near-uniform, unusable for the demo")
    print(
        f"  rescaled to realistic std {logits.std():.3f}, range "
        f"[{logits.min():.2f}, {logits.max():.2f}]"
    )
    print(
        "\n  Logits are unbounded real numbers, not probabilities. Only their\n"
        "  differences matter: adding a constant to all of them changes nothing."
    )
    shifted = softmax(logits + 1000.0)
    print(f"  softmax(logits) == softmax(logits + 1000): {np.allclose(softmax(logits), shifted)}")

    banner("2. Temperature reshapes the distribution before sampling")
    probs = softmax(logits)
    print(f"  {'T':>6} {'max p':>9} {'top-10 mass':>12} {'entropy':>9}  effective choices")
    for t in (0.01, 0.2, 0.7, 1.0, 1.5, 3.0):
        p = softmax(logits / t)
        top10 = np.sort(p)[::-1][:10].sum()
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        print(
            f"  {t:>6.2f} {p.max():>9.5f} {top10:>12.5f} {entropy:>9.3f}  {np.exp(entropy):>8.0f}"
        )
    print(
        "\n  T -> 0 approaches greedy (one choice). T = 1 is the raw distribution.\n"
        "  T > 1 flattens it and 'effective choices' explodes. Temperature is not a\n"
        "  creativity dial, it is the sharpness of a probability distribution."
    )

    banner("3. Greedy is deterministic; sampling is not")
    greedy = [int(np.argmax(logits)) for _ in range(5)]
    print(f"  greedy, 5 runs : {greedy}  -> identical: {len(set(greedy)) == 1}")
    sampled = [int(np.random.default_rng(s).choice(VOCAB, p=probs)) for s in range(5)]
    print(f"  sampled, 5 seeds: {sampled}  -> identical: {len(set(sampled)) == 1}")
    same_seed = [int(np.random.default_rng(42).choice(VOCAB, p=probs)) for _ in range(3)]
    print(f"  sampled, seed 42 x3: {same_seed} -> identical: {len(set(same_seed)) == 1}")
    print(
        "\n  Reproducibility needs the seed AND identical logits. Logits move with\n"
        "  batch composition, kernel version and hardware, so even temperature=0 on a\n"
        "  real server is not bit-reproducible across runs. This is why Phase 11\n"
        "  evaluates statistically over a dataset instead of asserting a fixture\n"
        "  (contrast docs/classical-ml-vs-llm-serving.md §4)."
    )

    banner("4. Truncation: top-k is fixed, top-p adapts")
    for label, filtered in (
        ("top-k=50", top_k_filter(logits, 50)),
        ("top-p=0.9", top_p_filter(logits, 0.9)),
    ):
        kept = int(np.isfinite(filtered).sum())
        print(f"  {label:<10} keeps {kept:>6} / {VOCAB:,} tokens")

    print("\n  Same filters on a DELIBERATELY CONFIDENT distribution:")
    confident = logits.copy()
    confident[1234] += 15.0
    for label, filtered in (
        ("top-k=50", top_k_filter(confident, 50)),
        ("top-p=0.9", top_p_filter(confident, 0.9)),
    ):
        kept = int(np.isfinite(filtered).sum())
        print(f"  {label:<10} keeps {kept:>6} tokens")
    print(
        "\n  top-k keeps 50 either way, including 49 tokens the model all but ruled\n"
        "  out. top-p collapses to the few that matter. For grounded financial\n"
        "  answers the adaptive cut is the safer default."
    )

    banner("5. What a 'token' decodes to")
    top = np.argsort(logits)[::-1][:8]
    for i in top:
        print(f"  id {int(i):>6}  p={probs[i]:.6f}  {tok.decode([int(i)])!r}")
    print(
        "\n  Random weights, so these are arbitrary ids — THIS IS A SIMPLIFICATION.\n"
        "  Real sampling repeats this per generated token: one unembedding matmul\n"
        "  over 50k vocabulary entries per step, which is why output length drives\n"
        "  decode cost linearly (docs/metrics-and-capacity.md §3)."
    )


if __name__ == "__main__":
    main()
