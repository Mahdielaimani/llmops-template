"""Text -> tokens -> ids, with the real GPT-2 tokenizer.

    uv run python inference/experiments/01_tokenizer.py

The finance-specific result is the one that matters: currency amounts and dates
tokenize far worse than prose, and every token is a multiplier on prefill FLOPs
(docs/metrics-and-capacity.md §2) and on API cost (§7).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _shared import VOCAB, banner, load_tokenizer

PROSE = "The company reported strong growth in the second quarter."
FINANCE = "Q2 FY2026 revenue was EUR 1,234,567.89, up 12.4% against plan of EUR 1,098,000.00."
IDS_LIKE = "Invoice INV-2026-0004471 for cost centre CC-88213-EMEA, PO 4500931882."


def show(tok, label: str, text: str) -> int:
    enc = tok.encode(text)
    n = len(enc.ids)
    chars = len(text)
    print(f"\n{label}")
    print(f"  text        {text}")
    print(f"  chars {chars:>4}   tokens {n:>4}   chars/token {chars / n:.2f}")
    print(f"  tokens      {enc.tokens}")
    print(f"  ids         {enc.ids}")
    return n


def main() -> None:
    tok = load_tokenizer()
    banner("1. A token is not a word, and not a character")
    print(f"vocabulary size: {tok.get_vocab_size()} (declared constant: {VOCAB})")

    n_prose = show(tok, "PROSE", PROSE)
    n_fin = show(tok, "FINANCIAL", FINANCE)
    n_ids = show(tok, "IDENTIFIERS", IDS_LIKE)

    banner("2. Why this matters for cost and latency")
    print(
        "Same information density, very different token counts per character.\n"
        f"  prose       {len(PROSE) / n_prose:.2f} chars/token\n"
        f"  financial   {len(FINANCE) / n_fin:.2f} chars/token\n"
        f"  identifiers {len(IDS_LIKE) / n_ids:.2f} chars/token"
    )
    print(
        "\nBPE was trained on web text. A number like 1,234,567.89 is split into many\n"
        "pieces because those digit groups are rare; an invoice id is worse. Every\n"
        "extra token costs prefill FLOPs (2*N*P, plus a P^2 attention term) and, on a\n"
        "metered API, money. A retrieval chunk full of tables is not comparable to a\n"
        "chunk of prose at the same character count."
    )

    banner("3. Round-trip is lossless, and whitespace lives inside the token")
    enc = tok.encode(PROSE)
    decoded = tok.decode(enc.ids)
    print(f"  decoded == original : {decoded == PROSE}")
    print(f"  'G' prefix in {sum('Ġ' in t for t in enc.tokens)}/{len(enc.tokens)} tokens")
    print(
        "  GPT-2 BPE encodes a leading space into the token itself, so ' revenue' and\n"
        "  'revenue' are different ids. Concatenating chunks without care changes the\n"
        "  tokenization of the join."
    )

    banner("4. The same word, different ids")
    for variant in ["revenue", " revenue", "Revenue", " Revenue", "REVENUE"]:
        e = tok.encode(variant)
        print(f"  {variant!r:<12} -> ids {e.ids}  tokens {e.tokens}")
    print(
        "\n  Casing and leading space change the id. This is why a prompt template's\n"
        "  whitespace is load-bearing and why prompts are versioned artifacts\n"
        "  (docs/versioning.md, artifact class 2)."
    )

    banner("5. Token budget for a realistic RAG request")
    chunk = FINANCE * 4
    per_chunk = len(tok.encode(chunk).ids)
    system = len(tok.encode("You are a financial analyst. Answer only from the sources.").ids)
    query = len(tok.encode("What was Q2 EMEA revenue versus plan?").ids)
    k = 5
    total = system + query + k * per_chunk
    print(f"  system prompt      {system:>6} tokens")
    print(f"  user query         {query:>6} tokens")
    print(f"  {k} financial chunks  {k * per_chunk:>6} tokens  ({per_chunk} each)")
    print("  ---------------------------------")
    print(f"  prefill P          {total:>6} tokens")
    print(
        "\n  P is the input to every latency and cost formula. Measured here rather than\n"
        "  estimated, because the estimate is wrong for financial text specifically."
    )


if __name__ == "__main__":
    main()
