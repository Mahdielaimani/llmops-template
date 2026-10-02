"""The KV cache as a VRAM budget: what fits, what does not, and why PagedAttention
exists.

    uv run python inference/experiments/07_kv_cache.py

Everything here is arithmetic over the real geometry of real models, applied to
the 8 GB budget in docs/environment.md. No GPU required to compute it; Phase 6
checks it against nvidia-smi.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _shared import banner

GB = 1e9
VRAM_TOTAL = 8188 / 1024 * GB  # RTX 4070 Laptop, from nvidia-smi: 8188 MiB


@dataclass(frozen=True)
class Model:
    name: str
    params_b: float
    layers: int
    q_heads: int
    kv_heads: int
    d_head: int
    quant_bytes: float  # bytes per parameter as served
    max_ctx: int

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        return 2 * self.layers * self.kv_heads * self.d_head * dtype_bytes

    def weight_bytes(self, quant_overhead: float = 0.10) -> float:
        return self.params_b * 1e9 * self.quant_bytes * (1 + quant_overhead)

    @property
    def gqa_ratio(self) -> float:
        return self.q_heads / self.kv_heads


MODELS = [
    Model("GPT-2 small (fp16)", 0.124, 12, 12, 12, 64, 2.0, 1024),
    Model("Qwen2.5-1.5B-Instruct (fp16)", 1.54, 28, 12, 2, 128, 2.0, 32768),
    Model("Qwen2.5-7B-Instruct-AWQ (int4)", 7.6, 28, 28, 4, 128, 0.5, 32768),
    Model("Llama-3.1-8B-Instruct-AWQ (int4)", 8.0, 32, 32, 8, 128, 0.5, 131072),
    Model("Qwen2.5-7B-Instruct (fp16)", 7.6, 28, 28, 4, 128, 2.0, 32768),
]

ENGINE_OVERHEAD = 0.9 * GB  # CUDA context + engine + graphs, measured in Phase 6
ACTIVATIONS = 0.3 * GB


def main() -> None:
    banner("1. KV bytes per token, per model")
    print("  2 (K and V) * layers * kv_heads * d_head * 2 B (fp16)")
    print()
    print(f"  {'model':<34}{'layers':>7}{'q/kv heads':>12}{'GQA':>6}{'KiB/token':>11}")
    for m in MODELS[:4]:
        print(
            f"  {m.name:<34}{m.layers:>7}{f'{m.q_heads}/{m.kv_heads}':>12}"
            f"{m.gqa_ratio:>5.0f}x{m.kv_bytes_per_token() / 1024:>11.0f}"
        )
    print()
    print("  GQA is the lever. Qwen2.5-7B has 28 query heads but only 4 KV heads, so")
    print("  its cache is 7x smaller than multi-head attention would need. Without")
    print("  GQA the same model would want 392 KiB/token and nothing would fit.")

    banner("2. Does the model even fit, before any cache?")
    print(
        f"  VRAM total {VRAM_TOTAL / GB:.2f} GB   engine overhead "
        f"{ENGINE_OVERHEAD / GB:.1f} GB   activations {ACTIVATIONS / GB:.1f} GB"
    )
    print()
    print(f"  {'model':<34}{'weights GB':>12}{'KV budget GB':>14}{'verdict':>10}")
    for m in MODELS:
        w = m.weight_bytes()
        budget = VRAM_TOTAL - w - ENGINE_OVERHEAD - ACTIVATIONS
        verdict = "OK" if budget > 0.2 * GB else ("TIGHT" if budget > 0 else "NO FIT")
        print(f"  {m.name:<34}{w / GB:>12.2f}{budget / GB:>14.2f}{verdict:>10}")
    print()
    print("  Qwen2.5-7B at fp16 does not fit at all: the weights alone exceed the card.")
    print("  Quantization is not an optimisation here, it is the entry ticket.")

    banner("3. Concurrency ceiling: context x sequences, for Qwen2.5-7B-AWQ")
    m = MODELS[2]
    budget = VRAM_TOTAL - m.weight_bytes() - ENGINE_OVERHEAD - ACTIVATIONS
    per_token = m.kv_bytes_per_token()
    print(
        f"  weights {m.weight_bytes() / GB:.2f} GB, KV budget {budget / GB:.2f} GB, "
        f"{per_token / 1024:.0f} KiB/token"
    )
    print()
    contexts = (1024, 2048, 4096, 8192, 16384, 32768)
    print(f"  {'context':>9}{'MB / sequence':>15}{'max sequences':>15}{'note':>22}")
    for ctx in contexts:
        per_seq = per_token * ctx
        max_seq = int(budget // per_seq)
        note = "single request only" if max_seq <= 1 else ""
        if max_seq == 0:
            note = "DOES NOT FIT"
        print(f"  {ctx:>9}{per_seq / 1e6:>15.0f}{max_seq:>15}{note:>22}")
    print()
    print("  This single table is the platform's capacity plan. docs/business-")
    print("  requirements.md assumes 5 req/s bursts at ~6 s each, so ~30 requests in")
    print("  flight. At 4k context the card supports ~11. The queue and the")
    print("  concurrency cap in Phase 21 are arithmetic, not taste.")

    banner("4. Context and concurrency trade against each other, linearly")
    print("  Same KV budget, spent differently:")
    print()
    print(f"  {'':>12}" + "".join(f"{c:>9}" for c in (1024, 2048, 4096, 8192)))
    for n_seq in (1, 2, 4, 8, 16, 32):
        row = f"  {n_seq:>3} seqs   "
        for ctx in (1024, 2048, 4096, 8192):
            used = per_token * ctx * n_seq
            row += f"{'fits' if used <= budget else '-':>9}"
        print(row)
    print()
    print("  Reading it the other way: halving max_model_len doubles max_num_seqs.")
    print("  Those are the two vLLM flags that matter most on a small card, and they")
    print("  are the same knob.")

    banner("5. Fragmentation, and why PagedAttention exists")
    print("  A naive engine reserves max_model_len for every sequence at admission,")
    print("  because it cannot know how long the answer will be.")
    print()
    reserved_ctx = 4096
    print(f"  {'actual tokens':>15}{'reserved':>11}{'used MB':>10}{'wasted MB':>12}{'waste':>8}")
    for actual in (128, 512, 1024, 2048, 4096):
        reserved_b = per_token * reserved_ctx
        used_b = per_token * actual
        waste = (reserved_b - used_b) / reserved_b
        print(
            f"  {actual:>15}{reserved_ctx:>11}{used_b / 1e6:>10.0f}"
            f"{(reserved_b - used_b) / 1e6:>12.0f}{waste:>7.0%}"
        )
    print()
    typical = 512
    naive_seqs = int(budget // (per_token * reserved_ctx))
    paged_seqs = int(budget // (per_token * typical))
    print(f"  With a {typical}-token typical answer and a {reserved_ctx} reservation:")
    print(f"    naive pre-allocation  -> {naive_seqs:>3} concurrent sequences")
    print(
        f"    paged, block-level    -> {paged_seqs:>3} concurrent sequences  "
        f"({paged_seqs / max(naive_seqs, 1):.0f}x)"
    )
    print()
    print("  PagedAttention does not shrink KV bytes per token - the formula in §1 is")
    print("  unchanged. It allocates fixed-size blocks on demand, so a sequence")
    print("  occupies what it uses rather than what it might use. On these numbers")
    print(f"  that is the difference between {naive_seqs} and {paged_seqs} concurrent")
    print("  sequences from the same VRAM. It also makes sharing possible: a common")
    print("  system prompt can be one set of blocks referenced by many sequences,")
    print("  which is what prefix caching is.")

    banner("6. What happens when the budget is exceeded")
    print("  Three behaviours, in order of how a server should prefer them:")
    print()
    print("  1. QUEUE      admission control holds the request. Latency rises,")
    print("                nothing fails. This is the correct response, and it is why")
    print("                queue depth is the saturation signal, not GPU utilisation.")
    print("  2. PREEMPT    the engine evicts a running sequence's blocks and")
    print("                recomputes or swaps them later. vLLM counts this as")
    print("                vllm:request_num_preemptions - a rising count is the")
    print("                earliest honest warning that the card is over-committed.")
    print("  3. OOM        CUDA out of memory. The process usually dies, taking every")
    print("                in-flight request with it. Recovery is a container restart")
    print("                plus a model reload, so a single over-admission becomes")
    print("                tens of seconds of total unavailability.")
    print()
    print("  The operational conclusion: cap concurrency below the arithmetic ceiling")
    print("  and alert on preemptions, because the alternative to shedding one")
    print("  request is losing all of them. Phase 21 builds the cap, Phase 37")
    print("  Incident 5 reproduces the OOM deliberately.")

    banner("7. Cross-checks against the rest of the docs")
    q = MODELS[2]
    print(
        f"  Qwen2.5-7B-AWQ KV/token  {q.kv_bytes_per_token() / 1024:.0f} KiB"
        f"   (metrics-and-capacity.md §5 states 56 KiB)"
    )
    print(
        f"  max sequences @ 4k ctx   {int(budget // (per_token * 4096))}"
        f"               (§5 states ~11)"
    )
    print(
        f"  max sequences @ 8k ctx   {int(budget // (per_token * 8192))}"
        f"               (§5 states ~5)"
    )
    print(
        "  GPT-2 small KV/token     "
        f"{MODELS[0].kv_bytes_per_token() / 1024:.0f} KiB"
        f"   (inference.md §3 states 36 KiB)"
    )


if __name__ == "__main__":
    main()
