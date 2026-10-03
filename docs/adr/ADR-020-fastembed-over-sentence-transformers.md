# ADR-020: fastembed (ONNX) rather than sentence-transformers for embeddings

**Status:** Accepted
**Date:** 2026-10-03
**Phase:** 8

## Context
Phase 8 needs a dense embedding model for retrieval, and Phase 9 needs a
cross-encoder reranker. The default choice in this ecosystem is
`sentence-transformers`, which depends on PyTorch.

The binding constraint on this machine is **disk, not compute**:
`docs/environment.md` records 97% used with 29 GB free, and §7.3 reserves the
remaining headroom for Phase 6 — the vLLM image (~10 GB) plus model weights
(~5 GB). Phase 4 already deferred torch for the same reason.

PyTorch is roughly 1 GB installed on Windows. Spending that on CPU embedding,
when the GPU work it would also enable is what the disk is being saved for, is
the wrong allocation.

## Decision
Use **fastembed** with `BAAI/bge-small-en-v1.5` (384-dim, 67 MB of ONNX weights).

- ONNX Runtime instead of PyTorch: no CUDA, no torch, ~130 MB total.
- Model weights are pre-downloaded into the Docker image at build time, so the
  first request does not pay an ~11 s download and the readiness probe does not
  flap.
- `embed_query()` prepends the bge instruction prefix; `embed_documents()` does
  not. The asymmetry is required by the model family, so the two are separate
  methods rather than one call with a flag.

## Alternatives considered
| Option | Why not |
| --- | --- |
| **sentence-transformers + bge-small** | same model, same quality; ~1 GB installed because of PyTorch, against a 29 GB disk budget already earmarked for vLLM |
| OpenAI `text-embedding-3-small` | no local disk cost and strong quality, but makes every ingestion and every query a paid network call, needs a key the owner has not supplied, and sends document content to a third party — wrong default for a corpus whose whole point is access control |
| Qdrant server-side inference | removes the dependency from our service entirely; rejected because the embedding step then disappears from our own latency breakdown, and measuring each stage separately is the point of this lab |
| `potion-base-8M` (30 MB, 256-dim) | smaller still, but static embeddings with materially lower retrieval quality; 67 MB is not the constraint worth optimising |
| bge-base / bge-large | better recall; 440 MB / 1.3 GB and 768/1024-dim vectors, which also grows the Qdrant index. Not justified on an 18-chunk corpus |
| TEI (HuggingFace text-embeddings-inference) | the right answer at scale — a dedicated embedding service with batching and GPU support; another container and another image on a full disk |

## Trade-offs
+ ~130 MB instead of ~1 GB, leaving the disk budget for the GPU phases
+ ONNX Runtime starts faster than torch with a smaller memory footprint
+ fastembed is Qdrant's own library, so the integration is direct
+ embedding latency stays in *our* latency breakdown (measured: 1.51 ms/doc, 4.6 ms/query)
− fewer models available than the sentence-transformers catalogue
− no fine-tuning path; a domain-adapted embedder would reopen this
− Phase 9's cross-encoder reranker must also exist as ONNX, or that phase
  reopens this decision. fastembed ships rerankers, so this is expected to hold
− one more artifact to pre-download at image build time

## Consequences
- `apps/rag-service` depends on `fastembed`, not `torch`.
- The embedding model name is part of `index_version` (ADR-015, artifact class 4),
  so changing it forces a reindex — correct, since vectors from two models are
  not comparable.
- Phase 9 uses a fastembed reranker; if one is unavailable or too weak, this ADR
  is superseded rather than worked around.
- Phase 11 evaluates retrieval configurations, and the embedding model is one
  axis — a larger bge variant gets compared on quality there, not on convenience.
- At production scale the answer is TEI or a managed embedding API, not a library
  inside the application process. Recorded so the local choice is not mistaken
  for a recommendation.
