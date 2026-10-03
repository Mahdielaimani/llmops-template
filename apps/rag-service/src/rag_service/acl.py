"""Access control for documents.

One `AclPredicate` object is the single source of truth, rendered three ways:
a Qdrant payload filter, a callable for the BM25 side, and a re-check at context
assembly. ADR-019 requires all three to agree, so they are derived rather than
written separately.

The predicate is applied **inside the retrieval query**, never as a filter on its
results — see docs/security.md §2 for the three ways post-filtering leaks.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field


class Classification(IntEnum):
    """Ordered so `doc.classification <= user.clearance` is the whole rule.

    An IntEnum rather than a string enum precisely because the comparison must be
    an ordering, and a typo in a string comparison fails open.
    """

    PUBLIC = 0
    INTERNAL = 1
    RESTRICTED = 2
    CONFIDENTIAL = 3

    @classmethod
    def parse(cls, raw: str) -> Classification:
        return cls[raw.strip().upper()]


class User(BaseModel):
    """Identity as it arrives from the token (Phase 16 replaces the stub issuer)."""

    user_id: str
    clearance: Classification = Classification.PUBLIC
    departments: frozenset[str] = frozenset()
    # Time-boxed external auditors are why this is ABAC and not RBAC: a role
    # cannot express "these documents, for six weeks" (ADR-019).
    valid_from: date | None = None
    valid_until: date | None = None
    scoped_doc_ids: frozenset[str] | None = None  # None = no restriction

    model_config = {"frozen": True}

    @property
    def is_within_validity(self) -> bool:
        today = datetime.now(UTC).date()
        if self.valid_from and today < self.valid_from:
            return False
        return not (self.valid_until and today > self.valid_until)


class AclPredicate(BaseModel):
    """Derived from a user, then rendered per backend. Never constructed by hand."""

    clearance: Classification
    departments: frozenset[str]
    scoped_doc_ids: frozenset[str] | None = None
    expired: bool = False

    model_config = {"frozen": True}

    @classmethod
    def for_user(cls, user: User) -> AclPredicate:
        return cls(
            clearance=user.clearance,
            departments=user.departments,
            scoped_doc_ids=user.scoped_doc_ids,
            expired=not user.is_within_validity,
        )

    def allows(self, payload: dict[str, Any]) -> bool:
        """The re-check at context assembly, and the BM25-side filter.

        Fails closed: a payload missing `classification` is treated as the most
        restrictive value, because an unlabelled document is not a public one.
        """
        if self.expired:
            return False
        raw = payload.get("classification")
        if raw is None:
            return False
        level = Classification.parse(raw) if isinstance(raw, str) else Classification(raw)
        if level > self.clearance:
            return False
        if self.scoped_doc_ids is not None and payload.get("doc_id") not in self.scoped_doc_ids:
            return False
        # Public documents are department-agnostic; everything else needs a match.
        if level is Classification.PUBLIC:
            return True
        return payload.get("department") in self.departments

    def qdrant_filter(self) -> dict[str, Any]:
        """A Qdrant `Filter` as a plain dict, so this module needs no qdrant import
        and stays unit-testable without a server.

        Expressed as: classification <= clearance AND (public OR department matches),
        plus the optional document scope. An expired user matches nothing.
        """
        if self.expired:
            # `must` an impossible condition rather than returning None: a caller
            # that forgets to check `expired` must still retrieve zero rows.
            return {"must": [{"key": "classification_level", "match": {"value": -1}}]}

        allowed_levels = [int(c) for c in Classification if c <= self.clearance]
        must: list[dict[str, Any]] = [
            {"key": "classification_level", "match": {"any": allowed_levels}}
        ]
        if self.departments:
            must.append(
                {
                    "should": [
                        {
                            "key": "classification_level",
                            "match": {"value": int(Classification.PUBLIC)},
                        },
                        {"key": "department", "match": {"any": sorted(self.departments)}},
                    ]
                }
            )
        else:
            must.append(
                {"key": "classification_level", "match": {"value": int(Classification.PUBLIC)}}
            )
        if self.scoped_doc_ids is not None:
            must.append({"key": "doc_id", "match": {"any": sorted(self.scoped_doc_ids)}})
        return {"must": must}

    def scope_hash(self) -> str:
        """Cache key component. Two users with different permissions must never share
        a cached answer — the bypass described in ADR-019 and docs/security.md §2."""
        parts = [
            str(int(self.clearance)),
            ",".join(sorted(self.departments)),
            ",".join(sorted(self.scoped_doc_ids)) if self.scoped_doc_ids is not None else "*",
            "expired" if self.expired else "valid",
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


class Chunk(BaseModel):
    """What the retriever returns. Carries everything a citation needs, so an answer
    can be traced to `(doc_id, doc_version, chunk_id, page)` forever."""

    chunk_id: str
    doc_id: str
    doc_version: int
    title: str
    text: str
    page: int
    classification: str
    department: str
    effective_date: str
    score: float = 0.0
    retrieved_by: str = "dense"

    @property
    def citation(self) -> str:
        return f"{self.doc_id}@v{self.doc_version}#{self.chunk_id} (p.{self.page})"


def payload_of(chunk: Chunk) -> dict[str, Any]:
    """Qdrant payload. `classification_level` is stored alongside the string form
    because the filter needs an ordered integer and humans need the label."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_version": chunk.doc_version,
        "title": chunk.title,
        "text": chunk.text,
        "page": chunk.page,
        "classification": chunk.classification,
        "classification_level": int(Classification.parse(chunk.classification)),
        "department": chunk.department,
        "effective_date": chunk.effective_date,
    }


def chunk_of(payload: dict[str, Any], score: float = 0.0, by: str = "dense") -> Chunk:
    return Chunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        doc_version=payload["doc_version"],
        title=payload["title"],
        text=payload["text"],
        page=payload["page"],
        classification=payload["classification"],
        department=payload["department"],
        effective_date=payload["effective_date"],
        score=score,
        retrieved_by=by,
    )


class RagRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    candidates: int = Field(default=20, ge=1, le=200)
