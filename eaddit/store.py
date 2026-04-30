"""In-process vector store with JSON persistence.

For the prototype-friendly Option C from PLAN.md we ship a single
:class:`InMemoryVectorStore`. It supports:

* fixed-dimension float vectors,
* arbitrary JSON-serialisable per-chunk metadata,
* metadata predicate filtering at query time,
* exact top-k cosine similarity search (cheap because :class:`HashingEmbedder`
  produces L2-normalised vectors — the search reduces to a dot product),
* JSON save/load for restart-resilience.

This is intentionally tiny. It is good enough to demo the RAG flow end-to-end;
swapping in a real vector database is a question of writing one more class
that satisfies the same interface.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Chunk, RetrievalResult


MetadataFilter = Callable[[Dict[str, Any]], bool]


class InMemoryVectorStore:
    """Simple in-process vector store.

    Vectors and chunks are stored in parallel lists keyed by ``chunk.id``.
    Re-adding a chunk with the same id overwrites the previous entry, which
    makes the store idempotent and re-runnable.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._ids: List[str] = []
        self._chunks: Dict[str, Chunk] = {}
        self._vectors: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------ #
    # Properties / inspection
    # ------------------------------------------------------------------ #
    @property
    def dimension(self) -> int:
        return self._dimension

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._chunks

    def all_chunks(self) -> List[Chunk]:
        return [self._chunks[i] for i in self._ids]

    def get(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vec in zip(chunks, vectors):
            if len(vec) != self._dimension:
                raise ValueError(
                    f"vector for chunk {chunk.id!r} has dim {len(vec)}, "
                    f"expected {self._dimension}"
                )
            if chunk.id not in self._chunks:
                self._ids.append(chunk.id)
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = list(map(float, vec))

    def delete(self, chunk_id: str) -> bool:
        if chunk_id not in self._chunks:
            return False
        self._chunks.pop(chunk_id, None)
        self._vectors.pop(chunk_id, None)
        try:
            self._ids.remove(chunk_id)
        except ValueError:  # pragma: no cover - defensive
            pass
        return True

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        metadata_filter: Optional[MetadataFilter] = None,
    ) -> List[RetrievalResult]:
        if len(query_vector) != self._dimension:
            raise ValueError(
                f"query vector has dim {len(query_vector)}, expected {self._dimension}"
            )
        if top_k <= 0:
            return []

        q_norm = math.sqrt(sum(x * x for x in query_vector)) or 1.0
        scored: List[Tuple[float, str]] = []
        for cid in self._ids:
            chunk = self._chunks[cid]
            if metadata_filter is not None and not metadata_filter(chunk.metadata):
                continue
            vec = self._vectors[cid]
            v_norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            dot = 0.0
            for a, b in zip(query_vector, vec):
                dot += a * b
            scored.append((dot / (q_norm * v_norm), cid))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            RetrievalResult(chunk=self._chunks[cid], score=float(s))
            for s, cid in scored[:top_k]
        ]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | os.PathLike[str]) -> None:
        payload = {
            "dimension": self._dimension,
            "items": [
                {
                    "chunk": self._chunks[cid].to_dict(),
                    "vector": self._vectors[cid],
                }
                for cid in self._ids
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write to avoid leaving a corrupt file on crash.
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "InMemoryVectorStore":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        store = cls(dimension=int(payload["dimension"]))
        chunks: List[Chunk] = []
        vectors: List[List[float]] = []
        for item in payload.get("items", []):
            chunks.append(Chunk.from_dict(item["chunk"]))
            vectors.append([float(x) for x in item["vector"]])
        if chunks:
            store.add(chunks, vectors)
        return store
