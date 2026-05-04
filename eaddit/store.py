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

import heapq
import json
import math
import operator
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
        self._vectors: List[List[float]] = []
        self._metadata: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}

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
        return [self._chunks[cid] for cid in self._ids]

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
            v_list = [float(x) for x in vec]
            norm = math.hypot(*v_list)
            if norm > 0:
                v_list = [v / norm for v in v_list]

            if chunk.id in self._chunks:
                # Update existing ($O(1)$ lookup via _id_to_idx)
                idx = self._id_to_idx[chunk.id]
                self._chunks[chunk.id] = chunk
                self._vectors[idx] = v_list
                self._metadata[idx] = chunk.metadata
            else:
                # Add new
                idx = len(self._ids)
                self._ids.append(chunk.id)
                self._chunks[chunk.id] = chunk
                self._vectors.append(v_list)
                self._metadata.append(chunk.metadata)
                self._id_to_idx[chunk.id] = idx

    def delete(self, chunk_id: str) -> bool:
        if chunk_id not in self._chunks:
            return False
        idx = self._id_to_idx.pop(chunk_id)
        self._ids.pop(idx)
        self._chunks.pop(chunk_id)
        self._vectors.pop(idx)
        self._metadata.pop(idx)

        # Update indices for all items that shifted.
        # This makes delete O(N), but we prioritize search and add.
        for i in range(idx, len(self._ids)):
            self._id_to_idx[self._ids[i]] = i

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

        # Normalize query vector.
        q_norm = math.hypot(*query_vector) or 1.0
        q_vec = [float(x) / q_norm for x in query_vector]

        if metadata_filter is None:
            # For L2-normalized vectors, dot product relates to Euclidean distance:
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a.b) = 2 - 2(a.b)
            # => a.b = 1 - ||a - b||^2 / 2
            # math.dist() is implemented in C and is much faster than manual sum-of-products.
            scored = [
                (1.0 - math.dist(q_vec, v) ** 2 / 2.0, cid)
                for v, cid in zip(self._vectors, self._ids)
            ]
        else:
            scored = [
                (1.0 - math.dist(q_vec, v) ** 2 / 2.0, cid)
                for v, m, cid in zip(self._vectors, self._metadata, self._ids)
                if metadata_filter(m)
            ]

        top = heapq.nlargest(top_k, scored, key=lambda t: t[0])
        return [
            RetrievalResult(chunk=self._chunks[cid], score=float(s))
            for s, cid in top
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
                    "vector": vec,
                }
                for cid, vec in zip(self._ids, self._vectors)
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

        # Use a single add() call to build the store.
        if chunks:
            store.add(chunks, vectors)
        return store
