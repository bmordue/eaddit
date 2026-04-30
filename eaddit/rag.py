"""RAG query interface.

The :class:`RAGQueryEngine` ties together an :class:`Embedder` and a vector
store. Given a natural-language query it:

1. embeds the query with the same model used at ingest time,
2. searches for the top-k most similar chunks (optionally filtered by
   metadata),
3. for comment chunks, optionally walks up the parent chain to attach the
   ancestor post/comments as additional context,
4. builds a prompt that can be handed to any LLM.

A pluggable :class:`LLM` protocol is provided so callers can supply their own
generation backend; ``EchoLLM`` is included as a deterministic stand-in for
tests and demos.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .embedder import Embedder
from .models import Chunk, RetrievalResult
from .store import InMemoryVectorStore, MetadataFilter


class LLM(ABC):
    """Trivial LLM interface — given a prompt, return a completion."""

    @abstractmethod
    def complete(self, prompt: str) -> str:  # pragma: no cover - trivial
        ...


class EchoLLM(LLM):
    """A no-op LLM that just echoes the prompt back. Useful for tests."""

    def complete(self, prompt: str) -> str:
        return prompt


@dataclass
class RAGAnswer:
    """The result of a full RAG query: prompt + LLM answer + retrieved hits."""

    query: str
    prompt: str
    answer: str
    results: List[RetrievalResult]


class RAGQueryEngine:
    """Run RAG-style queries against a vector store."""

    def __init__(
        self,
        store: InMemoryVectorStore,
        embedder: Embedder,
        llm: Optional[LLM] = None,
    ) -> None:
        if embedder.dimension != store.dimension:
            raise ValueError(
                f"embedder dim {embedder.dimension} does not match "
                f"store dim {store.dimension}"
            )
        self.store = store
        self.embedder = embedder
        self.llm = llm

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[MetadataFilter] = None,
        attach_ancestors: bool = True,
        max_ancestors: int = 4,
    ) -> List[RetrievalResult]:
        """Return the top-k most relevant chunks for ``query``.

        When ``attach_ancestors`` is true, comment hits are decorated with
        their ancestor chain (root post first, then parent comments). This
        lets the prompt builder include thread context even if only one
        comment in the thread was retrieved.
        """

        q_vec = self.embedder.embed_one(query)
        results = self.store.search(q_vec, top_k=top_k, metadata_filter=metadata_filter)
        if attach_ancestors:
            for r in results:
                r.ancestors = self._ancestor_chunks(r.chunk, max_ancestors=max_ancestors)
        return results

    def query(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[MetadataFilter] = None,
        attach_ancestors: bool = True,
    ) -> RAGAnswer:
        """Retrieve, build a prompt, and (if an LLM is configured) generate."""

        results = self.retrieve(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            attach_ancestors=attach_ancestors,
        )
        prompt = build_prompt(query, results)
        answer = self.llm.complete(prompt) if self.llm is not None else ""
        return RAGAnswer(query=query, prompt=prompt, answer=answer, results=results)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _ancestor_chunks(self, chunk: Chunk, max_ancestors: int) -> List[Chunk]:
        meta = chunk.metadata
        if meta.get("comment_id") is None:
            return []  # post chunks don't have ancestors

        ancestors: List[Chunk] = []
        seen = {chunk.id}

        # Walk up via parent_id, falling back to the post chunk at the root.
        parent_id = meta.get("parent_id")
        post_id = meta.get("post_id")
        depth_guard = 0
        while parent_id and depth_guard < max_ancestors:
            depth_guard += 1
            parent_chunk = self.store.get(f"comment:{parent_id}")
            if parent_chunk is None or parent_chunk.id in seen:
                break
            ancestors.append(parent_chunk)
            seen.add(parent_chunk.id)
            parent_id = parent_chunk.metadata.get("parent_id")

        if post_id:
            post_chunk = self.store.get(f"post:{post_id}")
            if post_chunk is not None and post_chunk.id not in seen:
                ancestors.append(post_chunk)

        ancestors.reverse()  # root-most first
        return ancestors


# ---------------------------------------------------------------------- #
# Prompt formatting
# ---------------------------------------------------------------------- #
def build_prompt(query: str, results: Sequence[RetrievalResult]) -> str:
    """Format ``query`` and retrieved chunks into a single prompt string.

    The format is intentionally model-agnostic: section headers and a final
    ``Question:`` line. Any LLM front-end can consume it.
    """

    sections: List[str] = []
    sections.append(
        "You are a helpful assistant. Use the following Reddit excerpts to "
        "answer the question. If the excerpts do not contain the answer, say "
        "you do not know."
    )
    for i, r in enumerate(results, start=1):
        sections.append(f"\n[Excerpt {i}] (score={r.score:.4f})")
        for a in r.ancestors:
            sections.append(_format_chunk(a, prefix="context: "))
        sections.append(_format_chunk(r.chunk))
    sections.append(f"\nQuestion: {query}\nAnswer:")
    return "\n".join(sections)


def _format_chunk(chunk: Chunk, prefix: str = "") -> str:
    meta = chunk.metadata
    kind = "post" if meta.get("comment_id") is None else "comment"
    author = meta.get("author") or "unknown"
    return f"{prefix}({kind} by {author}) {chunk.text}"
