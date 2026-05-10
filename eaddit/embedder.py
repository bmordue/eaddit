"""Embedding backends.

The :class:`Embedder` protocol exposes a single :meth:`embed` method that
takes an iterable of texts and returns an equally-long list of float vectors.
The default implementation, :class:`HashingEmbedder`, is deterministic, has no
external dependencies, and produces L2-normalised vectors — which makes cosine
similarity reduce to a plain dot product downstream.

For higher-quality semantic embeddings, install the optional ``embeddings``
extra and use :class:`SentenceTransformerEmbedder`.
"""

from __future__ import annotations

import collections
import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Sequence, Tuple

# A simple word tokeniser. We deliberately avoid heavy dependencies here.
_WORD_RE = re.compile(r"[A-Za-z0-9_']+")


class Embedder(ABC):
    """Abstract embedder. Implementations must produce vectors of fixed dim."""

    @property
    @abstractmethod
    def dimension(self) -> int:  # pragma: no cover - trivial
        """Return the dimensionality of vectors produced by :meth:`embed`."""

    @abstractmethod
    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed a batch of texts."""

    def embed_one(self, text: str) -> List[float]:
        """Convenience wrapper for embedding a single text."""

        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """A deterministic feature-hashing bag-of-words embedder.

    Each token is hashed into one of ``dim`` buckets; the resulting vector is
    L2-normalised. The technique is sometimes called the "hashing trick" and
    is fast, memory-light, and good enough for unit tests and offline demos.

    Two texts that share many tokens will have a high cosine similarity; this
    is sufficient for the retrieval tests in this repository, but for real
    semantic search you should plug in a proper embedding model.
    """

    def __init__(self, dim: int = 256, batch_size: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._dim = dim
        self._batch_size = batch_size
        self._hash_cache: Dict[str, Tuple[int, float]] = {}

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        # Materialise once so we can batch and report errors clearly.
        texts = list(texts)
        out: List[List[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            for text in batch:
                out.append(self._embed_one(text))
        return out

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        if not text:
            return vec

        # Performance optimization: use collections.Counter for faster token
        # counting. re.findall() + .lower() on the whole text is faster than
        # per-token operations in pure Python.
        counts = collections.Counter(_WORD_RE.findall(text.lower()))

        # Performance optimization: use local variable for cache lookup speed
        # and avoid double lookups with dict.get().
        cache = self._hash_cache
        for token, count in counts.items():
            res = cache.get(token)
            if res is not None:
                bucket, sign = res
            else:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self._dim
                # The 5th byte's lowest bit decides the sign — keeps the embedding
                # roughly zero-mean for unrelated tokens.
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                cache[token] = (bucket, sign)
            vec[bucket] += sign * count

        norm = math.hypot(*vec)
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec


class SentenceTransformerEmbedder(Embedder):  # pragma: no cover - optional dep
    """Embedder backed by sentence-transformers.

    Requires the optional ``embeddings`` extra. Vectors are L2-normalised so
    that downstream cosine similarity is a plain dot product.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required; install with the "
                "'embeddings' extra: pip install eaddit[embeddings]"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=False,
        )
        return [list(map(float, v)) for v in vectors]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Returns 0.0 if either vector has zero magnitude.
    """

    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")

    # Performance optimization: use math.hypot and math.dist for cosine similarity.
    # In pure Python, these C-level implementations are significantly faster than
    # manual loops or zip/map/sum combinations.
    # The identity: 2 * <a, b> = ||a||^2 + ||b||^2 - ||a - b||^2
    # Cosine(a, b) = <a, b> / (||a|| * ||b||)
    #              = (||a||/||b|| + ||b||/||a|| - ||a - b||^2 / (||a|| * ||b||)) / 2
    na = math.hypot(*a)
    nb = math.hypot(*b)
    if na == 0.0 or nb == 0.0:
        return 0.0

    d = math.dist(a, b)
    return (na / nb + nb / na - (d * d) / (na * nb)) / 2.0
