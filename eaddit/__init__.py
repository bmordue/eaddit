"""eaddit — ingest a subreddit into a vector database for RAG queries.

The package is organised into small, composable modules:

* :mod:`eaddit.models`     — dataclasses for posts, comments, chunks, retrieval results.
* :mod:`eaddit.collector`  — pluggable Reddit data collectors.
* :mod:`eaddit.chunker`    — thread-aware chunker that turns posts/comments into chunks.
* :mod:`eaddit.embedder`   — embedder interface plus a deterministic in-process backend.
* :mod:`eaddit.store`      — in-process vector store with JSON persistence.
* :mod:`eaddit.ingest`     — end-to-end ingestion pipeline (with dedup + incremental state).
* :mod:`eaddit.rag`        — RAG query interface (top-k search, ancestor expansion, prompts).
* :mod:`eaddit.cli`        — command-line entry point.

See :class:`eaddit.ingest.IngestionPipeline` and :class:`eaddit.rag.RAGQueryEngine`
for the typical entry points.
"""

from .models import Chunk, Comment, Post, RetrievalResult
from .chunker import Chunker
from .embedder import Embedder, HashingEmbedder
from .store import InMemoryVectorStore
from .ingest import IngestionPipeline
from .rag import RAGQueryEngine

__all__ = [
    "Chunk",
    "Chunker",
    "Comment",
    "Embedder",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "IngestionPipeline",
    "Post",
    "RAGQueryEngine",
    "RetrievalResult",
]

__version__ = "0.1.0"
