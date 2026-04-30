"""End-to-end ingestion pipeline.

The pipeline glues together a :class:`Collector`, a :class:`Chunker`, an
:class:`Embedder`, and a vector store. It also handles the two cross-cutting
concerns called out in PLAN.md:

* **Deduplication** — chunks are hashed by content (``Chunk.content_hash``).
  Chunks that are already present in the store with the same hash are skipped,
  so re-ingesting unchanged data is a no-op.
* **Incremental updates** — a small JSON state file remembers, per subreddit,
  the highest-``created_utc`` post seen during the last run. Subsequent runs
  only ingest posts newer than that watermark.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .chunker import Chunker
from .collector import Collector
from .embedder import Embedder
from .models import Chunk, Comment, Post
from .store import InMemoryVectorStore


@dataclass
class IngestStats:
    """Summary of an ingestion run."""

    posts_seen: int = 0
    comments_seen: int = 0
    chunks_emitted: int = 0
    chunks_added: int = 0
    chunks_skipped_duplicate: int = 0


@dataclass
class IngestionPipeline:
    """Combine collector + chunker + embedder + store into a runnable pipeline.

    The pipeline is stateless apart from ``state_path``: if provided, the
    pipeline reads/writes a small JSON file there to support incremental runs.
    """

    collector: Collector
    chunker: Chunker
    embedder: Embedder
    store: InMemoryVectorStore
    state_path: Optional[str] = None
    _state: Dict[str, Dict[str, int]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.embedder.dimension != self.store.dimension:
            raise ValueError(
                f"embedder dim {self.embedder.dimension} does not match "
                f"store dim {self.store.dimension}"
            )
        self._state = self._load_state()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ingest(
        self,
        subreddit: str,
        limit: int = 100,
        min_score: int = 0,
        incremental: bool = True,
    ) -> IngestStats:
        """Ingest one subreddit. Returns counters describing what happened."""

        posts, comments = self.collector.collect(
            subreddit, limit=limit, min_score=min_score
        )

        watermark = 0
        if incremental:
            watermark = self._state.get(subreddit, {}).get("last_created_utc", 0)
            posts = [p for p in posts if p.created_utc > watermark]
            kept_post_ids = {p.id for p in posts}
            comments = [c for c in comments if c.post_id in kept_post_ids]

        stats = IngestStats(posts_seen=len(posts), comments_seen=len(comments))

        if not posts:
            # Persist state regardless so concurrent runs can converge.
            self._save_state()
            return stats

        chunks = self._build_chunks(posts, comments)
        stats.chunks_emitted = len(chunks)

        # Deduplicate against what is already in the store.
        new_chunks: List[Chunk] = []
        for chunk in chunks:
            existing = self.store.get(chunk.id)
            if existing is not None and existing.content_hash == chunk.content_hash:
                stats.chunks_skipped_duplicate += 1
                continue
            new_chunks.append(chunk)

        if new_chunks:
            vectors = self.embedder.embed([c.text for c in new_chunks])
            self.store.add(new_chunks, vectors)
            stats.chunks_added = len(new_chunks)

        # Update incremental watermark to the newest post seen this run.
        newest = max(p.created_utc for p in posts)
        if newest > watermark:
            self._state.setdefault(subreddit, {})["last_created_utc"] = newest
        self._save_state()

        return stats

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_chunks(
        self,
        posts: List[Post],
        comments: List[Comment],
    ) -> List[Chunk]:
        comments_by_post: Dict[str, List[Comment]] = {}
        for c in comments:
            comments_by_post.setdefault(c.post_id, []).append(c)

        out: List[Chunk] = []
        for post in posts:
            out.extend(self.chunker.chunk_thread(post, comments_by_post.get(post.id, [])))
        return out

    def _load_state(self) -> Dict[str, Dict[str, int]]:
        if not self.state_path:
            return {}
        path = Path(self.state_path)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _save_state(self) -> None:
        if not self.state_path:
            return
        path = Path(self.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._state, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
