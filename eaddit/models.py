"""Core data models for eaddit.

These dataclasses are deliberately lightweight and JSON-serialisable. They form
the contract between the collector, chunker, embedder, store and query layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Post:
    """A Reddit post (the top-level item of a thread).

    Attributes mirror the Reddit API fields the project cares about. ``id`` is
    expected to be the Reddit ``t3_`` identifier (without prefix is also fine,
    as long as it is stable for deduplication).
    """

    id: str
    title: str
    body: str
    score: int
    url: str
    created_utc: int
    subreddit: str
    author: str

    def to_dict(self) -> Dict[str, Any]:
        # Performance optimization: manual dict creation is ~7x faster than
        # dataclasses.asdict() for simple flat models.
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "score": self.score,
            "url": self.url,
            "created_utc": self.created_utc,
            "subreddit": self.subreddit,
            "author": self.author,
        }


@dataclass(frozen=True)
class Comment:
    """A Reddit comment.

    ``parent_id`` is the immediate parent (post id for top-level comments,
    another comment id otherwise). ``depth`` is 1 for top-level comments,
    2 for replies, and so on. ``post_id`` always refers to the root post.
    """

    id: str
    post_id: str
    parent_id: str
    body: str
    score: int
    created_utc: int
    author: str
    depth: int

    def to_dict(self) -> Dict[str, Any]:
        # Performance optimization: manual dict creation is ~7x faster than
        # dataclasses.asdict() for simple flat models.
        return {
            "id": self.id,
            "post_id": self.post_id,
            "parent_id": self.parent_id,
            "body": self.body,
            "score": self.score,
            "created_utc": self.created_utc,
            "author": self.author,
            "depth": self.depth,
        }


@dataclass
class Chunk:
    """A unit of text plus metadata that gets embedded and stored.

    ``text`` is the content actually fed to the embedder. ``content_hash`` is
    used by the ingestion pipeline for deduplication so that re-running ingest
    on unchanged data is cheap and idempotent.
    """

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=dict(data.get("metadata", {})),
            content_hash=data.get("content_hash", ""),
        )


@dataclass
class RetrievalResult:
    """A single search hit returned by the vector store / RAG engine."""

    chunk: Chunk
    score: float
    ancestors: List[Chunk] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "ancestors": [a.to_dict() for a in self.ancestors],
        }


def sanitize(value: Optional[str], limit: Optional[int] = 200) -> Optional[str]:
    """Truncate and strip control characters to prevent prompt/log injection."""
    if value is None:
        return None
    # Truncate first to minimize work on long strings.
    s = str(value)
    if limit is not None:
        s = s[:limit]
    # Replace newlines and other control characters with spaces.
    return "".join(c if c.isprintable() else " " for c in s).strip()


def post_metadata(post: Post) -> Dict[str, Any]:
    """Return the metadata schema described in PLAN.md for a post chunk."""

    return {
        "post_id": sanitize(post.id),
        "comment_id": None,
        "score": post.score,
        "created_utc": post.created_utc,
        "author": sanitize(post.author),
        "url": sanitize(post.url, limit=None),
        "parent_id": None,
        "depth": 0,
        "subreddit": sanitize(post.subreddit),
    }


def comment_metadata(comment: Comment, post: Optional[Post] = None) -> Dict[str, Any]:
    """Return the metadata schema described in PLAN.md for a comment chunk."""

    return {
        "post_id": sanitize(comment.post_id),
        "comment_id": sanitize(comment.id),
        "score": comment.score,
        "created_utc": comment.created_utc,
        "author": sanitize(comment.author),
        "url": sanitize(post.url, limit=None) if post is not None else None,
        "parent_id": sanitize(comment.parent_id),
        "depth": comment.depth,
        "subreddit": sanitize(post.subreddit) if post is not None else None,
    }
