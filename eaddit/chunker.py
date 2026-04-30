"""Thread-aware chunker.

Posts are emitted as a single chunk containing ``title\\n\\nbody``. Comments are
emitted as thread-aware chunks: each comment's text is prefixed with a short
context block describing the post and (optionally) its ancestor comments, so
that every chunk is self-contained when retrieved later.

When the resulting text exceeds ``chunk_size`` (measured in whitespace tokens —
a deliberately simple heuristic to avoid heavy tokeniser dependencies), it is
split into overlapping windows of ``chunk_size`` tokens with ``chunk_overlap``
tokens of overlap.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .models import Chunk, Comment, Post, comment_metadata, post_metadata


@dataclass
class Chunker:
    """Turn :class:`Post` and :class:`Comment` objects into :class:`Chunk` objects.

    Parameters
    ----------
    chunk_size:
        Target chunk size, measured in whitespace-separated tokens.
    chunk_overlap:
        Number of tokens of overlap between consecutive windows when a chunk
        has to be split. Must be strictly less than ``chunk_size``.
    include_thread_context:
        When ``True`` (the default), each comment chunk is prefixed with a short
        block summarising its post and ancestor comments. When ``False`` the
        comment text is embedded as-is (the "flat" mode discussed in PLAN.md).
    max_ancestors:
        Cap on the number of ancestor comments included in the context block.
    """

    chunk_size: int = 256
    chunk_overlap: int = 32
    include_thread_context: bool = True
    max_ancestors: int = 4

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chunk_post(self, post: Post) -> List[Chunk]:
        title = post.title.strip()
        body = post.body.strip()
        text = f"{title}\n\n{body}".strip() if body else title
        return self._make_chunks(
            base_id=f"post:{post.id}",
            text=text,
            metadata=post_metadata(post),
        )

    def chunk_comment(
        self,
        comment: Comment,
        post: Optional[Post] = None,
        ancestors: Optional[Sequence[Comment]] = None,
    ) -> List[Chunk]:
        body = comment.body.strip()
        if self.include_thread_context:
            context = self._build_context(post, ancestors or ())
            text = f"{context}\n---\n{body}".strip() if context else body
        else:
            text = body
        return self._make_chunks(
            base_id=f"comment:{comment.id}",
            text=text,
            metadata=comment_metadata(comment, post),
        )

    def chunk_thread(
        self,
        post: Post,
        comments: Iterable[Comment],
    ) -> List[Chunk]:
        """Chunk a whole thread, computing each comment's ancestor chain.

        The input ``comments`` does not need to be sorted; the chunker rebuilds
        the parent → child graph internally.
        """

        comments = list(comments)
        by_id: Dict[str, Comment] = {c.id: c for c in comments}

        chunks: List[Chunk] = []
        chunks.extend(self.chunk_post(post))
        for c in comments:
            ancestors = _ancestor_chain(c, by_id, self.max_ancestors)
            chunks.extend(self.chunk_comment(c, post=post, ancestors=ancestors))
        return chunks

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_context(
        self,
        post: Optional[Post],
        ancestors: Sequence[Comment],
    ) -> str:
        lines: List[str] = []
        if post is not None:
            lines.append(f"Post (r/{post.subreddit}): {post.title.strip()}")
            if post.body.strip():
                lines.append(post.body.strip())
        if ancestors:
            lines.append("Thread context:")
            for a in ancestors:
                snippet = a.body.strip().replace("\n", " ")
                if len(snippet) > 240:
                    snippet = snippet[:237] + "..."
                lines.append(f"- {a.author or 'unknown'}: {snippet}")
        return "\n".join(lines)

    def _make_chunks(
        self,
        *,
        base_id: str,
        text: str,
        metadata: Dict,
    ) -> List[Chunk]:
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        if len(tokens) <= self.chunk_size:
            return [self._build_chunk(base_id, 0, text, metadata)]

        out: List[Chunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for idx, start in enumerate(range(0, len(tokens), step)):
            window = tokens[start : start + self.chunk_size]
            if not window:
                break
            chunk_text = " ".join(window)
            out.append(self._build_chunk(base_id, idx, chunk_text, metadata))
            if start + self.chunk_size >= len(tokens):
                break
        return out

    @staticmethod
    def _build_chunk(base_id: str, idx: int, text: str, metadata: Dict) -> Chunk:
        cid = base_id if idx == 0 else f"{base_id}#{idx}"
        meta = dict(metadata)
        meta["chunk_index"] = idx
        return Chunk(
            id=cid,
            text=text,
            metadata=meta,
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )


def _ancestor_chain(
    comment: Comment,
    by_id: Dict[str, Comment],
    max_ancestors: int,
) -> List[Comment]:
    chain: List[Comment] = []
    current = by_id.get(comment.parent_id)
    seen = {comment.id}
    while current is not None and current.id not in seen and len(chain) < max_ancestors:
        chain.append(current)
        seen.add(current.id)
        current = by_id.get(current.parent_id)
    chain.reverse()  # root-most first
    return chain
