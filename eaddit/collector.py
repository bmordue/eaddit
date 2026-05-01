"""Reddit data collectors.

Two implementations are shipped:

* :class:`JSONFixtureCollector` — reads posts and comments from a JSON file or
  a Python dict. It has zero external dependencies and is the default backend
  used by tests and offline development.
* :class:`PRAWCollector` — thin adapter around the `praw` library for live
  Reddit access. It is only imported lazily so that the optional ``praw``
  dependency is not required for normal use.

All collectors implement :class:`Collector`, returning ``(posts, comments)``
filtered by a minimum-score threshold.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Sequence, Tuple, Union

from .models import Comment, Post


class Collector(ABC):
    """Abstract Reddit data collector."""

    @abstractmethod
    def collect(
        self,
        subreddit: str,
        limit: int = 100,
        min_score: int = 0,
    ) -> Tuple[List[Post], List[Comment]]:
        """Collect posts and comments from ``subreddit``.

        ``limit`` caps the number of posts returned. ``min_score`` filters
        out posts and comments whose score is below the threshold.
        """


def _filter(items: Sequence, min_score: int) -> List:
    return [i for i in items if getattr(i, "score", 0) >= min_score]


class JSONFixtureCollector(Collector):
    """Collector backed by an in-memory mapping or a JSON file.

    The expected JSON structure is::

        {
          "<subreddit>": {
            "posts":    [ { ...Post fields... }, ... ],
            "comments": [ { ...Comment fields... }, ... ]
          },
          ...
        }

    Unknown fields are ignored. This collector is deterministic, hermetic, and
    fast — ideal for tests and reproducible local development.
    """

    def __init__(self, source: Union[str, Path, Mapping[str, Any]]):
        if isinstance(source, (str, Path)):
            with open(source, "r", encoding="utf-8") as fh:
                self._data: Mapping[str, Any] = json.load(fh)
        else:
            self._data = source

    def collect(
        self,
        subreddit: str,
        limit: int = 100,
        min_score: int = 0,
    ) -> Tuple[List[Post], List[Comment]]:
        bucket = self._data.get(subreddit, {})
        raw_posts: Iterable[Mapping[str, Any]] = bucket.get("posts", [])
        raw_comments: Iterable[Mapping[str, Any]] = bucket.get("comments", [])

        posts = [_post_from_dict(p) for p in raw_posts][:limit]
        post_ids = {p.id for p in posts}
        comments = [_comment_from_dict(c) for c in raw_comments]
        # Only keep comments belonging to the posts we are returning.
        comments = [c for c in comments if c.post_id in post_ids]

        return _filter(posts, min_score), _filter(comments, min_score)


def _post_from_dict(d: Mapping[str, Any]) -> Post:
    return Post(
        id=str(d["id"]),
        title=str(d.get("title", "")),
        body=str(d.get("body", "") or ""),
        score=int(d.get("score", 0)),
        url=str(d.get("url", "")),
        created_utc=int(d.get("created_utc", 0)),
        subreddit=str(d.get("subreddit", "")),
        author=str(d.get("author", "") or ""),
    )


def _comment_from_dict(d: Mapping[str, Any]) -> Comment:
    return Comment(
        id=str(d["id"]),
        post_id=str(d["post_id"]),
        parent_id=str(d.get("parent_id", d["post_id"])),
        body=str(d.get("body", "") or ""),
        score=int(d.get("score", 0)),
        created_utc=int(d.get("created_utc", 0)),
        author=str(d.get("author", "") or ""),
        depth=int(d.get("depth", 1)),
    )


class PRAWCollector(Collector):  # pragma: no cover - exercised only with network
    """Live Reddit collector backed by the `praw` library.

    This class is only imported lazily; instantiation requires the optional
    ``reddit`` extra. Pass either a configured ``praw.Reddit`` instance via
    ``reddit=`` or the credentials needed to construct one.
    """

    def __init__(
        self,
        reddit: Any = None,
        *,
        client_id: str = "",
        client_secret: str = "",
        user_agent: str = "eaddit/0.1",
    ) -> None:
        if reddit is None:
            try:
                import praw  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "praw is required for PRAWCollector; install with the "
                    "'reddit' extra: pip install eaddit[reddit]"
                ) from exc
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
        self._reddit = reddit

    def collect(
        self,
        subreddit: str,
        limit: int = 100,
        min_score: int = 0,
    ) -> Tuple[List[Post], List[Comment]]:
        sub = self._reddit.subreddit(subreddit)
        posts: List[Post] = []
        comments: List[Comment] = []
        for submission in sub.new(limit=limit):
            posts.append(
                Post(
                    id=str(submission.id),
                    title=str(submission.title or ""),
                    body=str(getattr(submission, "selftext", "") or ""),
                    score=int(getattr(submission, "score", 0) or 0),
                    url=str(getattr(submission, "url", "") or ""),
                    created_utc=int(getattr(submission, "created_utc", 0) or 0),
                    subreddit=subreddit,
                    author=str(submission.author.name if submission.author else ""),
                )
            )
            submission.comments.replace_more(limit=0)
            for c in submission.comments.list():
                parent_id = str(getattr(c, "parent_id", submission.id) or submission.id)
                # Strip Reddit "t1_"/"t3_" prefixes for consistency with our model.
                if "_" in parent_id:
                    parent_id = parent_id.split("_", 1)[1]
                comments.append(
                    Comment(
                        id=str(c.id),
                        post_id=str(submission.id),
                        parent_id=parent_id,
                        body=str(getattr(c, "body", "") or ""),
                        score=int(getattr(c, "score", 0) or 0),
                        created_utc=int(getattr(c, "created_utc", 0) or 0),
                        author=str(c.author.name if c.author else ""),
                        depth=int(getattr(c, "depth", 0)) + 1,
                    )
                )
        return _filter(posts, min_score), _filter(comments, min_score)
