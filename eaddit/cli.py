"""Command-line entry point for eaddit.

Subcommands:

* ``ingest``  — collect a subreddit (live via PRAW or from a JSON fixture) and
  ingest it into a local vector store.
* ``query``   — run a RAG-style query against a previously-built store.
* ``info``    — print summary statistics about a store.

The CLI is intentionally dependency-free: it relies only on the standard
library plus the eaddit package itself. The optional PRAW backend is loaded
lazily and only when ``ingest --source reddit`` is requested.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .chunker import Chunker
from .collector import Collector, JSONFixtureCollector
from .embedder import Embedder, HashingEmbedder
from .ingest import IngestionPipeline
from .models import Chunk
from .rag import RAGQueryEngine, build_prompt
from .store import InMemoryVectorStore


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _load_or_create_store(path: str, dimension: int) -> InMemoryVectorStore:
    p = Path(path)
    if p.exists():
        store = InMemoryVectorStore.load(p)
        if store.dimension != dimension:
            raise SystemExit(
                f"existing store at {path} has dim {store.dimension}, "
                f"but embedder produces dim {dimension}"
            )
        return store
    return InMemoryVectorStore(dimension=dimension)


def _build_embedder(dim: int) -> Embedder:
    return HashingEmbedder(dim=dim)


def _build_collector(args: argparse.Namespace) -> Collector:
    if args.source == "json":
        if not args.fixture:
            raise SystemExit("--fixture is required when --source json")
        return JSONFixtureCollector(args.fixture)
    if args.source == "reddit":  # pragma: no cover - requires praw + network
        from .collector import PRAWCollector

        return PRAWCollector(
            client_id=args.client_id or "",
            client_secret=args.client_secret or "",
            user_agent=args.user_agent or "eaddit/0.1",
        )
    raise SystemExit(f"unknown source: {args.source}")


def _make_metadata_filter(
    min_score: Optional[int],
    since: Optional[int],
    until: Optional[int],
) -> Optional[Callable[[Dict[str, Any]], bool]]:
    if min_score is None and since is None and until is None:
        return None

    def predicate(meta: Dict[str, Any]) -> bool:
        if min_score is not None and int(meta.get("score") or 0) < min_score:
            return False
        if since is not None and int(meta.get("created_utc") or 0) < since:
            return False
        if until is not None and int(meta.get("created_utc") or 0) > until:
            return False
        return True

    return predicate


# ---------------------------------------------------------------------- #
# Subcommand handlers
# ---------------------------------------------------------------------- #
def cmd_ingest(args: argparse.Namespace) -> int:
    embedder = _build_embedder(args.dim)
    store = _load_or_create_store(args.store, dimension=embedder.dimension)
    chunker = Chunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        include_thread_context=not args.flat_comments,
    )
    pipeline = IngestionPipeline(
        collector=_build_collector(args),
        chunker=chunker,
        embedder=embedder,
        store=store,
        state_path=args.state,
    )
    stats = pipeline.ingest(
        subreddit=args.subreddit,
        limit=args.limit,
        min_score=args.min_score,
        incremental=not args.full,
    )
    store.save(args.store)
    print(json.dumps(stats.__dict__, indent=2))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    embedder = _build_embedder(args.dim)
    store_path = Path(args.store)
    if not store_path.exists():
        raise SystemExit(f"no store found at {args.store}; run `ingest` first")
    store = InMemoryVectorStore.load(store_path)
    if store.dimension != embedder.dimension:
        raise SystemExit(
            f"store dim {store.dimension} does not match embedder dim "
            f"{embedder.dimension}; pass --dim {store.dimension}"
        )
    engine = RAGQueryEngine(store=store, embedder=embedder)
    metadata_filter = _make_metadata_filter(args.min_score, args.since, args.until)
    results = engine.retrieve(
        args.text,
        top_k=args.top_k,
        metadata_filter=metadata_filter,
        attach_ancestors=not args.no_ancestors,
    )
    if args.format == "prompt":
        print(build_prompt(args.text, results))
    else:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    store_path = Path(args.store)
    if not store_path.exists():
        raise SystemExit(f"no store found at {args.store}")
    store = InMemoryVectorStore.load(store_path)
    chunks = store.all_chunks()
    posts = sum(1 for c in chunks if c.metadata.get("comment_id") is None)
    comments = len(chunks) - posts
    subreddits = sorted({c.metadata.get("subreddit") for c in chunks if c.metadata.get("subreddit")})
    print(json.dumps(
        {
            "path": str(store_path),
            "dimension": store.dimension,
            "chunks": len(chunks),
            "post_chunks": posts,
            "comment_chunks": comments,
            "subreddits": subreddits,
        },
        indent=2,
    ))
    return 0


# ---------------------------------------------------------------------- #
# argparse plumbing
# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eaddit",
        description="Ingest a subreddit into a local vector store and run RAG queries.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest --------------------------------------------------------------
    p_ing = sub.add_parser("ingest", help="Collect, chunk, embed and store a subreddit.")
    p_ing.add_argument("subreddit", help="Subreddit name, e.g. 'python'.")
    p_ing.add_argument("--store", required=True, help="Path to the JSON-backed vector store.")
    p_ing.add_argument("--source", choices=("json", "reddit"), default="json")
    p_ing.add_argument("--fixture", help="Path to a JSON fixture (when --source json).")
    p_ing.add_argument("--client-id", help="Reddit API client id (when --source reddit).")
    p_ing.add_argument("--client-secret", help="Reddit API client secret (when --source reddit).")
    p_ing.add_argument("--user-agent", help="Reddit API user agent.")
    p_ing.add_argument("--limit", type=int, default=100)
    p_ing.add_argument("--min-score", type=int, default=0)
    p_ing.add_argument("--chunk-size", type=int, default=256)
    p_ing.add_argument("--chunk-overlap", type=int, default=32)
    p_ing.add_argument("--dim", type=int, default=256, help="Embedding dimension.")
    p_ing.add_argument("--flat-comments", action="store_true",
                       help="Disable thread context prefix for comment chunks.")
    p_ing.add_argument("--state", help="Path to JSON file tracking incremental state.")
    p_ing.add_argument("--full", action="store_true",
                       help="Ignore the incremental watermark and re-ingest everything.")
    p_ing.set_defaults(func=cmd_ingest)

    # query ---------------------------------------------------------------
    p_q = sub.add_parser("query", help="Run a RAG-style query.")
    p_q.add_argument("text", help="The natural-language query.")
    p_q.add_argument("--store", required=True)
    p_q.add_argument("--top-k", type=int, default=5)
    p_q.add_argument("--dim", type=int, default=256)
    p_q.add_argument("--min-score", type=int, default=None)
    p_q.add_argument("--since", type=int, default=None, help="Earliest created_utc to include.")
    p_q.add_argument("--until", type=int, default=None, help="Latest created_utc to include.")
    p_q.add_argument("--no-ancestors", action="store_true",
                     help="Do not attach ancestor chains to comment hits.")
    p_q.add_argument("--format", choices=("json", "prompt"), default="json")
    p_q.set_defaults(func=cmd_query)

    # info ----------------------------------------------------------------
    p_i = sub.add_parser("info", help="Show statistics about a vector store.")
    p_i.add_argument("--store", required=True)
    p_i.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
