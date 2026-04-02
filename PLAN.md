# eaddit Implementation Plan

## Overview

Ingest a subreddit into a vector database to support RAG (Retrieval-Augmented Generation) queries.

---

## Phase 1: Data Ingestion Pipeline

### 1.1 Reddit Data Collection

- Use `praw` (or `asyncpraw` for async) to collect posts and comments from target subreddits
- Fields to collect:
  - `post_id`, `title`, `selftext`, `score`, `url`, `created_utc`, `subreddit`, `author`
  - `comment_id`, `body`, `score`, `created_utc`, `author`, `parent_id`, `depth`
- Filter by score threshold to avoid low-quality noise

### 1.2 Chunking Strategy

- **Posts**: chunk as `title + body` together (usually fits in one chunk)
- **Comments**: thread-aware chunking — prepend parent context to each comment so chunks are self-contained
- Target chunk size: ~512 tokens with ~50 token overlap
- Use `tiktoken` to count tokens accurately

**Key decision**: full thread context per chunk vs flat comments. Full thread context improves retrieval quality but increases storage and embedding cost.

### 1.3 Embedding

- **Option A (API)**: `text-embedding-3-small` via OpenAI API — easier to set up
- **Option B (local/private)**: `sentence-transformers` with `all-MiniLM-L6-v2` or `bge-m3` — free and private
- Batch embed for throughput

### 1.4 Vector Database

- **Qdrant** (recommended) — self-hosted, Rust-based, fast, good metadata filtering support
- **pgvector** — good if a Postgres instance is already available
- **Chroma** — simpler setup, suitable for prototyping

### 1.5 Metadata Schema

Each vector should store:

```
post_id        string
comment_id     string (null for posts)
score          int
created_utc    int (Unix timestamp)
author         string
url            string
parent_id      string (null for top-level)
depth          int (0 for posts, 1+ for comments)
```

This metadata enables filtering by recency, score, or thread structure at query time.

---

## Phase 2: RAG Query Flow

```
User query
  → embed query
  → vector search (top-k chunks)
  → optionally filter by metadata (e.g. score > 10, date range)
  → fetch parent post for context if chunk is a comment
  → pass retrieved chunks to LLM with prompt
```

---

## Phase 3: NixOS / Dev Environment Setup

### Dev Shell (`flake.nix`)

```nix
devShells.default = pkgs.mkShell {
  packages = with pkgs; [
    python312
    python312Packages.praw
    python312Packages.sentence-transformers
    python312Packages.qdrant-client
    python312Packages.tiktoken
  ];
};
```

### Qdrant Service (`configuration.nix`)

```nix
services.qdrant = {
  enable = true;  # available in nixpkgs
};
```

Alternatively, run Qdrant via Docker during development.

---

## Key Design Decisions

| Decision | Options | Recommended |
|---|---|---|
| Embedding model | Local (`sentence-transformers`) vs API (OpenAI) | Local for privacy; API for ease |
| Incremental updates | Poll new posts periodically vs one-shot ingest | Periodic polling for ongoing freshness |
| Comment depth | Full threads vs top-level only | Full threads for better context |
| Deduplication | Hash post/comment content before inserting | Hash-based dedup to avoid duplicates |

---

## Implementation Order

1. [ ] Set up dev environment (`flake.nix` or `requirements.txt`)
2. [ ] Implement Reddit data collector (`collector.py`)
3. [ ] Implement chunker with thread-aware comment handling (`chunker.py`)
4. [ ] Implement embedder with batching (`embedder.py`)
5. [ ] Set up Qdrant (local service or Docker)
6. [ ] Implement vector store ingestion (`ingest.py`)
7. [ ] Implement RAG query interface (`query.py`)
8. [ ] Wire together into a CLI or service (`main.py`)
9. [ ] Add incremental update support (poll for new posts/comments)
10. [ ] Add deduplication logic (hash-based)
