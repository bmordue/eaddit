# eaddit

Ingest a subreddit into a vector database to power Retrieval-Augmented
Generation (RAG) queries.

`eaddit` is a small, dependency-light Python package that:

1. **collects** posts and comments from a subreddit (live via PRAW, or from a
   JSON fixture for hermetic testing/dev),
2. **chunks** them with thread-aware comment handling so each chunk is
   self-contained,
3. **embeds** the chunks with a deterministic in-process embedder (or, with
   the `embeddings` extra, a `sentence-transformers` model),
4. **stores** the resulting vectors and metadata in a tiny in-process vector
   store with JSON persistence,
5. **queries** the store with a RAG flow that supports metadata filters and
   thread-context expansion for comment hits.

The full design is captured in [PLAN.md](./PLAN.md) and the per-task notes in
[`.tsks/docs/`](./.tsks/docs).

## Installation

```bash
pip install -e .            # core, stdlib-only
pip install -e ".[reddit]"  # add PRAW for live Reddit access
pip install -e ".[embeddings]"  # add sentence-transformers
pip install -e ".[test]"    # add pytest
```

Python 3.10+ is required.

## Quick start (CLI)

The CLI is exposed both as the `eaddit` console script and as
`python -m eaddit`.

```bash
# 1. Ingest from a JSON fixture (no Reddit credentials needed).
eaddit ingest python \
    --source json --fixture examples/fixture.json \
    --store ./store.json \
    --state ./state.json \
    --min-score 1

# 2. Inspect the store.
eaddit info --store ./store.json

# 3. Run a RAG query (prints retrieved chunks as JSON by default,
#    or a ready-to-feed-an-LLM prompt with `--format prompt`).
eaddit query "how do I read large CSV files in python" \
    --store ./store.json --top-k 3 --format prompt
```

To collect live data instead, use `--source reddit` and provide
`--client-id`, `--client-secret`, and `--user-agent` (PRAW required).

## Quick start (Python API)

```python
from eaddit import (
    Chunker, HashingEmbedder, InMemoryVectorStore,
    IngestionPipeline, RAGQueryEngine,
)
from eaddit.collector import JSONFixtureCollector

collector = JSONFixtureCollector("examples/fixture.json")
embedder  = HashingEmbedder(dim=256)
store     = InMemoryVectorStore(dimension=embedder.dimension)

pipeline = IngestionPipeline(
    collector=collector,
    chunker=Chunker(chunk_size=256, chunk_overlap=32),
    embedder=embedder,
    store=store,
    state_path="./state.json",
)
pipeline.ingest("python", min_score=1)

engine  = RAGQueryEngine(store=store, embedder=embedder)
results = engine.retrieve("how do I read large CSV files", top_k=3)
for r in results:
    print(f"{r.score:.3f}  {r.chunk.id}  {r.chunk.text[:80]}")
```

## Architecture

See [PLAN.md](./PLAN.md) for the full diagram and design decisions. In short:

```
collector → chunker → embedder → vector store
                                      ↑
                           query embedder ──→ retrieve top-k → build prompt → LLM
```

Each stage has a small, swappable interface:

| Layer       | Interface                | Default impl                |
|-------------|--------------------------|-----------------------------|
| Collector   | `eaddit.collector.Collector`           | `JSONFixtureCollector` (also `PRAWCollector`) |
| Chunker     | `eaddit.chunker.Chunker`               | thread-aware, configurable size/overlap       |
| Embedder    | `eaddit.embedder.Embedder`             | `HashingEmbedder` (also `SentenceTransformerEmbedder`) |
| Vector store| —                                      | `eaddit.store.InMemoryVectorStore` (JSON-persistent)   |
| RAG engine  | `eaddit.rag.RAGQueryEngine`            | top-k cosine + ancestor expansion + prompt builder      |

## Cross-cutting features

* **Deduplication.** Each chunk carries a SHA-256 `content_hash`. The
  ingestion pipeline skips chunks whose id+hash already exist in the store, so
  re-runs on unchanged data are no-ops.
* **Incremental updates.** When a `state_path` is configured, the pipeline
  records the highest `created_utc` it has seen per subreddit and skips older
  posts on subsequent runs. Pass `incremental=False` (or `--full`) to ignore
  the watermark.
* **Metadata filters.** `RAGQueryEngine.retrieve` and the `query` CLI accept
  a metadata predicate (programmatically) or `--min-score / --since / --until`
  flags (CLI) to restrict results by score or recency.
* **Ancestor expansion.** When a comment chunk is retrieved, the engine walks
  its `parent_id` chain back to the root post and attaches each ancestor
  chunk as additional context — so the prompt always includes thread context
  even if only one comment was retrieved.

## Development

```bash
pip install -e ".[test]"
pytest
```

The test suite is hermetic: it uses the JSON fixture collector and the
deterministic hashing embedder, so it runs anywhere with no network access
and no external models.

### Task tracking

This project uses [tssk](https://github.com/bmordue/tssk) for task tracking.
Tasks are stored in `.tssk.json` and `.tsks/`.

```bash
tssk list              # View all tasks
tssk list --status todo
tssk show <id>
tssk status <id> done
```
