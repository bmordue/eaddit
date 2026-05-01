import json

import pytest

from eaddit.chunker import Chunker
from eaddit.embedder import HashingEmbedder
from eaddit.ingest import IngestionPipeline
from eaddit.store import InMemoryVectorStore


def _build(collector, tmp_path, **kwargs):
    embedder = HashingEmbedder(dim=128)
    store = InMemoryVectorStore(dimension=embedder.dimension)
    return IngestionPipeline(
        collector=collector,
        chunker=Chunker(),
        embedder=embedder,
        store=store,
        state_path=str(tmp_path / "state.json"),
        **kwargs,
    ), store


def test_ingest_populates_store(collector, tmp_path):
    pipeline, store = _build(collector, tmp_path)
    stats = pipeline.ingest("python", min_score=1)
    assert stats.posts_seen == 2  # p3 filtered for low score
    assert stats.comments_seen == 3  # c4 has score=-3
    assert stats.chunks_added > 0
    assert stats.chunks_added == stats.chunks_emitted
    assert len(store) == stats.chunks_added


def test_ingest_is_idempotent(collector, tmp_path):
    pipeline, _ = _build(collector, tmp_path)
    first = pipeline.ingest("python", min_score=1, incremental=False)
    second = pipeline.ingest("python", min_score=1, incremental=False)
    assert first.chunks_added > 0
    assert second.chunks_added == 0
    assert second.chunks_skipped_duplicate == second.chunks_emitted


def test_incremental_skips_already_seen_posts(collector, tmp_path):
    pipeline, _ = _build(collector, tmp_path)
    pipeline.ingest("python", min_score=1)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["python"]["last_created_utc"] == 1_700_000_500

    # Second run sees no new posts.
    stats = pipeline.ingest("python", min_score=1)
    assert stats.posts_seen == 0
    assert stats.chunks_added == 0


def test_full_run_ignores_watermark(collector, tmp_path):
    pipeline, _ = _build(collector, tmp_path)
    pipeline.ingest("python", min_score=1)
    stats = pipeline.ingest("python", min_score=1, incremental=False)
    # Re-runs yield zero added because of dedup, but they re-process every post.
    assert stats.posts_seen == 2
    assert stats.chunks_skipped_duplicate > 0


def test_empty_subreddit_returns_zero_stats(collector, tmp_path):
    pipeline, _ = _build(collector, tmp_path)
    stats = pipeline.ingest("does-not-exist")
    assert stats.posts_seen == 0
    assert stats.chunks_added == 0


def test_dimension_mismatch_is_rejected(collector, tmp_path):
    embedder = HashingEmbedder(dim=64)
    store = InMemoryVectorStore(dimension=128)
    with pytest.raises(ValueError):
        IngestionPipeline(
            collector=collector,
            chunker=Chunker(),
            embedder=embedder,
            store=store,
        )


def test_state_file_recovers_from_corruption(collector, tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("not valid json")
    pipeline, _ = _build(collector, tmp_path)
    # Should treat corrupt state as empty and ingest everything.
    stats = pipeline.ingest("python", min_score=1)
    assert stats.posts_seen == 2
