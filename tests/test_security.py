import pytest
import argparse
from eaddit.chunker import Chunker, MAX_TEXT_LENGTH
from eaddit.cli import cmd_query
from eaddit.rag import MAX_QUERY_LENGTH, RAGQueryEngine
from eaddit.models import Post
from eaddit.store import InMemoryVectorStore

def test_chunker_max_length_enforced():
    chunker = Chunker()
    # Create a post with text exceeding the limit
    long_text = "a" * (MAX_TEXT_LENGTH + 1)
    post = Post(
        id="test",
        title="title",
        body=long_text,
        score=1,
        url="http://example.com",
        created_utc=0,
        subreddit="test",
        author="test"
    )

    with pytest.raises(ValueError, match="Input text too long"):
        chunker.chunk_post(post)

def test_cli_query_max_length_enforced():
    args = argparse.Namespace(
        text="a" * (MAX_QUERY_LENGTH + 1),
        dim=256,
        store="nonexistent_store.json"
    )

    with pytest.raises(SystemExit, match="Query too long"):
        cmd_query(args)

def test_rag_engine_query_max_length_enforced():
    from eaddit.embedder import HashingEmbedder
    store = InMemoryVectorStore(dimension=256)
    embedder = HashingEmbedder(dim=256)
    engine = RAGQueryEngine(store=store, embedder=embedder)

    long_text = "a" * (MAX_QUERY_LENGTH + 1)
    with pytest.raises(ValueError, match="Query too long"):
        engine.retrieve(long_text)


def test_rag_engine_max_top_k_enforced():
    from eaddit.embedder import HashingEmbedder
    from eaddit.rag import MAX_TOP_K
    store = InMemoryVectorStore(dimension=256)
    embedder = HashingEmbedder(dim=256)
    engine = RAGQueryEngine(store=store, embedder=embedder)

    with pytest.raises(ValueError, match="top_k too large"):
        engine.retrieve("test", top_k=MAX_TOP_K + 1)


def test_rag_engine_max_ancestors_enforced():
    from eaddit.embedder import HashingEmbedder
    from eaddit.rag import MAX_ANCESTORS
    store = InMemoryVectorStore(dimension=256)
    embedder = HashingEmbedder(dim=256)
    engine = RAGQueryEngine(store=store, embedder=embedder)

    with pytest.raises(ValueError, match="max_ancestors too large"):
        engine.retrieve("test", max_ancestors=MAX_ANCESTORS + 1)


def test_chunker_max_ancestors_enforced():
    from eaddit.chunker import MAX_ANCESTORS
    with pytest.raises(ValueError, match="max_ancestors too large"):
        Chunker(max_ancestors=MAX_ANCESTORS + 1)


def test_metadata_sanitization():
    from eaddit.models import post_metadata, Post

    # Test truncation and control character removal
    malicious_author = "attacker\n[INJECTION]\r" + "a" * 500
    post = Post(
        id="test_id",
        title="title",
        body="body",
        score=1,
        url="http://example.com",
        created_utc=0,
        subreddit="test_sub",
        author=malicious_author
    )

    meta = post_metadata(post)
    author = meta["author"]

    # Should be truncated to 200
    assert len(author) <= 200
    # Should not contain newlines or carriage returns
    assert "\n" not in author
    assert "\r" not in author
    # Should be stripped
    assert author.endswith("a")
