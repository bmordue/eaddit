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
