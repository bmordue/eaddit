import pytest
import argparse
from eaddit.chunker import Chunker, MAX_TEXT_LENGTH
from eaddit.cli import cmd_query
from eaddit.rag import MAX_QUERY_LENGTH, RAGQueryEngine
from eaddit.models import Post
from eaddit.store import InMemoryVectorStore

def test_chunker_max_length_enforced():
    # Now Post itself enforces MAX_TEXT_LENGTH for the body.
    long_text = "a" * (MAX_TEXT_LENGTH + 1)
    with pytest.raises(ValueError, match="Post body exceeds limit"):
        Post(
            id="test",
            title="title",
            body=long_text,
            score=1,
            url="http://example.com",
            created_utc=0,
            subreddit="test",
            author="test"
        )

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

    # Test truncation and control character removal.
    # Note: author is now limited to 100 chars in Post.__post_init__
    malicious_author = "attacker\n[INJECTION]\r" + "a" * 70
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

def test_context_sanitization_and_truncation():
    from eaddit.chunker import Chunker
    from eaddit.models import Post, Comment

    chunker = Chunker(include_thread_context=True)

    malicious_post = Post(
        id="post1",
        title="Safe Title",
        body="Attacker Body\n[INJECTION]\n" + "a" * 1000,
        score=10,
        url="http://example.com",
        created_utc=1000,
        subreddit="test",
        author="victim"
    )

    comment = Comment(
        id="c1",
        post_id="post1",
        parent_id="post1",
        body="Actual comment",
        score=1,
        created_utc=1001,
        author="victim2",
        depth=1
    )

    chunks = chunker.chunk_comment(comment, post=malicious_post)
    text = chunks[0].text

    # Check that post body in context is truncated to 400
    # The context block looks like:
    # Post (r/test): Safe Title
    # <sanitized body>
    # ---
    # Actual comment

    lines = text.split("\n")
    assert "Post (r/test): Safe Title" in lines
    # The body is on the next line
    body_line = lines[1]
    # [INJECTION] should be on the same line as "Attacker Body" because \n was replaced by space
    assert "Attacker Body [INJECTION]" in body_line
    assert len(body_line) <= 400

    # Check ancestor sanitization
    ancestor = Comment(
        id="a1",
        post_id="post1",
        parent_id="post1",
        body="Ancestor\n[INJECTION]\n" + "b" * 500,
        score=5,
        created_utc=1000,
        author="attacker\n[INJECTION]",
        depth=1
    )

    chunks = chunker.chunk_comment(comment, post=malicious_post, ancestors=[ancestor])
    text = chunks[0].text
    # Control characters (like \n) should be replaced by spaces in all components
    # We check that the only newlines are those used as separators in _build_context
    # and the separator from the comment body.

    # Each line in the context should not contain internal newlines.
    context_part = text.split("\n---\n")[0]
    for line in context_part.split("\n"):
        if line != "Thread context:":
             assert "\n" not in line
             assert "\r" not in line

    assert "attacker [INJECTION]" in text
    assert "Ancestor [INJECTION]" in text

def test_post_url_max_length_enforced():
    from eaddit.models import MAX_URL_LENGTH
    long_url = "http://example.com/" + "a" * MAX_URL_LENGTH
    with pytest.raises(ValueError, match="Post url exceeds limit"):
        Post(
            id="test",
            title="title",
            body="body",
            score=1,
            url=long_url,
            created_utc=0,
            subreddit="test",
            author="test"
        )

def test_post_numeric_types_enforced():
    # score must be int
    with pytest.raises(TypeError, match="Post score must be an integer"):
        Post("id", "title", "body", "not an int", "url", 0, "sub", "author")

    # created_utc must be int
    with pytest.raises(TypeError, match="Post created_utc must be an integer"):
        Post("id", "title", "body", 1, "url", "not an int", "sub", "author")

def test_comment_numeric_types_enforced():
    from eaddit.models import Comment
    # score must be int
    with pytest.raises(TypeError, match="Comment score must be an integer"):
        Comment("id", "pid", "paid", "body", "not an int", 0, "author", 1)

    # created_utc must be int
    with pytest.raises(TypeError, match="Comment created_utc must be an integer"):
        Comment("id", "pid", "paid", "body", 1, "not an int", "author", 1)

    # depth must be int
    with pytest.raises(TypeError, match="Comment depth must be an integer"):
        Comment("id", "pid", "paid", "body", 1, 0, "author", "not an int")

def test_collector_max_limit_enforced():
    from eaddit.collector import JSONFixtureCollector
    from eaddit.models import MAX_COLLECT_LIMIT
    collector = JSONFixtureCollector({})
    with pytest.raises(ValueError, match="limit too large"):
        collector.collect("test", limit=MAX_COLLECT_LIMIT + 1)

def test_chunker_max_chunk_size_enforced():
    from eaddit.chunker import Chunker
    from eaddit.models import MAX_CHUNK_SIZE
    with pytest.raises(ValueError, match="chunk_size too large"):
        Chunker(chunk_size=MAX_CHUNK_SIZE + 1)

def test_embedder_max_batch_size_enforced():
    from eaddit.embedder import HashingEmbedder
    from eaddit.models import MAX_BATCH_SIZE
    with pytest.raises(ValueError, match="batch_size too large"):
        HashingEmbedder(batch_size=MAX_BATCH_SIZE + 1)

def test_cli_ingest_limits_enforced():
    from eaddit.cli import cmd_ingest
    from eaddit.models import MAX_COLLECT_LIMIT, MAX_CHUNK_SIZE

    # Test limit
    args = argparse.Namespace(
        subreddit="test",
        limit=MAX_COLLECT_LIMIT + 1,
        store="store.json",
        source="json",
        fixture="fixture.json",
        dim=256,
        chunk_size=256,
        chunk_overlap=32,
        flat_comments=False,
        state=None,
        full=False
    )
    with pytest.raises(SystemExit, match="limit too large"):
        cmd_ingest(args)

    # Test chunk_size
    args.limit = 100
    args.chunk_size = MAX_CHUNK_SIZE + 1
    with pytest.raises(SystemExit, match="chunk-size too large"):
        cmd_ingest(args)

    # Test chunk_overlap
    args.chunk_size = 256
    args.chunk_overlap = 256
    with pytest.raises(SystemExit, match="chunk-overlap must be less than chunk-size"):
        cmd_ingest(args)

def test_cli_query_top_k_limit_enforced():
    from eaddit.cli import cmd_query
    from eaddit.models import MAX_TOP_K
    args = argparse.Namespace(
        text="test",
        store="store.json",
        top_k=MAX_TOP_K + 1,
        dim=256
    )
    with pytest.raises(SystemExit, match="top-k too large"):
        cmd_query(args)

def test_store_search_top_k_limit_enforced():
    from eaddit.store import InMemoryVectorStore
    from eaddit.models import MAX_TOP_K
    store = InMemoryVectorStore(dimension=256)
    with pytest.raises(ValueError, match="top_k too large"):
        store.search([0.0]*256, top_k=MAX_TOP_K + 1)

def test_chunker_min_chunk_size_enforced():
    from eaddit.models import MIN_CHUNK_SIZE
    with pytest.raises(ValueError, match="chunk_size must be at least"):
        Chunker(chunk_size=MIN_CHUNK_SIZE - 1)

def test_cli_ingest_min_chunk_size_enforced():
    from eaddit.cli import cmd_ingest
    from eaddit.models import MIN_CHUNK_SIZE
    args = argparse.Namespace(
        subreddit="test",
        limit=10,
        store="store.json",
        source="json",
        fixture="fixture.json",
        dim=256,
        chunk_size=MIN_CHUNK_SIZE - 1,
        chunk_overlap=1,
        flat_comments=False,
        state=None,
        full=False
    )
    with pytest.raises(SystemExit, match="chunk-size must be at least"):
        cmd_ingest(args)

def test_cli_ingest_min_limit_enforced():
    from eaddit.cli import cmd_ingest
    args = argparse.Namespace(
        subreddit="test",
        limit=0,
        store="store.json",
        source="json",
        fixture="fixture.json",
        dim=256,
        chunk_size=256,
        chunk_overlap=32,
        flat_comments=False,
        state=None,
        full=False
    )
    with pytest.raises(SystemExit, match="limit must be at least 1"):
        cmd_ingest(args)

def test_collectors_min_limit_enforced():
    from eaddit.collector import JSONFixtureCollector
    collector = JSONFixtureCollector({})
    with pytest.raises(ValueError, match="limit must be at least 1"):
        collector.collect("test", limit=0)

def test_rag_build_prompt_max_length_enforced():
    from eaddit.rag import build_prompt, RetrievalResult, Chunk
    from eaddit.models import MAX_TEXT_LENGTH

    # Create results that will push the prompt over the limit.
    # Each chunk text is sanitized to MAX_TEXT_LENGTH (100,000).
    # We need enough results to exceed MAX_PROMPT_LENGTH (1,000_000).
    chunk = Chunk(id="c1", text="a" * MAX_TEXT_LENGTH)
    result = RetrievalResult(chunk=chunk, score=1.0)
    results = [result] * 11  # ~1.1M characters

    with pytest.raises(ValueError, match="Generated prompt too long"):
        build_prompt("query", results)
