import pytest

from eaddit.chunker import Chunker
from eaddit.models import Comment, Post


def _post(**kw):
    base = dict(
        id="p1", title="title", body="body", score=1, url="u",
        created_utc=10, subreddit="sub", author="alice",
    )
    base.update(kw)
    return Post(**base)


def _comment(**kw):
    base = dict(
        id="c1", post_id="p1", parent_id="p1", body="hello",
        score=1, created_utc=11, author="bob", depth=1,
    )
    base.update(kw)
    return Comment(**base)


def test_chunk_post_combines_title_and_body():
    chunker = Chunker()
    chunks = chunker.chunk_post(_post(title="Hello", body="World"))
    assert len(chunks) == 1
    assert "Hello" in chunks[0].text and "World" in chunks[0].text
    assert chunks[0].metadata["post_id"] == "p1"
    assert chunks[0].metadata["comment_id"] is None
    assert chunks[0].content_hash  # not empty


def test_chunk_post_empty_body():
    chunker = Chunker()
    chunks = chunker.chunk_post(_post(title="title-only", body=""))
    assert len(chunks) == 1
    assert chunks[0].text == "title-only"


def test_chunk_post_blank_skipped():
    chunker = Chunker()
    assert chunker.chunk_post(_post(title="", body="")) == []


def test_chunk_comment_includes_thread_context():
    chunker = Chunker(include_thread_context=True)
    post = _post(title="My Post", body="post body")
    parent = _comment(id="c0", body="parent comment text", depth=1)
    child = _comment(
        id="c1", parent_id="c0", body="child reply", depth=2,
    )
    chunks = chunker.chunk_comment(child, post=post, ancestors=[parent])
    assert len(chunks) == 1
    text = chunks[0].text
    assert "My Post" in text
    assert "parent comment text" in text
    assert "child reply" in text
    assert chunks[0].metadata["comment_id"] == "c1"
    assert chunks[0].metadata["depth"] == 2


def test_chunk_comment_flat_mode_strips_context():
    chunker = Chunker(include_thread_context=False)
    post = _post(title="X", body="y")
    chunks = chunker.chunk_comment(_comment(body="just me"), post=post, ancestors=[])
    assert chunks[0].text == "just me"


def test_chunk_thread_runs_post_and_all_comments():
    chunker = Chunker()
    post = _post()
    c1 = _comment(id="c1", parent_id="p1", body="root", depth=1)
    c2 = _comment(id="c2", parent_id="c1", body="reply", depth=2)
    chunks = chunker.chunk_thread(post, [c2, c1])  # out-of-order on purpose
    ids = [c.id for c in chunks]
    assert "post:p1" in ids
    assert "comment:c1" in ids
    assert "comment:c2" in ids


def test_chunk_size_overlap_splits_long_text():
    chunker = Chunker(chunk_size=10, chunk_overlap=2)
    long_body = " ".join(f"word{i}" for i in range(35))
    post = _post(title="t", body=long_body)
    chunks = chunker.chunk_post(post)
    assert len(chunks) > 1
    # First chunk gets the canonical id; subsequent chunks are suffixed.
    assert chunks[0].id == "post:p1"
    assert chunks[1].id == "post:p1#1"
    # Each window respects the size budget (in tokens).
    for c in chunks:
        assert len(c.text.split()) <= 10
    # Consecutive windows overlap.
    a = chunks[0].text.split()[-2:]
    b = chunks[1].text.split()[:2]
    assert a == b


def test_chunker_validates_args():
    with pytest.raises(ValueError):
        Chunker(chunk_size=0)
    with pytest.raises(ValueError):
        Chunker(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        Chunker(chunk_size=10, chunk_overlap=-1)


def test_chunk_thread_handles_cycle_in_parent_chain():
    """Defensive: malformed data with parent cycles must not loop forever."""

    chunker = Chunker(max_ancestors=10)
    post = _post()
    # c1 -> c2 -> c1 (cycle)
    c1 = _comment(id="c1", parent_id="c2", depth=1, body="one")
    c2 = _comment(id="c2", parent_id="c1", depth=2, body="two")
    chunks = chunker.chunk_thread(post, [c1, c2])
    # Just assert we returned something for every input without hanging.
    assert any(c.id == "comment:c1" for c in chunks)
    assert any(c.id == "comment:c2" for c in chunks)
