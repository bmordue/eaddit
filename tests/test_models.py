from eaddit.models import (
    Chunk,
    Comment,
    Post,
    comment_metadata,
    post_metadata,
)


def test_post_to_dict_round_trip():
    p = Post(
        id="p1",
        title="t",
        body="b",
        score=1,
        url="u",
        created_utc=10,
        subreddit="s",
        author="a",
    )
    d = p.to_dict()
    assert d["id"] == "p1"
    assert d["score"] == 1


def test_comment_to_dict():
    c = Comment(
        id="c1",
        post_id="p1",
        parent_id="p1",
        body="x",
        score=2,
        created_utc=11,
        author="a",
        depth=1,
    )
    assert c.to_dict()["depth"] == 1


def test_chunk_round_trip():
    chunk = Chunk(
        id="post:p1",
        text="hello world",
        metadata={"post_id": "p1", "score": 7},
        content_hash="abc",
    )
    restored = Chunk.from_dict(chunk.to_dict())
    assert restored == chunk
    # Independent metadata copy.
    chunk.metadata["score"] = 99
    assert restored.metadata["score"] == 7


def test_post_metadata_shape():
    p = Post("p", "t", "b", 5, "u", 10, "sub", "alice")
    meta = post_metadata(p)
    assert meta == {
        "post_id": "p",
        "comment_id": None,
        "score": 5,
        "created_utc": 10,
        "author": "alice",
        "url": "u",
        "parent_id": None,
        "depth": 0,
        "subreddit": "sub",
    }


def test_comment_metadata_includes_post_url():
    p = Post("p", "t", "b", 5, "u", 10, "sub", "alice")
    c = Comment("c", "p", "p", "body", 1, 11, "bob", 1)
    meta = comment_metadata(c, post=p)
    assert meta["url"] == "u"
    assert meta["subreddit"] == "sub"
    assert meta["depth"] == 1
    # Without a post, url and subreddit are None.
    meta = comment_metadata(c, post=None)
    assert meta["url"] is None
    assert meta["subreddit"] is None
