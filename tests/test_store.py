import pytest

from eaddit.embedder import HashingEmbedder
from eaddit.models import Chunk
from eaddit.store import InMemoryVectorStore


def _make_store(chunks_with_text):
    emb = HashingEmbedder(dim=128)
    store = InMemoryVectorStore(dimension=emb.dimension)
    chunks = [
        Chunk(id=cid, text=text, metadata=meta or {})
        for cid, text, meta in chunks_with_text
    ]
    vectors = emb.embed([c.text for c in chunks])
    store.add(chunks, vectors)
    return store, emb


def test_add_and_search_returns_best_match():
    store, emb = _make_store([
        ("a", "python list comprehensions are great", {"score": 10}),
        ("b", "go channels and concurrency primer", {"score": 5}),
        ("c", "best chocolate cookie recipe", {"score": 1}),
    ])
    q = emb.embed_one("learn python comprehensions")
    results = store.search(q, top_k=2)
    assert results[0].chunk.id == "a"
    assert len(results) == 2


def test_metadata_filter_excludes_chunks():
    store, emb = _make_store([
        ("a", "python text one", {"score": 1}),
        ("b", "python text two", {"score": 100}),
    ])
    q = emb.embed_one("python text")
    results = store.search(q, top_k=5, metadata_filter=lambda m: m.get("score", 0) >= 50)
    assert [r.chunk.id for r in results] == ["b"]


def test_re_adding_chunk_overwrites():
    store, _ = _make_store([("a", "first", {"v": 1})])
    new_chunk = Chunk(id="a", text="updated", metadata={"v": 2})
    store.add([new_chunk], [[0.0] * store.dimension])
    assert len(store) == 1
    assert store.get("a").metadata["v"] == 2


def test_delete_removes_chunk():
    store, _ = _make_store([("a", "x", None), ("b", "y", None)])
    assert store.delete("a") is True
    assert store.delete("a") is False
    assert "a" not in store
    assert {c.id for c in store.all_chunks()} == {"b"}


def test_dimension_validation_on_add():
    store = InMemoryVectorStore(dimension=4)
    with pytest.raises(ValueError):
        store.add([Chunk(id="x", text="t")], [[1.0, 2.0]])  # wrong dim
    with pytest.raises(ValueError):
        store.add([Chunk(id="x", text="t")], [])  # mismatched lengths


def test_dimension_validation_on_search():
    store = InMemoryVectorStore(dimension=4)
    with pytest.raises(ValueError):
        store.search([1.0, 2.0])


def test_search_with_zero_top_k_returns_empty():
    store, emb = _make_store([("a", "hi", None)])
    assert store.search(emb.embed_one("hi"), top_k=0) == []


def test_save_and_load_round_trip(tmp_path):
    store, emb = _make_store([
        ("a", "python rocks", {"score": 7, "post_id": "p1"}),
        ("b", "go rocks too", {"score": 9, "post_id": "p2"}),
    ])
    path = tmp_path / "store.json"
    store.save(path)
    loaded = InMemoryVectorStore.load(path)
    assert loaded.dimension == store.dimension
    assert len(loaded) == 2
    res = loaded.search(emb.embed_one("python rocks"), top_k=1)
    assert res[0].chunk.id == "a"
    assert res[0].chunk.metadata["score"] == 7


def test_constructor_validates_dimension():
    with pytest.raises(ValueError):
        InMemoryVectorStore(dimension=0)
