import math

import pytest

from eaddit.embedder import HashingEmbedder, cosine_similarity


def test_dimension_is_respected():
    emb = HashingEmbedder(dim=64)
    vecs = emb.embed(["hello world", "another text"])
    assert len(vecs) == 2
    for v in vecs:
        assert len(v) == 64


def test_embedding_is_deterministic():
    emb = HashingEmbedder(dim=128)
    a = emb.embed_one("Reddit RAG over comments")
    b = emb.embed_one("Reddit RAG over comments")
    assert a == b


def test_embedding_is_l2_normalised_for_nonempty_text():
    emb = HashingEmbedder(dim=128)
    v = emb.embed_one("python is great")
    norm = math.sqrt(sum(x * x for x in v))
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_empty_text_yields_zero_vector():
    emb = HashingEmbedder(dim=32)
    v = emb.embed_one("")
    assert v == [0.0] * 32


def test_similarity_higher_for_related_texts():
    emb = HashingEmbedder(dim=512)
    related_a = emb.embed_one("how to learn list comprehensions in python")
    related_b = emb.embed_one("python list comprehension tutorial for beginners")
    unrelated = emb.embed_one("the best chocolate chip cookie recipe ever")
    assert cosine_similarity(related_a, related_b) > cosine_similarity(related_a, unrelated)


def test_batching_does_not_change_results():
    e1 = HashingEmbedder(dim=64, batch_size=1)
    e2 = HashingEmbedder(dim=64, batch_size=64)
    texts = [f"text number {i}" for i in range(10)]
    assert e1.embed(texts) == e2.embed(texts)


def test_cosine_similarity_dimension_check():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_cosine_similarity_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_constructor_validates_args():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=0)
    with pytest.raises(ValueError):
        HashingEmbedder(batch_size=0)
