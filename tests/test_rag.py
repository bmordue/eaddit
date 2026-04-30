from eaddit.chunker import Chunker
from eaddit.embedder import HashingEmbedder
from eaddit.ingest import IngestionPipeline
from eaddit.rag import EchoLLM, RAGQueryEngine, build_prompt
from eaddit.store import InMemoryVectorStore


def _ingested_engine(collector):
    embedder = HashingEmbedder(dim=256)
    store = InMemoryVectorStore(dimension=embedder.dimension)
    pipeline = IngestionPipeline(
        collector=collector,
        chunker=Chunker(include_thread_context=False),  # makes hits more discriminative
        embedder=embedder,
        store=store,
    )
    pipeline.ingest("python", min_score=1, incremental=False)
    return RAGQueryEngine(store=store, embedder=embedder, llm=EchoLLM()), store


def test_retrieve_finds_relevant_chunk(collector):
    engine, _ = _ingested_engine(collector)
    results = engine.retrieve("how do I read large CSV files in python", top_k=3)
    assert results, "expected at least one result"
    top_ids = [r.chunk.id for r in results]
    # The CSV post or its comment should rank in the top-k.
    assert any(cid in {"post:p2", "comment:c3"} for cid in top_ids)


def test_metadata_filter_restricts_results(collector):
    engine, _ = _ingested_engine(collector)
    results = engine.retrieve(
        "python tips",
        top_k=10,
        metadata_filter=lambda m: m.get("post_id") == "p1",
    )
    assert results
    for r in results:
        assert r.chunk.metadata["post_id"] == "p1"


def test_ancestor_chain_attached_for_comment_hit(collector):
    engine, _ = _ingested_engine(collector)
    # Force the c2 -> c1 -> p1 chain by querying for c2's content directly.
    results = engine.retrieve(
        "Generator expressions are even better for huge inputs",
        top_k=1,
        attach_ancestors=True,
    )
    assert results[0].chunk.id == "comment:c2"
    ancestor_ids = [c.id for c in results[0].ancestors]
    # Root post first, then immediate parent comment.
    assert ancestor_ids == ["post:p1", "comment:c1"]


def test_no_ancestors_when_disabled(collector):
    engine, _ = _ingested_engine(collector)
    results = engine.retrieve(
        "Generator expressions are even better for huge inputs",
        top_k=1,
        attach_ancestors=False,
    )
    assert results[0].ancestors == []


def test_query_builds_prompt_and_calls_llm(collector):
    engine, _ = _ingested_engine(collector)
    answer = engine.query("how do I read large CSV files in python", top_k=2)
    # EchoLLM echoes the prompt.
    assert answer.prompt == answer.answer
    assert "Question: how do I read large CSV files in python" in answer.prompt
    assert len(answer.results) == 2


def test_build_prompt_includes_context_lines(collector):
    engine, _ = _ingested_engine(collector)
    results = engine.retrieve("Generator expressions", top_k=1)
    prompt = build_prompt("anything", results)
    assert "context:" in prompt
    assert "[Excerpt 1]" in prompt
