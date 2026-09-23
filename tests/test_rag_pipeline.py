from pathlib import Path

from agentic_ai_workflow.rag.pipeline import RagPipeline
from agentic_ai_workflow.rag.vector_store import LocalJsonVectorStore


def test_rag_pipeline_returns_citations(tmp_path):
    docs_dir = Path(__file__).resolve().parents[1] / "data" / "docs"
    index_path = tmp_path / "local_vectors.json"
    pipeline = RagPipeline.from_directory(docs_dir, chunk_size=60, overlap=10, top_k=3, index_path=index_path)

    result = pipeline.query("What is the function registry used for?")

    assert result.answer
    assert result.citations
    assert index_path.exists()
    assert any("vibo_ai_workflows" in citation["source"] for citation in result.citations)


def test_rag_pipeline_loads_existing_local_index(tmp_path):
    docs_dir = Path(__file__).resolve().parents[1] / "data" / "docs"
    index_path = tmp_path / "local_vectors.json"
    RagPipeline.from_directory(docs_dir, index_path=index_path, rebuild_index=True)

    vector_store = LocalJsonVectorStore(index_path=index_path)

    assert vector_store.load() is True
    assert vector_store.search("function registry", top_k=1)


def test_rag_pipeline_requires_question(tmp_path):
    docs_dir = Path(__file__).resolve().parents[1] / "data" / "docs"
    pipeline = RagPipeline.from_directory(docs_dir, index_path=tmp_path / "local_vectors.json")

    try:
        pipeline.query("   ")
    except ValueError as exc:
        assert "question is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError for blank question")
