from pathlib import Path

from agentic_ai_workflow.rag.pipeline import RagPipeline


def test_rag_pipeline_returns_citations():
    docs_dir = Path(__file__).resolve().parents[1] / "data" / "docs"
    pipeline = RagPipeline.from_directory(docs_dir, chunk_size=60, overlap=10, top_k=3)

    result = pipeline.query("What is the function registry used for?")

    assert result.answer
    assert result.citations
    assert any("vibo_ai_workflows" in citation["source"] for citation in result.citations)


def test_rag_pipeline_requires_question():
    docs_dir = Path(__file__).resolve().parents[1] / "data" / "docs"
    pipeline = RagPipeline.from_directory(docs_dir)

    try:
        pipeline.query("   ")
    except ValueError as exc:
        assert "question is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError for blank question")
