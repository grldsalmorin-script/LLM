from agentic_ai_workflow.api import create_app


def test_rag_api_query_returns_answer():
    app = create_app()
    client = app.test_client()

    response = client.post("/api/rag/query", json={"question": "How should AI accounting recommendations be handled?"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["answer"]
    assert data["citations"]


def test_rag_api_rejects_blank_question():
    app = create_app()
    client = app.test_client()

    response = client.post("/api/rag/query", json={"question": ""})

    assert response.status_code == 400
    assert response.get_json()["error"] == "question is required"
