from agentic_ai_workflow.api import create_app


def test_rag_api_query_returns_answer():
    app = create_app()
    client = app.test_client()

    response = client.post("/api/rag/query", json={"question": "How should AI accounting recommendations be handled?"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["answer"]
    assert data["citations"]
    assert data["provider"] == "local"


def test_rag_api_health_reports_provider_and_index():
    app = create_app()
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    data = response.get_json()
    assert data["provider"] == "local"
    assert data["index_path"].endswith("local_vectors.json")


def test_rag_api_rejects_blank_question():
    app = create_app()
    client = app.test_client()

    response = client.post("/api/rag/query", json={"question": ""})

    assert response.status_code == 400
    assert response.get_json()["error"] == "question is required"
