# Agentic AI Workflow Sample

This is a small, runnable reference implementation for explaining the Vibo-style AI workflow architecture in interviews.

It models four logical services:

- Jasmine: user-facing assistant/orchestrator
- Copilot: workflow planner that chooses registered tools
- Gemini: LLM adapter that returns structured JSON-like output
- Nicole: accounting recommendation/review service

The sample also includes:

- Function Registry: contract catalog for tools/functions
- Command Bus: in-memory stand-in for Pub/Sub
- Async Handler: command consumer that runs handlers and publishes result events
- Context Provider: structured context retrieval, plus an optional RAG-like document snippet path

Important: this is intentionally not a full production RAG stack. It demonstrates context retrieval and prompt grounding. A full RAG implementation would add chunking, embeddings, vector search, reranking, and source citation enforcement.

## Run

```powershell
python -m agentic_ai_workflow.main
```

Expected output shows Jasmine receiving a user request, Copilot selecting a registered function, the command bus dispatching a Nicole review command, and a result event returning to Jasmine.
## Local RAG Implementation

This repo now includes a complete local RAG pipeline:

1. `DocumentLoader` reads local `.md` and `.txt` files from `data/docs`.
2. `TextChunker` splits documents into overlapping chunks.
3. `BagOfWordsEmbedder` vectorizes chunks without external dependencies.
4. `InMemoryVectorStore` performs cosine-similarity search.
5. `Retriever` returns top matching chunks.
6. `PromptBuilder` builds a grounded context prompt.
7. `LocalAnswerGenerator` creates a deterministic local answer.
8. `RagPipeline` returns the answer plus citations.

Run the RAG demo:

```powershell
uv run --no-project python -m agentic_ai_workflow.rag.demo
```

Run the Flask REST API locally:

```powershell
uv run --no-project --with Flask python -m agentic_ai_workflow.api
```

Then test it from another terminal:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5055/api/rag/query -ContentType 'application/json' -Body '{"question":"What is the Function Registry used for?"}'
```

Run tests:

```powershell
uv run --no-project --with Flask --with pytest pytest
```

Production upgrade path:

- Replace `BagOfWordsEmbedder` with OpenAI or Vertex AI embeddings.
- Replace `InMemoryVectorStore` with pgvector, Qdrant, Pinecone, Weaviate, or Elasticsearch.
- Replace `LocalAnswerGenerator` with Gemini/OpenAI while preserving retrieved citations.
- Add document ingestion jobs for PDFs, web pages, Google Drive, or app data.
