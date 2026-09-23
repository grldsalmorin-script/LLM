# Agentic AI Workflow Sample

This is a small, runnable reference implementation for explaining the Vibo-style AI workflow architecture in interviews.

It models four logical services:

- Jasmine: user-facing assistant/orchestrator
- Copilot: workflow planner that chooses registered tools
- Gemini: LLM adapter stand-in that returns structured JSON-like output
- Nicole: accounting recommendation/review service

The sample also includes:

- Function Registry: contract catalog for tools/functions
- Command Bus: in-memory stand-in for Pub/Sub
- Async Handler: command consumer that runs handlers and publishes result events
- Context Provider: structured context retrieval
- Local RAG: offline document indexing, retrieval, answer generation, and citations

## Run the Workflow Demo

```powershell
uv run --no-project python -m agentic_ai_workflow.main
```

Expected output shows Jasmine receiving a user request, Copilot selecting a registered function, the command bus dispatching a Nicole review command, and a result event returning to Jasmine.

## Local No-Key RAG Implementation

This repo includes a complete local RAG pipeline that does not require OpenAI, Gemini, Vertex AI, Pinecone, Chroma, Postgres, or any API key.

1. `DocumentLoader` reads local `.md` and `.txt` files from `data/docs`.
2. `TextChunker` splits documents into overlapping chunks.
3. `LocalHashingEmbedder` creates deterministic local vectors.
4. `LocalJsonVectorStore` persists vectors to `data/index/local_vectors.json`.
5. `Retriever` returns top matching chunks.
6. `PromptBuilder` builds a grounded context prompt.
7. `ExtractiveAnswerGenerator` creates a deterministic local answer from retrieved text.
8. `RagPipeline` returns the answer plus citations.

Build or rebuild the local index:

```powershell
uv run --no-project python -m agentic_ai_workflow.rag.index_docs
```

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

## Optional Production Upgrade Path

The default implementation is intentionally offline. Later, if you want managed AI services, replace these provider classes while keeping the same `RagPipeline` shape:

- `LocalHashingEmbedder` -> OpenAI, Gemini, Vertex AI, or sentence-transformer embeddings
- `LocalJsonVectorStore` -> pgvector, Qdrant, Pinecone, Weaviate, Chroma, or Elasticsearch
- `ExtractiveAnswerGenerator` -> Gemini/OpenAI/local LLM answer generator
