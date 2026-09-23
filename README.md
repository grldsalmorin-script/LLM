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
- Local RAG: document indexing, retrieval, answer generation, and citations

## Run the Workflow Demo

```powershell
uv run --no-project python -m agentic_ai_workflow.main
```

Expected output shows Jasmine receiving a user request, Copilot selecting a registered function, the command bus dispatching a Nicole review command, and a result event returning to Jasmine.

## RAG Providers

The RAG pipeline supports two modes:

| Provider | API key required | Uses network | Index file |
|---|---:|---:|---|
| `local` | No | No | `data/index/local_vectors.json` |
| `gemini` | Yes, `GEMINI_API_KEY` | Yes | `data/index/gemini_vectors.json` |

Google currently offers a Gemini API Free Tier for eligible accounts and models, but it is still quota/rate-limit based. Treat it as free for local testing, not unlimited production usage.

## Local No-Key RAG

Build or rebuild the local index:

```powershell
uv run --no-project python -m agentic_ai_workflow.rag.index_docs --provider local
```

Run the local RAG demo:

```powershell
uv run --no-project python -m agentic_ai_workflow.rag.demo
```

Run the Flask REST API locally:

```powershell
uv run --no-project --with Flask python -m agentic_ai_workflow.api
```

## Gemini RAG

Set your Gemini API key in PowerShell:

```powershell
$env:GEMINI_API_KEY="your_gemini_api_key_here"
$env:RAG_PROVIDER="gemini"
```

Build the Gemini vector index. This calls the Gemini embedding API for your local docs:

```powershell
uv run --no-project --with google-genai python -m agentic_ai_workflow.rag.index_docs --provider gemini
```

Run the Gemini-backed Flask API:

```powershell
uv run --no-project --with Flask --with google-genai python -m agentic_ai_workflow.api
```

Then test it from another terminal, also with the same environment variables set:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5055/api/rag/query -ContentType 'application/json' -Body '{"question":"What is the Function Registry used for?"}'
```

Optional Gemini model overrides:

```powershell
$env:GEMINI_EMBEDDING_MODEL="gemini-embedding-001"
$env:GEMINI_EMBEDDING_DIMENSIONS="768"
$env:GEMINI_GENERATION_MODEL="gemini-2.5-flash-lite"
```

## Tests

```powershell
uv run --no-project --with Flask --with pytest pytest
```

The default test suite runs in `local` mode and does not call Gemini.
