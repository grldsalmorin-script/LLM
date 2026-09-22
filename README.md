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
