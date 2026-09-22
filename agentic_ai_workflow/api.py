from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request

from agentic_ai_workflow.rag.pipeline import RagPipeline


def create_app(docs_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    resolved_docs_dir = Path(docs_dir or os.getenv("RAG_DOCS_DIR") or Path(__file__).resolve().parents[1] / "data" / "docs")
    pipeline = RagPipeline.from_directory(resolved_docs_dir)

    @app.get("/health")
    def health():
        return jsonify({"status": "healthy", "docs_dir": str(resolved_docs_dir)}), 200

    @app.post("/api/rag/query")
    def rag_query():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question") or "").strip()
        if not question:
            return jsonify({"error": "question is required"}), 400
        try:
            result = pipeline.query(question)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        return jsonify({
            "question": result.question,
            "answer": result.answer,
            "citations": result.citations,
        }), 200

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5055, debug=True)
