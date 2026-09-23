from __future__ import annotations

import json
import os
from pathlib import Path

from .pipeline import RagPipeline


def main() -> None:
    docs_dir = Path(__file__).resolve().parents[2] / "data" / "docs"
    pipeline = RagPipeline.from_directory(docs_dir, provider=os.getenv("RAG_PROVIDER", "local"))
    result = pipeline.query("How do Jasmine, Copilot, Gemini, and Nicole work with function registry commands?")
    print(json.dumps({
        "provider": pipeline.provider,
        "question": result.question,
        "answer": result.answer,
        "citations": result.citations,
    }, indent=2))


if __name__ == "__main__":
    main()
