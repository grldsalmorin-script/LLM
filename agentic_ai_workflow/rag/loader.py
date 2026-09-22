from __future__ import annotations

from pathlib import Path

from .types import Document

SUPPORTED_SUFFIXES = {".md", ".txt"}


class DocumentLoader:
    """Loads local markdown/text documents for RAG indexing."""

    def load_directory(self, docs_dir: str | Path) -> list[Document]:
        root = Path(docs_dir)
        if not root.exists():
            raise FileNotFoundError(f"Document directory not found: {root}")

        documents: list[Document] = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                documents.append(Document(source=str(path.relative_to(root)), text=path.read_text(encoding="utf-8")))
        if not documents:
            raise ValueError(f"No supported documents found in: {root}")
        return documents
