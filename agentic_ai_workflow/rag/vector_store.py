from __future__ import annotations

import json
from pathlib import Path

from .embeddings import LocalHashingEmbedder
from .types import Chunk, RetrievedChunk


class LocalJsonVectorStore:
    """Persistent local vector store backed by a JSON file.

    This gives you a no-key local RAG flow: index documents once, persist vectors,
    then query them from the Flask API without calling external services.
    """

    def __init__(self, index_path: str | Path = "data/index/local_vectors.json", embedder: LocalHashingEmbedder | None = None) -> None:
        self.index_path = Path(index_path)
        self.embedder = embedder or LocalHashingEmbedder()
        self._rows: list[tuple[Chunk, list[float]]] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        existing_ids = {chunk.chunk_id for chunk, _ in self._rows}
        for chunk in chunks:
            if chunk.chunk_id in existing_ids:
                continue
            self._rows.append((chunk, self.embedder.embed(chunk.text)))

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedder": "LocalHashingEmbedder",
            "dimensions": self.embedder.dimensions,
            "rows": [
                {
                    "chunk": {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "text": chunk.text,
                        "start_word": chunk.start_word,
                        "end_word": chunk.end_word,
                    },
                    "vector": vector,
                }
                for chunk, vector in self._rows
            ],
        }
        self.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> bool:
        if not self.index_path.exists():
            return False
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        dimensions = int(payload.get("dimensions", self.embedder.dimensions))
        self.embedder = LocalHashingEmbedder(dimensions=dimensions)
        self._rows = [
            (
                Chunk(
                    chunk_id=row["chunk"]["chunk_id"],
                    source=row["chunk"]["source"],
                    text=row["chunk"]["text"],
                    start_word=int(row["chunk"]["start_word"]),
                    end_word=int(row["chunk"]["end_word"]),
                ),
                [float(value) for value in row["vector"]],
            )
            for row in payload.get("rows", [])
        ]
        return True

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = self.embedder.embed(query)
        scored = [RetrievedChunk(chunk=chunk, score=self.embedder.cosine(query_vector, vector)) for chunk, vector in self._rows]
        scored.sort(key=lambda row: row.score, reverse=True)
        return [row for row in scored[:top_k] if row.score > 0]
