from __future__ import annotations

from .embeddings import BagOfWordsEmbedder
from .types import Chunk, RetrievedChunk


class InMemoryVectorStore:
    """Stores chunk vectors and returns nearest chunks by cosine similarity."""

    def __init__(self, embedder: BagOfWordsEmbedder | None = None) -> None:
        self.embedder = embedder or BagOfWordsEmbedder()
        self._rows: list[tuple[Chunk, dict[str, float]]] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._rows.append((chunk, self.embedder.embed(chunk.text)))

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = self.embedder.embed(query)
        scored = [RetrievedChunk(chunk=chunk, score=self.embedder.cosine(query_vector, vector)) for chunk, vector in self._rows]
        scored.sort(key=lambda row: row.score, reverse=True)
        return [row for row in scored[:top_k] if row.score > 0]
