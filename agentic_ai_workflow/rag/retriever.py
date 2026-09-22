from __future__ import annotations

from .types import RetrievedChunk
from .vector_store import InMemoryVectorStore


class Retriever:
    def __init__(self, vector_store: InMemoryVectorStore, top_k: int = 4) -> None:
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        return self.vector_store.search(question, top_k=self.top_k)
