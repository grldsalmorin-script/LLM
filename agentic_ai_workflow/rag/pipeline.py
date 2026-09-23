from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from .answer_generator import ExtractiveAnswerGenerator, GeminiAnswerGenerator
from .chunker import TextChunker
from .embeddings import GeminiEmbedder, LocalHashingEmbedder
from .loader import DocumentLoader
from .prompt_builder import PromptBuilder
from .retriever import Retriever
from .types import RagAnswer
from .vector_store import LocalJsonVectorStore


class AnswerGenerator(Protocol):
    def answer(self, question: str, retrieved_chunks: list) -> str:
        ...


class RagPipeline:
    """End-to-end local RAG pipeline with local or Gemini providers."""

    def __init__(self, retriever: Retriever, prompt_builder: PromptBuilder, answer_generator: AnswerGenerator, provider: str) -> None:
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.answer_generator = answer_generator
        self.provider = provider

    @classmethod
    def from_directory(
        cls,
        docs_dir: str | Path,
        chunk_size: int = 90,
        overlap: int = 20,
        top_k: int = 4,
        index_path: str | Path | None = None,
        rebuild_index: bool = False,
        provider: str | None = None,
    ) -> "RagPipeline":
        resolved_provider = (provider or os.getenv("RAG_PROVIDER") or "local").strip().lower()
        if resolved_provider == "gemini":
            embedder = GeminiEmbedder(
                model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
                dimensions=int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768")),
            )
            answer_generator: AnswerGenerator = GeminiAnswerGenerator(
                model=os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash-lite")
            )
        elif resolved_provider == "local":
            embedder = LocalHashingEmbedder()
            answer_generator = ExtractiveAnswerGenerator()
        else:
            raise ValueError("RAG_PROVIDER must be 'local' or 'gemini'")

        resolved_index_path = Path(index_path or f"data/index/{resolved_provider}_vectors.json")
        vector_store = LocalJsonVectorStore(index_path=resolved_index_path, embedder=embedder)
        loaded = False if rebuild_index else vector_store.load()
        if not loaded:
            documents = DocumentLoader().load_directory(docs_dir)
            chunks = TextChunker(chunk_size=chunk_size, overlap=overlap).split(documents)
            vector_store.add_chunks(chunks)
            vector_store.save()

        return cls(
            retriever=Retriever(vector_store, top_k=top_k),
            prompt_builder=PromptBuilder(),
            answer_generator=answer_generator,
            provider=resolved_provider,
        )

    def query(self, question: str) -> RagAnswer:
        question = question.strip()
        if not question:
            raise ValueError("question is required")
        retrieved = self.retriever.retrieve(question)
        self.prompt_builder.build(question, retrieved)
        answer = self.answer_generator.answer(question, retrieved)
        citations = [
            {
                "source": item.chunk.source,
                "chunk_id": item.chunk.chunk_id,
                "score": round(item.score, 4),
                "start_word": item.chunk.start_word,
                "end_word": item.chunk.end_word,
            }
            for item in retrieved
        ]
        return RagAnswer(question=question, answer=answer, citations=citations, retrieved_chunks=retrieved)
