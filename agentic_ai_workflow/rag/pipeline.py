from __future__ import annotations

from pathlib import Path

from .answer_generator import LocalAnswerGenerator
from .chunker import TextChunker
from .loader import DocumentLoader
from .prompt_builder import PromptBuilder
from .retriever import Retriever
from .types import RagAnswer
from .vector_store import InMemoryVectorStore


class RagPipeline:
    """End-to-end local RAG pipeline."""

    def __init__(self, retriever: Retriever, prompt_builder: PromptBuilder, answer_generator: LocalAnswerGenerator) -> None:
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.answer_generator = answer_generator

    @classmethod
    def from_directory(cls, docs_dir: str | Path, chunk_size: int = 90, overlap: int = 20, top_k: int = 4) -> "RagPipeline":
        documents = DocumentLoader().load_directory(docs_dir)
        chunks = TextChunker(chunk_size=chunk_size, overlap=overlap).split(documents)
        vector_store = InMemoryVectorStore()
        vector_store.add_chunks(chunks)
        return cls(
            retriever=Retriever(vector_store, top_k=top_k),
            prompt_builder=PromptBuilder(),
            answer_generator=LocalAnswerGenerator(),
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
