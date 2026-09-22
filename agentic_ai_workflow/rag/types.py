from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    start_word: int
    end_word: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    citations: list[dict]
    retrieved_chunks: list[RetrievedChunk]
