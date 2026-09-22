from __future__ import annotations

import re

from .types import Chunk, Document

_WORD_RE = re.compile(r"\S+")


class TextChunker:
    """Splits documents into overlapping word chunks."""

    def __init__(self, chunk_size: int = 90, overlap: int = 20) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            words = [match.group(0) for match in _WORD_RE.finditer(doc.text)]
            if not words:
                continue
            step = self.chunk_size - self.overlap
            start = 0
            chunk_number = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_words = words[start:end]
                chunk_id = f"{doc.source}::chunk-{chunk_number}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source=doc.source,
                        text=" ".join(chunk_words),
                        start_word=start,
                        end_word=end,
                    )
                )
                if end == len(words):
                    break
                start += step
                chunk_number += 1
        return chunks
