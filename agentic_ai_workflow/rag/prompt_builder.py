from __future__ import annotations

from .types import RetrievedChunk


class PromptBuilder:
    """Builds a grounded prompt from retrieved context."""

    def build(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        context_blocks = []
        for index, item in enumerate(retrieved_chunks, start=1):
            context_blocks.append(
                f"[source {index}: {item.chunk.source}, score={item.score:.3f}]\n{item.chunk.text}"
            )
        context = "\n\n".join(context_blocks) or "No relevant context found."
        return (
            "Answer the user using only the retrieved context. "
            "If the context is insufficient, say what is missing.\n\n"
            f"Question:\n{question}\n\nRetrieved context:\n{context}\n"
        )
