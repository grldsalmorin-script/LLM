from __future__ import annotations

from .embeddings import tokenize
from .types import RetrievedChunk


class ExtractiveAnswerGenerator:
    """Offline answer generator that extracts grounded sentences from retrieved chunks."""

    def answer(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "I do not have enough retrieved context to answer that. Add relevant documents or improve the query."

        question_terms = set(tokenize(question))
        selected_sentences: list[str] = []
        seen: set[str] = set()

        for item in retrieved_chunks:
            sentences = [segment.strip() for segment in item.chunk.text.replace("\n", " ").split(".") if segment.strip()]
            ranked = sorted(
                sentences,
                key=lambda sentence: len(question_terms.intersection(tokenize(sentence))),
                reverse=True,
            )
            for sentence in ranked:
                if sentence in seen:
                    continue
                if question_terms.intersection(tokenize(sentence)) or not selected_sentences:
                    selected_sentences.append(sentence)
                    seen.add(sentence)
                if len(selected_sentences) >= 4:
                    break
            if len(selected_sentences) >= 4:
                break

        return " ".join(selected_sentences).rstrip(".") + "."
