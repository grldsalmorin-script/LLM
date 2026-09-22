from __future__ import annotations

from .embeddings import tokenize
from .types import RetrievedChunk


class LocalAnswerGenerator:
    """Deterministic answer generator for local runs.

    A production version can call Gemini/OpenAI here. The rest of the RAG pipeline
    remains unchanged because this class only consumes question + retrieved chunks.
    """

    def answer(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "I do not have enough retrieved context to answer that. Add relevant documents or improve the query."

        question_terms = set(tokenize(question))
        selected_sentences: list[str] = []
        for item in retrieved_chunks:
            sentences = [segment.strip() for segment in item.chunk.text.replace("\n", " ").split(".") if segment.strip()]
            for sentence in sentences:
                if question_terms.intersection(tokenize(sentence)):
                    selected_sentences.append(sentence)
                if len(selected_sentences) >= 4:
                    break
            if len(selected_sentences) >= 4:
                break

        if not selected_sentences:
            selected_sentences = [retrieved_chunks[0].chunk.text]

        return " ".join(selected_sentences) + "."
