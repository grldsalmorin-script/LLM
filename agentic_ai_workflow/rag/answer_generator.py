from __future__ import annotations

import os
from typing import Any

from .embeddings import tokenize
from .types import RetrievedChunk


class ExtractiveAnswerGenerator:
    """Offline answer generator that extracts grounded sentences from retrieved chunks."""

    provider_name = "local"

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


class GeminiAnswerGenerator:
    """Gemini answer generator using retrieved chunks as grounded context."""

    provider_name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash-lite", api_key: str | None = None) -> None:
        self.model = model
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is required when RAG_PROVIDER=gemini")
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError("Install google-genai to use RAG_PROVIDER=gemini") from exc
        self.client = genai.Client(api_key=key)

    def answer(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "I do not have enough retrieved context to answer that. Add relevant documents or improve the query."

        context = "\n\n".join(
            f"Source: {item.chunk.source}\nChunk: {item.chunk.chunk_id}\nText: {item.chunk.text}"
            for item in retrieved_chunks
        )
        prompt = f"""You are answering from retrieved context only.
If the context is insufficient, say you do not have enough context.
Keep the answer concise and practical.

Question:
{question}

Retrieved context:
{context}
"""
        result: Any = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        text = getattr(result, "text", None)
        if text:
            return text.strip()
        candidates = getattr(result, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            texts = [getattr(part, "text", "") for part in parts]
            answer = "".join(texts).strip()
            if answer:
                return answer
        raise RuntimeError("Gemini response did not include answer text")
