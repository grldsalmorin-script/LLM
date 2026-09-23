from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from typing import Any

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "with", "without",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOPWORDS]


class LocalHashingEmbedder:
    """Deterministic local embedder that does not require API keys."""

    provider_name = "local"

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        counts = Counter(tokenize(text))
        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * float(count)

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    @staticmethod
    def cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("vectors must have the same dimensions")
        return sum(a * b for a, b in zip(left, right))


class GeminiEmbedder:
    """Gemini embedding provider using GEMINI_API_KEY.

    Import is lazy so local tests and offline mode do not require google-genai.
    """

    provider_name = "gemini"

    def __init__(self, model: str = "gemini-embedding-001", dimensions: int = 768, api_key: str | None = None) -> None:
        self.model = model
        self.dimensions = dimensions
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is required when RAG_PROVIDER=gemini")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError("Install google-genai to use RAG_PROVIDER=gemini") from exc
        self._types = types
        self.client = genai.Client(api_key=key)

    def embed(self, text: str) -> list[float]:
        config = self._types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=self.dimensions,
        )
        result: Any = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=config,
        )
        embeddings = getattr(result, "embeddings", None)
        if embeddings:
            values = getattr(embeddings[0], "values", embeddings[0])
            return [float(value) for value in values]
        embedding = getattr(result, "embedding", None)
        values = getattr(embedding, "values", embedding)
        if values is None:
            raise RuntimeError("Gemini embedding response did not include vector values")
        return [float(value) for value in values]

    @staticmethod
    def cosine(left: list[float], right: list[float]) -> float:
        return LocalHashingEmbedder.cosine(left, right)
