from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "with", "without",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOPWORDS]


class BagOfWordsEmbedder:
    """Small deterministic embedding substitute for local RAG demos.

    This is intentionally dependency-free. Swap this class for OpenAI/Vertex
    embeddings in production while keeping the Retriever and RagPipeline intact.
    """

    def embed(self, text: str) -> dict[str, float]:
        counts = Counter(tokenize(text))
        norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
        return {token: value / norm for token, value in counts.items()}

    @staticmethod
    def cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(weight * right.get(token, 0.0) for token, weight in left.items())
