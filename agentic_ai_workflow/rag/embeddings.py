from __future__ import annotations

import hashlib
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


class LocalHashingEmbedder:
    """Deterministic local embedder that does not require API keys.

    It maps tokens into a fixed-size vector using feature hashing. This is not as
    semantically rich as OpenAI/Gemini embeddings, but it behaves like a real
    embedding provider from the pipeline's perspective and works fully offline.
    """

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
