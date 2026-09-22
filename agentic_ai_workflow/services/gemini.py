from __future__ import annotations


class GeminiAdapter:
    """LLM adapter boundary.

    Keep LLM behavior behind a small interface so production code can validate
    structured output before it becomes an accounting action.
    """

    def plan(self, prompt: str, context: dict, snippets: list[dict]) -> dict:
        return {
            "intent": "review_reconciliation_candidate",
            "confidence": 0.82,
            "recommended_function_id": "nicole.review_reconciliation",
            "reason": "The request is about reviewing an unmatched bank transaction.",
            "uses_retrieved_context": bool(snippets),
        }
