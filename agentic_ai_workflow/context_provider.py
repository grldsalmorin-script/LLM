from __future__ import annotations


class ContextProvider:
    """Structured context retrieval with an optional RAG-like snippet path."""

    def get_company_context(self, company_id: str) -> dict:
        return {
            "company_id": company_id,
            "currency": "SGD",
            "open_tasks": ["review_unmatched_bank_transaction"],
            "policy": "Do not post AI recommendations without approval.",
        }

    def retrieve_document_snippets(self, query: str) -> list[dict]:
        # This mimics retrieval. A production RAG version would use embeddings and
        # vector search instead of this fixed snippet list.
        return [
            {
                "source": "accounting_policy.md",
                "text": "Bank reconciliation suggestions require evidence and reviewer approval.",
            }
        ] if "reconcile" in query.lower() else []
