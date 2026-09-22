from __future__ import annotations


class NicoleRecommendationService:
    """Accounting recommendation service with a human-review friendly response."""

    def review_reconciliation(self, payload: dict) -> dict:
        transaction = payload.get("transaction", {})
        return {
            "status": "SUCCESS",
            "recommendation_id": "rec_demo_001",
            "message_to_show": "Nicole found a likely reconciliation candidate.",
            "candidate": {
                "transaction_id": transaction.get("id"),
                "suggested_account": "Accounts Receivable",
                "confidence": 0.82,
                "requires_human_approval": True,
            },
        }
