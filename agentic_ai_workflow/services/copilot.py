from __future__ import annotations

from agentic_ai_workflow.context_provider import ContextProvider
from agentic_ai_workflow.services.gemini import GeminiAdapter


class CopilotPlanner:
    """Plans which registered tool/function should run next."""

    def __init__(self, context_provider: ContextProvider, llm: GeminiAdapter) -> None:
        self.context_provider = context_provider
        self.llm = llm

    def choose_function(self, company_id: str, user_message: str) -> dict:
        context = self.context_provider.get_company_context(company_id)
        snippets = self.context_provider.retrieve_document_snippets(user_message)
        return self.llm.plan(user_message, context, snippets)
