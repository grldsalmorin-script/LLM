from __future__ import annotations

import json

from agentic_ai_workflow.command_bus import InMemoryCommandBus
from agentic_ai_workflow.context_provider import ContextProvider
from agentic_ai_workflow.function_registry import FunctionRegistry
from agentic_ai_workflow.models import FunctionDefinition
from agentic_ai_workflow.services.copilot import CopilotPlanner
from agentic_ai_workflow.services.gemini import GeminiAdapter
from agentic_ai_workflow.services.jasmine import JasmineAssistant
from agentic_ai_workflow.services.nicole import NicoleRecommendationService
from agentic_ai_workflow.vibo_async_handler import ViboAsyncHandler


def build_demo() -> tuple[JasmineAssistant, InMemoryCommandBus, ViboAsyncHandler]:
    registry = FunctionRegistry()
    bus = InMemoryCommandBus()
    async_handler = ViboAsyncHandler()
    nicole = NicoleRecommendationService()

    registry.register(
        FunctionDefinition(
            function_id="nicole.review_reconciliation",
            friendly_name="Review reconciliation candidate",
            owning_service="nicole",
            execution_type="SAGA_COMMAND",
            command_topic="nicole-commands",
            result_topic="jasmine-results",
            parameter_schema={"required": ["transaction", "plan"]},
            response_schema={"event_name": "NicoleReconciliationReviewed"},
        )
    )

    bus.subscribe_command(
        "nicole-commands",
        async_handler.wrap("NicoleReconciliationReviewed", nicole.review_reconciliation),
    )

    planner = CopilotPlanner(ContextProvider(), GeminiAdapter())
    jasmine = JasmineAssistant(registry, bus, planner)
    bus.subscribe_result("jasmine-results", jasmine.receive_result)
    return jasmine, bus, async_handler


def main() -> None:
    jasmine, bus, async_handler = build_demo()
    command_id = jasmine.handle_user_message(
        company_id="company_abc",
        user_message="Please reconcile this Stripe bank transaction.",
    )
    bus.drain()

    print(json.dumps({
        "command_id": command_id,
        "tracker_status": async_handler.status_by_command_id[command_id],
        "frontend_events": jasmine.events,
    }, indent=2))


if __name__ == "__main__":
    main()
