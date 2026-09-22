from __future__ import annotations

from uuid import uuid4

from agentic_ai_workflow.command_bus import InMemoryCommandBus
from agentic_ai_workflow.function_registry import FunctionRegistry
from agentic_ai_workflow.models import Command
from agentic_ai_workflow.services.copilot import CopilotPlanner


class JasmineAssistant:
    """User-facing assistant that turns intent into registered commands."""

    def __init__(self, registry: FunctionRegistry, bus: InMemoryCommandBus, planner: CopilotPlanner) -> None:
        self.registry = registry
        self.bus = bus
        self.planner = planner
        self.events: list[dict] = []

    def handle_user_message(self, company_id: str, user_message: str) -> str:
        plan = self.planner.choose_function(company_id, user_message)
        function_id = plan["recommended_function_id"]
        function_def = self.registry.get(function_id)

        command_id = f"cmd_{uuid4().hex[:8]}"
        command = Command(
            command_id=command_id,
            command_name=function_id,
            company_id=company_id,
            result_topic=function_def.result_topic or "ui-results",
            payload={
                "user_message": user_message,
                "transaction": {"id": "txn_123", "description": "Stripe payout", "amount": 1200},
                "plan": plan,
            },
            metadata={"command_topic": function_def.command_topic or "commands"},
        )
        self.bus.publish_command(function_def.command_topic or "commands", command)
        return command_id

    def receive_result(self, result) -> None:
        self.events.append(
            {
                "event_name": result.event_name,
                "command_id": result.command_id,
                "status": result.status,
                "payload": result.payload,
            }
        )
