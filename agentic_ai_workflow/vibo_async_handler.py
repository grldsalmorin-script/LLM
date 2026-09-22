from __future__ import annotations

from .models import Command, CommandResult


class ViboAsyncHandler:
    """Simplified version of the vibo_async_handler concept.

    The real platform uses decorators and report-back events. This compact sample
    shows the core idea: run a command handler, normalize the outcome, and publish
    a structured result event back to the workflow/frontend layer.
    """

    def __init__(self) -> None:
        self.status_by_command_id: dict[str, str] = {}

    def wrap(self, event_name: str, handler):
        def command_handler(command: Command) -> CommandResult:
            self.status_by_command_id[command.command_id] = "PROCESSING"
            try:
                payload = handler(command.payload)
                status = payload.get("status", "SUCCESS")
            except Exception as exc:
                payload = {"error": str(exc), "message_to_show": "Command failed"}
                status = "FAILED"
            self.status_by_command_id[command.command_id] = status
            return CommandResult(
                event_name=event_name,
                command_id=command.command_id,
                status=status,
                company_id=command.company_id,
                payload=payload,
                metadata=command.metadata,
            )

        return command_handler
