from __future__ import annotations

from .models import FunctionDefinition


class FunctionRegistry:
    """In-memory contract catalog for callable AI/workflow actions."""

    def __init__(self) -> None:
        self._functions: dict[str, FunctionDefinition] = {}

    def register(self, definition: FunctionDefinition) -> None:
        if definition.execution_type == "SAGA_COMMAND":
            if not definition.command_topic or not definition.result_topic:
                raise ValueError("SAGA_COMMAND requires command_topic and result_topic")
        self._functions[definition.function_id] = definition

    def get(self, function_id: str) -> FunctionDefinition:
        try:
            return self._functions[function_id]
        except KeyError as exc:
            raise KeyError(f"Function not registered: {function_id}") from exc

    def list_functions(self) -> list[FunctionDefinition]:
        return list(self._functions.values())
