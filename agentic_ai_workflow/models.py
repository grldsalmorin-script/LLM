from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

ExecutionType = Literal["DIRECT_FUNCTION_CALL", "HTTP_POST", "SAGA_COMMAND"]


@dataclass(frozen=True)
class FunctionDefinition:
    """Contract stored in a function registry.

    This is the same architectural idea as the SOP Function Registry: the workflow
    layer depends on a stable function_id and schema, not a hard-coded Python call.
    """

    function_id: str
    friendly_name: str
    owning_service: str
    execution_type: ExecutionType
    parameter_schema: dict[str, Any]
    response_schema: dict[str, Any]
    command_topic: str | None = None
    result_topic: str | None = None
    handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None


@dataclass(frozen=True)
class Command:
    command_id: str
    command_name: str
    company_id: str
    payload: dict[str, Any]
    result_topic: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandResult:
    event_name: str
    command_id: str
    status: str
    company_id: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
