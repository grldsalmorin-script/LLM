from __future__ import annotations

from collections import defaultdict, deque
from typing import Callable

from .models import Command, CommandResult

CommandHandler = Callable[[Command], CommandResult]
ResultHandler = Callable[[CommandResult], None]


class InMemoryCommandBus:
    """Tiny Pub/Sub stand-in used for local explanation and tests."""

    def __init__(self) -> None:
        self._command_handlers: dict[str, CommandHandler] = {}
        self._result_handlers: dict[str, list[ResultHandler]] = defaultdict(list)
        self._queue: deque[Command] = deque()

    def subscribe_command(self, topic: str, handler: CommandHandler) -> None:
        self._command_handlers[topic] = handler

    def subscribe_result(self, topic: str, handler: ResultHandler) -> None:
        self._result_handlers[topic].append(handler)

    def publish_command(self, topic: str, command: Command) -> None:
        if topic not in self._command_handlers:
            raise KeyError(f"No command handler subscribed for topic: {topic}")
        self._queue.append(command)

    def drain(self) -> None:
        while self._queue:
            command = self._queue.popleft()
            handler = self._command_handlers[command.metadata["command_topic"]]
            result = handler(command)
            for result_handler in self._result_handlers[command.result_topic]:
                result_handler(result)
