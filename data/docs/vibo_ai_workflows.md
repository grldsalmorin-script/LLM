# Vibo AI Workflow Notes

The platform uses a tool and command oriented architecture for AI workflows. Jasmine is the user-facing assistant, Copilot plans the next action, Gemini can provide structured LLM output, and Nicole produces accounting recommendations. Backend services should validate AI output before changing accounting state.

A Function Registry stores callable actions with stable function IDs, ownership, input schemas, output schemas, command topics, result topics, and execution type. Workflows should depend on the registry contract rather than hard-coded service calls.
