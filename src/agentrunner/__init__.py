"""agentrunner — a standalone, host-agnostic LLM agent-execution package.

``AgentRunner`` orchestrates the OpenAI Agents SDK with model fallback,
streaming, ``<think>``-tag filtering, structured-output extraction, and
tracing. It depends on an injected :class:`ModelClientProvider` (see
:func:`configure_agentrunner`) rather than on any host model layer, so it can
be carved out as an open-source library (BOU-1750).
"""
from agentrunner.agent_runner import (  # noqa: F401
    AGENT_RUN_TIMEOUT_SECONDS,
    AgentRunner,
    ThinkTagFilter,
)
from agentrunner.model_provider import ModelClientProvider  # noqa: F401
from agentrunner.runtime import (  # noqa: F401
    configure_agentrunner,
    get_active_provider,
    is_configured,
)

__all__ = [
    "AGENT_RUN_TIMEOUT_SECONDS",
    "AgentRunner",
    "ThinkTagFilter",
    "ModelClientProvider",
    "configure_agentrunner",
    "get_active_provider",
    "is_configured",
]
