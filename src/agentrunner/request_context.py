"""Thread-local request context for LLM rate-limit logging (BOU-1750).

Set per-request so HTTP event hooks can annotate rate-limit warnings with the
agent name and model. Pure plumbing — no gaia dependency.
"""
import threading

_request_context = threading.local()


def set_request_context(agent_name: str | None = None, model: str | None = None) -> None:
    """Set context for the current request (for logging purposes)."""
    _request_context.agent_name = agent_name
    _request_context.model = model


def get_request_context() -> tuple[str | None, str | None]:
    """Get the current request context."""
    return (
        getattr(_request_context, "agent_name", None),
        getattr(_request_context, "model", None),
    )


def clear_request_context() -> None:
    """Clear the request context."""
    _request_context.agent_name = None
    _request_context.model = None
