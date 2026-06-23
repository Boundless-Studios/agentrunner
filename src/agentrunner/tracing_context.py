"""Contextvar-based tracing context for automatic trace propagation (BOU-1750).

Set the tracing context once at the turn/request boundary and every
``AgentRunner`` call inherits ``group_id`` / ``user_id`` / metadata without
per-agent plumbing. This is the gaia-free core; host-specific helpers that
resolve human-readable tags from a database live in the host application.
"""
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TracingContext:
    """Immutable snapshot of the current tracing context."""

    group_id: Optional[str] = None
    """Maps to Langfuse session_id (typically campaign_id / session_id)."""

    user_id: Optional[str] = None
    """Maps to Langfuse user_id for per-user trace filtering."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Arbitrary key-value pairs forwarded as trace_metadata."""


_tracing_ctx: ContextVar[Optional[TracingContext]] = ContextVar(
    "llm_tracing_ctx", default=None
)


def set_tracing_context(
    group_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Set the tracing context for the current async task.

    Call this at the start of a turn / request handler.  All subsequent
    ``AgentRunner.run()`` / ``run_streamed()`` calls within the same async
    context will pick up these values as defaults.
    """
    _tracing_ctx.set(TracingContext(
        group_id=group_id,
        user_id=user_id,
        metadata=metadata or {},
    ))


def get_tracing_context() -> Optional[TracingContext]:
    """Return the current tracing context, or ``None`` if unset."""
    return _tracing_ctx.get()


def clear_tracing_context() -> None:
    """Clear the tracing context for the current async task."""
    _tracing_ctx.set(None)
