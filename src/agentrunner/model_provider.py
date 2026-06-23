"""The model-layer dependency-inversion seam for AgentRunner (BOU-1750).

``AgentRunner`` depends on this protocol, never on a concrete model layer.
The host injects a provider via ``configure_agentrunner``; gaia supplies an
adapter over its existing ``model_manager``. The method set mirrors exactly the
symbols ``AgentRunner`` previously imported from ``gaia.infra.llm.model_manager``.

(A batteries-included OpenRouter default provider is deferred to a follow-up —
see the Linear restore ticket; the implementation lives in this branch's git
history at commit cc353a2d1.)
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

# A callback that executes one agent run against a resolved model + the
# agents-SDK model provider, returning the SDK ``RunResult``.
RunWithModel = Callable[[str, Any], Awaitable[Any]]


@runtime_checkable
class ModelClientProvider(Protocol):
    """Resolves models/providers and owns fallback + error-classification policy."""

    #: Backoff floor/ceiling (seconds) read by the streaming-fallback loop.
    default_rate_limit_delay_seconds: float
    max_rate_limit_delay_seconds: float

    def create_model_provider_for_model(
        self, model_key: str, provider_override: str | None = None
    ) -> tuple[Any, str]:
        """Return ``(agents_sdk_model_provider, resolved_model)`` for a model key."""
        ...

    async def retry_with_fallback(
        self,
        model_key: str,
        run_with_model: RunWithModel,
        *,
        max_retries_per_model: int = 2,
        retry_on_validation_failure: bool = True,
        task_name: str | None = None,
        provider_override: str | None = None,
    ) -> tuple[Any, str]:
        """Run ``run_with_model`` with fallback, returning ``(result, successful_model)``."""
        ...

    def get_fallback_models(self, model: str) -> list[str]:
        """Ordered fallback models to try after ``model`` fails."""
        ...

    def get_provider_for_model(self, model: str) -> str:
        """Provider slug for a model (used for the ``provider:`` trace tag)."""
        ...

    def get_model_setting_aliases(self, resolved_model: str) -> list[str]:
        """Alternate keys to try for a ``model_settings_by_model`` lookup.

        After the exact ``resolved_model`` key misses, these aliases (e.g. a
        canonical model id and its enum name) are tried in order. Hosts that
        have no alias scheme return an empty list.
        """
        ...

    def clamp_max_tokens(self, model: str, max_tokens: int) -> int:
        """Clamp a requested ``max_tokens`` to the model's deployment cap."""
        ...

    def is_rate_limit_error(self, exc: Exception) -> bool:
        """Whether ``exc`` is a rate-limit (429 / overloaded) error."""
        ...

    def is_provider_error(self, exc: Exception) -> bool:
        """Whether ``exc`` is a provider/transport error eligible for fallback."""
        ...
