"""A gaia-free ``ModelClientProvider`` test double (BOU-1750).

Proves ``AgentRunner.run`` delegates to the injected provider without any gaia
dependency or real LLM call. ``retry_with_fallback`` returns a canned result
(or raises an injected error) instead of invoking the network-bound
``run_with_model`` callback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FakeRunResult:
    """Minimal stand-in for the agents-SDK ``RunResult``."""

    final_output: Any


class FakeModelClientProvider:
    """In-memory provider satisfying the ``ModelClientProvider`` protocol."""

    default_rate_limit_delay_seconds: float = 1.0
    max_rate_limit_delay_seconds: float = 8.0

    def __init__(
        self,
        result_text: str = "ok",
        error: Optional[Exception] = None,
        fallback_models: Optional[list[str]] = None,
    ) -> None:
        self.result_text = result_text
        self.error = error
        self._fallback_models = fallback_models or []
        self.was_called = False
        self.last_model_key: Optional[str] = None

    def create_model_provider_for_model(
        self, model_key: str, provider_override: str | None = None
    ) -> tuple[Any, str]:
        self.was_called = True
        self.last_model_key = model_key
        return object(), model_key

    async def retry_with_fallback(
        self,
        model_key: str,
        run_with_model,
        *,
        max_retries_per_model: int = 2,
        retry_on_validation_failure: bool = True,
        task_name: str | None = None,
        provider_override: str | None = None,
    ) -> tuple[Any, str]:
        self.was_called = True
        self.last_model_key = model_key
        if self.error is not None:
            raise self.error
        return FakeRunResult(final_output=self.result_text), model_key

    def get_fallback_models(self, model: str) -> list[str]:
        return list(self._fallback_models)

    def get_provider_for_model(self, model: str) -> str:
        return "fake"

    def get_model_setting_aliases(self, resolved_model: str) -> list:
        return []

    def clamp_max_tokens(self, model: str, max_tokens: int) -> int:
        return max_tokens

    def is_rate_limit_error(self, exc: Exception) -> bool:
        return False

    def is_provider_error(self, exc: Exception) -> bool:
        return False
