"""OpenRouter ModelClientProvider — the batteries-included default (BOU-1750).

OpenRouter fronts every model family behind one OpenAI-compatible endpoint and
one API key, with native provider failover. This makes agentrunner usable by an
external consumer with a single ``OPENROUTER_API_KEY`` and no per-provider
credential setup. We force Chat Completions (OpenRouter does not fully implement
the Responses API) and keep a thin model-fallback loop on top of OpenRouter's
own in-model routing.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from agents import Model, ModelProvider, OpenAIChatCompletionsModel
from openai import AsyncOpenAI

from agentrunner.model_provider import RunWithModel

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Substrings (lower-cased) that mark a rate-limit vs. a general provider error.
_RATE_LIMIT_MARKERS = ("rate limit", "too many requests", "overloaded", "429")
_PROVIDER_ERROR_MARKERS = _RATE_LIMIT_MARKERS + (
    "timeout",
    "connection",
    "api",
    "badrequesterror",
    "service unavailable",
    "502",
    "503",
    "504",
)


class _OpenRouterModelProvider(ModelProvider):
    """agents-SDK ModelProvider backed by a single OpenRouter client."""

    def __init__(self, client: AsyncOpenAI, default_model: str) -> None:
        self._client = client
        self._default_model = default_model

    def get_model(self, model_name: Optional[str]) -> Model:
        # Chat Completions, not Responses — OpenRouter only fully supports the former.
        return OpenAIChatCompletionsModel(
            model=model_name or self._default_model,
            openai_client=self._client,
        )


class OpenRouterModelClientProvider:
    """Default :class:`ModelClientProvider` routing everything through OpenRouter."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        fallback_models: Optional[list[str]] = None,
        max_tokens_caps: Optional[dict[str, int]] = None,
        base_url: str = OPENROUTER_BASE_URL,
        default_rate_limit_delay_seconds: float = 1.0,
        max_rate_limit_delay_seconds: float = 8.0,
    ) -> None:
        self._api_key = api_key
        self._fallback_models = list(fallback_models or [])
        self._max_tokens_caps = dict(max_tokens_caps or {})
        self.base_url = base_url
        self.default_rate_limit_delay_seconds = default_rate_limit_delay_seconds
        self.max_rate_limit_delay_seconds = max_rate_limit_delay_seconds

    def _resolve_api_key(self) -> str:
        key = self._api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OpenRouter requires an API key: pass api_key=... or set OPENROUTER_API_KEY"
            )
        return key

    def create_model_provider_for_model(
        self, model_key: str, provider_override: str | None = None
    ) -> tuple[Any, str]:
        client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self._resolve_api_key(),
            max_retries=0,  # our retry_with_fallback owns retries
        )
        return _OpenRouterModelProvider(client, model_key), model_key

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
        models = [model_key] + self.get_fallback_models(model_key)
        last_error: Optional[Exception] = None

        for model in models:
            provider, resolved = self.create_model_provider_for_model(model)
            for attempt in range(max(1, max_retries_per_model)):
                try:
                    result = await run_with_model(resolved, provider)
                    return result, resolved
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    has_retries = attempt + 1 < max_retries_per_model
                    if self.is_rate_limit_error(exc) and has_retries:
                        delay = min(
                            self.default_rate_limit_delay_seconds * (2 ** attempt),
                            self.max_rate_limit_delay_seconds,
                        )
                        logger.info(
                            "[%s] %s rate-limited; retrying in %.2fs",
                            task_name or "agentrunner", resolved, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    if self.is_provider_error(exc):
                        break  # move to next fallback model
                    # Non-provider error (e.g. malformed JSON / schema validation):
                    # retry the same model within the budget before giving up, so a
                    # transient bad response doesn't fail the run on the first try.
                    if retry_on_validation_failure and has_retries:
                        logger.info(
                            "[%s] %s validation failure; retrying (attempt %d/%d)",
                            task_name or "agentrunner", resolved, attempt + 1,
                            max_retries_per_model,
                        )
                        continue
                    raise
        assert last_error is not None
        raise last_error

    def get_fallback_models(self, model: str) -> list[str]:
        return [m for m in self._fallback_models if m != model]

    def get_provider_for_model(self, model: str) -> str:
        # OpenRouter slugs are "vendor/model"; surface the vendor for the trace tag.
        return model.split("/", 1)[0] if "/" in model else "openrouter"

    def get_model_setting_aliases(self, resolved_model: str) -> list[str]:
        # OpenRouter addresses models by their full slug; there's no canonical
        # alias scheme, so model_settings_by_model is keyed by the slug directly.
        return []

    def clamp_max_tokens(self, model: str, max_tokens: int) -> int:
        cap = self._max_tokens_caps.get(model)
        return min(max_tokens, cap) if cap else max_tokens

    def is_rate_limit_error(self, exc: Exception) -> bool:
        if getattr(exc, "status_code", None) == 429:
            return True
        message = str(exc).lower()
        return any(marker in message for marker in _RATE_LIMIT_MARKERS)

    def is_provider_error(self, exc: Exception) -> bool:
        if isinstance(getattr(exc, "status_code", None), int) and exc.status_code >= 429:
            return True
        message = str(exc).lower()
        return any(marker in message for marker in _PROVIDER_ERROR_MARKERS)
