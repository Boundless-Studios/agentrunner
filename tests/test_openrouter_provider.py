"""Unit tests for the OpenRouter default provider (BOU-1750, Task 5)."""
import pytest

from agentrunner.model_provider import ModelClientProvider
from agentrunner.providers.openrouter import OpenRouterModelClientProvider


class _Status429(Exception):
    status_code = 429


def test_satisfies_protocol():
    assert isinstance(OpenRouterModelClientProvider(api_key="x"), ModelClientProvider)


def test_create_model_provider_targets_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    prov = OpenRouterModelClientProvider()
    model_provider, resolved = prov.create_model_provider_for_model("anthropic/claude-sonnet-4.5")
    assert resolved == "anthropic/claude-sonnet-4.5"
    assert "openrouter.ai" in str(model_provider._client.base_url)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    prov = OpenRouterModelClientProvider()
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        prov.create_model_provider_for_model("anthropic/claude-sonnet-4.5")


def test_error_classification():
    prov = OpenRouterModelClientProvider(api_key="x")
    assert prov.is_rate_limit_error(_Status429()) is True
    assert prov.is_rate_limit_error(Exception("429 Too Many Requests")) is True
    assert prov.is_provider_error(Exception("connection reset")) is True
    assert prov.is_provider_error(TimeoutError()) is True
    assert prov.is_rate_limit_error(Exception("bad schema")) is False
    assert prov.is_provider_error(Exception("validation error: missing field")) is False


def test_clamp_max_tokens_honors_caps():
    prov = OpenRouterModelClientProvider(api_key="x", max_tokens_caps={"m": 4096})
    assert prov.clamp_max_tokens("m", 65536) == 4096
    assert prov.clamp_max_tokens("uncapped", 65536) == 65536


def test_provider_tag_from_slug():
    prov = OpenRouterModelClientProvider(api_key="x")
    assert prov.get_provider_for_model("moonshotai/kimi-k2") == "moonshotai"
    assert prov.get_provider_for_model("bare-model") == "openrouter"


@pytest.mark.asyncio
async def test_retry_with_fallback_returns_first_success():
    prov = OpenRouterModelClientProvider(api_key="x")
    calls = []

    async def run_with_model(resolved, provider):
        calls.append(resolved)
        return "result"

    result, model = await prov.retry_with_fallback("primary", run_with_model)
    assert result == "result"
    assert model == "primary"
    assert calls == ["primary"]


@pytest.mark.asyncio
async def test_retry_with_fallback_moves_to_next_model_on_provider_error():
    prov = OpenRouterModelClientProvider(api_key="x", fallback_models=["backup"])
    calls = []

    async def run_with_model(resolved, provider):
        calls.append(resolved)
        if resolved == "primary":
            raise Exception("connection refused")  # provider error → fall back
        return "ok"

    result, model = await prov.retry_with_fallback("primary", run_with_model)
    assert result == "ok"
    assert model == "backup"
    assert calls == ["primary", "backup"]


@pytest.mark.asyncio
async def test_retry_with_fallback_reraises_non_provider_error_when_validation_retry_disabled():
    prov = OpenRouterModelClientProvider(api_key="x", fallback_models=["backup"])
    attempts = []

    async def run_with_model(resolved, provider):
        attempts.append(resolved)
        raise ValueError("schema validation failed")  # not a provider error

    with pytest.raises(ValueError, match="schema validation failed"):
        await prov.retry_with_fallback(
            "primary", run_with_model, max_retries_per_model=3,
            retry_on_validation_failure=False,
        )
    assert attempts == ["primary"]  # no retry, no fallback


@pytest.mark.asyncio
async def test_retry_with_fallback_honors_validation_retry_budget():
    prov = OpenRouterModelClientProvider(api_key="x", fallback_models=["backup"])
    attempts = []

    async def run_with_model(resolved, provider):
        attempts.append(resolved)
        raise ValueError("Invalid JSON")  # validation failure, retryable in-model

    with pytest.raises(ValueError, match="Invalid JSON"):
        await prov.retry_with_fallback(
            "primary", run_with_model, max_retries_per_model=3,
            retry_on_validation_failure=True,
        )
    # Retried the same model up to the budget (not moved to fallback), then raised.
    assert attempts == ["primary", "primary", "primary"]
