"""Logging behavior for recoverable streaming fallback attempts."""

import logging

import pytest
from agents import Agent

from agentrunner import AgentRunner, configure_agentrunner, runtime
from support.fake_model_provider import FakeModelClientProvider


@pytest.fixture(autouse=True)
def _reset_runtime():
    runtime.reset_for_tests()
    yield
    runtime.reset_for_tests()


@pytest.mark.asyncio
async def test_recovered_streaming_attempt_does_not_emit_error(monkeypatch, caplog):
    provider = FakeModelClientProvider(fallback_models=["backup-model"])
    provider.is_provider_error = lambda exc: isinstance(exc, TimeoutError)
    configure_agentrunner(model_provider=provider)

    monkeypatch.setattr(
        AgentRunner,
        "run_streamed",
        staticmethod(lambda **kwargs: kwargs["model"]),
    )

    async def iter_streamed_text(model):
        if model == "primary-model":
            raise TimeoutError("provider timed out")
        yield "fallback result", True

    monkeypatch.setattr(
        AgentRunner,
        "iter_streamed_text",
        staticmethod(iter_streamed_text),
    )

    caplog.set_level(logging.DEBUG, logger="agentrunner.agent_runner")
    chunks = [
        chunk
        async for chunk in AgentRunner.iter_streamed_text_with_fallback(
            agent=Agent(name="streaming-probe", instructions="x"),
            prompt="hi",
            model="primary-model",
        )
    ]

    assert chunks == [("fallback result", True, "backup-model")]
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert any(
        record.levelno == logging.WARNING
        and "primary-model failed: TimeoutError" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_exhausted_streaming_fallback_still_emits_error(monkeypatch, caplog):
    provider = FakeModelClientProvider(fallback_models=["backup-model"])
    provider.is_provider_error = lambda exc: isinstance(exc, TimeoutError)
    configure_agentrunner(model_provider=provider)

    monkeypatch.setattr(
        AgentRunner,
        "run_streamed",
        staticmethod(lambda **kwargs: kwargs["model"]),
    )

    async def iter_streamed_text(_model):
        raise TimeoutError("provider timed out")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(
        AgentRunner,
        "iter_streamed_text",
        staticmethod(iter_streamed_text),
    )

    caplog.set_level(logging.DEBUG, logger="agentrunner.agent_runner")
    with pytest.raises(Exception, match="All models failed for streaming-probe"):
        _ = [
            chunk
            async for chunk in AgentRunner.iter_streamed_text_with_fallback(
                agent=Agent(name="streaming-probe", instructions="x"),
                prompt="hi",
                model="primary-model",
            )
        ]

    error_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.ERROR
    ]
    assert error_messages == [
        "[streaming-probe] ❌ All models failed: "
        "primary-model(TimeoutError) → backup-model(TimeoutError)"
    ]
