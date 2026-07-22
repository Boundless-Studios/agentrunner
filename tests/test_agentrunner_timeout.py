"""Timeout and fallback regression tests for ``AgentRunner``."""

import asyncio
from types import SimpleNamespace

import pytest
from agents import Agent

from agentrunner import AgentRunner, configure_agentrunner, runtime
from agentrunner.providers.openrouter import OpenRouterModelClientProvider


@pytest.fixture(autouse=True)
def _reset_runtime():
    runtime.reset_for_tests()
    yield
    runtime.reset_for_tests()


@pytest.mark.asyncio
async def test_timed_out_model_attempt_uses_fallback(monkeypatch):
    provider = OpenRouterModelClientProvider(
        api_key="test-key",
        fallback_models=["backup-model"],
    )
    configure_agentrunner(model_provider=provider)
    attempted_models = []

    async def fake_run(self, agent, prompt, **kwargs):
        model = kwargs["run_config"].model
        attempted_models.append(model)
        if model == "primary-model":
            await asyncio.sleep(1)
        return SimpleNamespace(final_output="fallback result")

    monkeypatch.setattr("agentrunner.agent_runner.Runner.run", fake_run)

    agent = Agent(name="timeout-probe", instructions="x")
    result = await AgentRunner.run(
        agent,
        "hi",
        model="primary-model",
        timeout=0.01,
        use_fallback=True,
    )

    assert result.final_output == "fallback result"
    assert attempted_models == ["primary-model", "backup-model"]
