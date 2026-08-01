"""``AgentRunner.run()`` surfaces the resolved model on results (BOU-2161).

Proves the returned result carries a ``model_used`` attribute naming the model
that actually produced the output:
- the fallback model when ``retry_with_fallback`` selected one;
- the requested model on the no-fallback path;
- attribution is best-effort — a result that refuses attribute assignment is
  left unchanged and no exception escapes.

Uses the fake ModelClientProvider + a stubbed ``Runner.run`` — no gaia, no LLM.
"""
import pytest
from agents import Agent
from pydantic import BaseModel

from agentrunner import AgentRunner, runtime
from support.fake_model_provider import FakeModelClientProvider, FakeRunResult


@pytest.fixture(autouse=True)
def _reset():
    runtime.reset_for_tests()
    yield
    runtime.reset_for_tests()


@pytest.mark.asyncio
async def test_run_surfaces_fallback_model_on_result():
    provider = FakeModelClientProvider(result_model="fallback-model")
    runtime.configure_agentrunner(model_provider=provider)

    agent = Agent(name="Probe", model="primary-model", instructions="x")
    result = await AgentRunner.run(agent, "hi")

    assert result.model_used == "fallback-model"
    assert result.model_used != "primary-model"


@pytest.mark.asyncio
async def test_run_surfaces_requested_model_without_fallback(monkeypatch):
    provider = FakeModelClientProvider(result_model="fallback-model")
    runtime.configure_agentrunner(model_provider=provider)

    async def _fake_runner_run(self, agent, prompt, **kwargs):
        return FakeRunResult(final_output="ok")

    # use_fallback=False calls the real SDK Runner.run (network) — stub it.
    monkeypatch.setattr("agentrunner.agent_runner.Runner.run", _fake_runner_run)

    agent = Agent(name="Probe", model="primary-model", instructions="x")
    result = await AgentRunner.run(agent, "hi", use_fallback=False)

    assert result.model_used == "primary-model"


def test_attach_model_used_is_best_effort():
    class _Sealed(BaseModel):
        value: str

    sealed = _Sealed(value="x")
    returned = AgentRunner._attach_model_used(sealed, "m")
    assert returned is sealed
    assert not hasattr(sealed, "model_used")

    plain = {"k": "v"}
    assert AgentRunner._attach_model_used(plain, "m") is plain
    assert plain == {"k": "v"}
