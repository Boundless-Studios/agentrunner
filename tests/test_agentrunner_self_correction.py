"""Self-correction loop through the carved-out package (BOU-1750 × #2299 merge).

Proves the ported correction loop drives the *injected* emitter + langfuse
resolver (not gaia imports) and honors the validator contract end-to-end:
- a validator that fails once then passes triggers exactly one re-prompt;
- the injected prompt-correction emitter is invoked;
- when corrections are exhausted, normalize() is applied as the fallback.

Uses a fake ModelClientProvider + a stub validator — no gaia, no LLM, no DB.
"""
from types import SimpleNamespace

import pytest
from agents import Agent

from agentrunner import AgentRunner, runtime
from agentrunner.output_validation import Violation


class _SequenceValidator:
    """Emits a violation for the first N validate() calls, then none."""

    name = "seq"

    def __init__(self, fail_times: int):
        self._remaining = fail_times
        self.normalize_called = False

    def validate(self, output):
        if self._remaining > 0:
            self._remaining -= 1
            return [Violation(field="label", kind="too_long", message="too long")]
        return []

    def normalize(self, output):
        self.normalize_called = True
        return {"label": "fixed"}


class _CountingProvider:
    """Fake provider returning a fresh structured result each call."""

    default_rate_limit_delay_seconds = 1.0
    max_rate_limit_delay_seconds = 8.0

    def __init__(self):
        self.run_calls = 0

    def create_model_provider_for_model(self, model_key, provider_override=None):
        return object(), model_key

    async def retry_with_fallback(self, model_key, run_with_model, **kwargs):
        self.run_calls += 1
        # final_output is a dict so extract_structured_output returns it.
        return SimpleNamespace(final_output={"label": "candidate"}), model_key

    def get_fallback_models(self, model):
        return []

    def get_provider_for_model(self, model):
        return "fake"

    def get_model_setting_aliases(self, resolved_model):
        return []

    def clamp_max_tokens(self, model, max_tokens):
        return max_tokens

    def is_rate_limit_error(self, exc):
        return False

    def is_provider_error(self, exc):
        return False


@pytest.fixture(autouse=True)
def _reset():
    runtime.reset_for_tests()
    yield
    runtime.reset_for_tests()


@pytest.mark.asyncio
async def test_correction_loop_reprompts_once_then_succeeds_and_emits_event():
    provider = _CountingProvider()
    emitted = []

    async def emitter(**kwargs):
        emitted.append(kwargs)

    runtime.configure_agentrunner(
        model_provider=provider, prompt_correction_emitter=emitter
    )

    validator = _SequenceValidator(fail_times=1)  # fail first, pass after re-prompt
    agent = Agent(name="probe", model="fake", instructions="x")

    result = await AgentRunner.run(
        agent, "hi", model="fake", output_validators=[validator], max_corrections=1
    )

    # Initial run + one corrective re-prompt.
    assert provider.run_calls == 2
    # The gap event was emitted on the first violation (attempt 0).
    assert any(e["attempt"] == 0 for e in emitted)
    # Correction succeeded → original (now-valid) result returned, normalize() unused.
    assert validator.normalize_called is False
    assert AgentRunner.extract_structured_output(result) == {"label": "candidate"}


@pytest.mark.asyncio
async def test_exhausted_corrections_apply_normalize_fallback_and_emit_exhausted():
    provider = _CountingProvider()
    emitted = []

    async def emitter(**kwargs):
        emitted.append(kwargs)

    runtime.configure_agentrunner(
        model_provider=provider, prompt_correction_emitter=emitter
    )

    validator = _SequenceValidator(fail_times=99)  # never passes
    agent = Agent(name="probe", model="fake", instructions="x")

    result = await AgentRunner.run(
        agent, "hi", model="fake", output_validators=[validator], max_corrections=1
    )

    # normalize() fallback applied and written back onto the result wrapper.
    assert validator.normalize_called is True
    assert result.final_output == {"label": "fixed"}
    # Both a "new" (attempt 0) and an "exhausted" event were emitted.
    assert any(e["attempt"] == 0 and not e["exhausted"] for e in emitted)
    assert any(e["exhausted"] for e in emitted)


@pytest.mark.asyncio
async def test_no_validators_is_a_noop_no_emitter_call():
    provider = _CountingProvider()
    emitted = []

    async def emitter(**kwargs):
        emitted.append(kwargs)

    runtime.configure_agentrunner(
        model_provider=provider, prompt_correction_emitter=emitter
    )

    agent = Agent(name="probe", model="fake", instructions="x")
    await AgentRunner.run(agent, "hi", model="fake")  # no output_validators

    assert provider.run_calls == 1
    assert emitted == []


@pytest.mark.asyncio
async def test_lazy_bootstrap_runs_before_sdk_setup(monkeypatch):
    """Provider bootstrap must fire before configure_agents_sdk reads the
    trace-processor factory (BOU-1778 review): on a bootstrap-reliant path,
    configure_agents_sdk must observe an already-configured runtime.
    """
    import agentrunner.agent_runner as ar

    provider = _CountingProvider()

    def _bootstrap():
        runtime.configure_agentrunner(model_provider=provider)

    runtime.register_bootstrap(_bootstrap)

    seen = {}

    def _spy_configure_agents_sdk():
        seen["configured_when_sdk_setup_ran"] = runtime.is_configured()

    monkeypatch.setattr(ar, "configure_agents_sdk", _spy_configure_agents_sdk)

    agent = Agent(name="probe", model="fake", instructions="x")
    # use_fallback=True routes through _CountingProvider.retry_with_fallback
    # (canned result, no real Runner). Ordering is independent of fallback.
    await AgentRunner.run(agent, "hi", model="fake", use_fallback=True)

    # The bootstrap fired (provider resolved) BEFORE SDK setup ran — the ordering
    # the fix guarantees. Pre-fix this would be False (SDK setup ran first).
    assert seen["configured_when_sdk_setup_ran"] is True
    assert provider.run_calls == 1
