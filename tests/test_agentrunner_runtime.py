"""Unit tests for agentrunner runtime configuration."""
import pytest

from agentrunner import runtime
from agentrunner.model_provider import ModelClientProvider
from support.fake_model_provider import FakeModelClientProvider


@pytest.fixture(autouse=True)
def _reset_runtime():
    runtime.reset_for_tests()
    yield
    runtime.reset_for_tests()


def test_get_active_provider_raises_before_configuration():
    assert runtime.is_configured() is False
    with pytest.raises(RuntimeError, match="not configured"):
        runtime.get_active_provider()


def test_configure_installs_provider_and_factory():
    fake = FakeModelClientProvider()
    sentinel = object()
    runtime.configure_agentrunner(
        model_provider=fake, trace_processor_factory=lambda: sentinel
    )
    assert runtime.is_configured() is True
    assert runtime.get_active_provider() is fake
    assert runtime.get_trace_processor_factory()() is sentinel


def test_fake_provider_satisfies_protocol():
    # runtime_checkable protocol — structural conformance of the test double.
    assert isinstance(FakeModelClientProvider(), ModelClientProvider)


def test_register_bootstrap_fires_lazily_on_first_get_active_provider():
    """A registered bootstrap configures the provider on first use (BOU-1778)."""
    fake = FakeModelClientProvider()
    calls = []

    def _bootstrap():
        calls.append(1)
        runtime.configure_agentrunner(model_provider=fake)

    runtime.register_bootstrap(_bootstrap)
    assert runtime.is_configured() is False  # not until first use

    assert runtime.get_active_provider() is fake  # bootstrap fires here
    assert calls == [1]
    # Second call does not re-run the bootstrap (already configured).
    assert runtime.get_active_provider() is fake
    assert calls == [1]
