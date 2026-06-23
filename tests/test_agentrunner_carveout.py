"""Package boundary + provider-injection gate.

Two guarantees:

1. No module under ``src/agentrunner/`` imports ``gaia`` / ``gaia_private`` — the
   package is host-agnostic (this is the invariant the gaia carve-out, BOU-1750,
   was built around; kept here as a permanent regression guard).
2. ``AgentRunner.run`` works against an injected ``ModelClientProvider`` with no
   host dependency.

Container-free: pure import + AST scan + an in-memory fake provider.
"""
import ast
import pathlib

import pytest

# tests/<this file> -> parents[1] == repo root; package lives at src/agentrunner.
_PKG = pathlib.Path(__file__).resolve().parents[1] / "src" / "agentrunner"
_HOST_ROOTS = {"gaia", "gaia_private"}


def _host_imports(py: pathlib.Path) -> list[str]:
    """Return the gaia/gaia_private modules imported by a single source file."""
    tree = ast.parse(py.read_text())
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [n.name for n in node.names if n.name.split(".")[0] in _HOST_ROOTS]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in _HOST_ROOTS:
                hits.append(node.module)
    return hits


def test_agentrunner_package_has_no_host_imports():
    assert _PKG.is_dir(), f"agentrunner package not found at {_PKG}"
    violations = {}
    for py in _PKG.rglob("*.py"):
        hits = _host_imports(py)
        if hits:
            violations[str(py.relative_to(_PKG))] = hits
    assert not violations, f"agentrunner must stay host-agnostic: {violations}"


@pytest.mark.asyncio
async def test_run_uses_injected_model_provider():
    """AgentRunner.run must resolve the injected provider, not a hard-coded one."""
    from agents import Agent

    from agentrunner import AgentRunner, configure_agentrunner
    from support.fake_model_provider import FakeModelClientProvider

    fake = FakeModelClientProvider(result_text="ok")
    configure_agentrunner(model_provider=fake)

    agent = Agent(name="carveout-probe", model="fake-model", instructions="x")
    result = await AgentRunner.run(agent, "hi", model="fake-model", use_fallback=True)

    assert fake.was_called, "injected provider was never consulted"
    assert AgentRunner.extract_text_response(result) == "ok"
