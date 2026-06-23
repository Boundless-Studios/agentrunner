"""Process-wide AgentRunner configuration (BOU-1750).

The host application calls :func:`configure_agentrunner` once at startup to
inject the model provider and optional tracing/request hooks. ``AgentRunner``
reads them back via the getters here, so the runner itself imports nothing from
the host.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from agentrunner.model_provider import ModelClientProvider

# A factory returning an agents-SDK trace processor (e.g. a Langfuse processor),
# or ``None`` to leave SDK tracing at its default (disabled without creds).
TraceProcessorFactory = Callable[[], Any]

# An async callable that records a prompt-correction/gap event. Host-specific
# (gaia writes to a DB repository); ``None`` makes the correction loop's event
# emission a no-op, keeping the package free of any storage dependency.
PromptCorrectionEmitter = Callable[..., Any]

# A callable returning ``(prompt_name, prompt_version)`` for the active Langfuse
# prompt, or ``None``. Host-specific (gaia reads a contextvar).
LangfusePromptResolver = Callable[[], Any]

_active_provider: Optional[ModelClientProvider] = None
_trace_processor_factory: Optional[TraceProcessorFactory] = None
_tracing_context_resolver: Optional[Callable[..., Any]] = None
_request_context_hooks: Optional[Any] = None
_prompt_correction_emitter: Optional[PromptCorrectionEmitter] = None
_langfuse_prompt_resolver: Optional[LangfusePromptResolver] = None
_bootstrap: Optional[Callable[[], None]] = None
_bootstrapping: bool = False


def configure_agentrunner(
    *,
    model_provider: ModelClientProvider,
    trace_processor_factory: Optional[TraceProcessorFactory] = None,
    tracing_context_resolver: Optional[Callable[..., Any]] = None,
    request_context_hooks: Optional[Any] = None,
    prompt_correction_emitter: Optional[PromptCorrectionEmitter] = None,
    langfuse_prompt_resolver: Optional[LangfusePromptResolver] = None,
) -> None:
    """Install the active model provider and optional hooks. Idempotent."""
    global _active_provider, _trace_processor_factory
    global _tracing_context_resolver, _request_context_hooks
    global _prompt_correction_emitter, _langfuse_prompt_resolver
    _active_provider = model_provider
    _trace_processor_factory = trace_processor_factory
    _tracing_context_resolver = tracing_context_resolver
    _request_context_hooks = request_context_hooks
    _prompt_correction_emitter = prompt_correction_emitter
    _langfuse_prompt_resolver = langfuse_prompt_resolver


def get_active_provider() -> ModelClientProvider:
    """Return the configured model provider, raising if unconfigured.

    If a lazy bootstrap was registered (see :func:`register_bootstrap`) and no
    provider is configured yet, run it once first. This lets a host configure
    itself on first use without every entrypoint calling configure_agentrunner
    explicitly (gaia registers such a bootstrap at package import).
    """
    global _bootstrapping
    if _active_provider is None and _bootstrap is not None and not _bootstrapping:
        _bootstrapping = True
        try:
            _bootstrap()
        finally:
            _bootstrapping = False
    if _active_provider is None:
        raise RuntimeError(
            "agentrunner is not configured: call "
            "configure_agentrunner(model_provider=...) at startup first"
        )
    return _active_provider


def register_bootstrap(fn: Callable[[], None]) -> None:
    """Register a lazy, idempotent configuration callback.

    Invoked once by :func:`get_active_provider` when no provider is configured.
    The callback should call :func:`configure_agentrunner`. Keep registration
    cheap (no heavy imports) — the callback itself may import freely since it
    runs on first agent use, not at registration time.
    """
    global _bootstrap
    _bootstrap = fn


def is_configured() -> bool:
    return _active_provider is not None


def get_trace_processor_factory() -> Optional[TraceProcessorFactory]:
    return _trace_processor_factory


def get_prompt_correction_emitter() -> Optional[PromptCorrectionEmitter]:
    return _prompt_correction_emitter


def get_langfuse_prompt_resolver() -> Optional[LangfusePromptResolver]:
    return _langfuse_prompt_resolver


def reset_for_tests() -> None:
    """Clear configuration (incl. the lazy bootstrap). Test-only helper."""
    global _active_provider, _trace_processor_factory
    global _tracing_context_resolver, _request_context_hooks
    global _prompt_correction_emitter, _langfuse_prompt_resolver
    global _bootstrap, _bootstrapping
    _active_provider = None
    _trace_processor_factory = None
    _tracing_context_resolver = None
    _request_context_hooks = None
    _prompt_correction_emitter = None
    _langfuse_prompt_resolver = None
    _bootstrap = None
    _bootstrapping = False
