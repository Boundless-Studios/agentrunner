"""
Generic agent runner that handles agent execution with proper model configuration.
"""
import json

import asyncio
import logging
from dataclasses import replace
from typing import Any, AsyncIterator, Callable, List, Optional, Dict, Tuple, TypeVar
from agents import Agent, ModelSettings, Runner, RunConfig
from agentrunner.agents_sdk_setup import configure_agents_sdk
from agentrunner.runtime import (
    get_active_provider,
    get_langfuse_prompt_resolver,
    get_prompt_correction_emitter,
)
from agentrunner.request_context import set_request_context
from agentrunner.tracing_context import get_tracing_context

logger = logging.getLogger(__name__)

# Type variable for generic validation
T = TypeVar('T')

import re

# Regex for complete <think>...</think> blocks (possibly spanning buffered text)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class ThinkTagFilter:
    """Stateful filter that strips <think>...</think> blocks from streaming text.

    Some models (Nemotron, Kimi K2) emit chain-of-thought reasoning wrapped in
    <think>...</think> tags.  This filter strips them incrementally so they are
    never forwarded to the client during streaming.

    Some deployments (notably Baseten's Nemotron) prefill the assistant turn
    with ``<think>`` to force reasoning mode. The wire response from those
    deployments contains only the *closing* tag — no opener. Until the filter
    has yielded any real content, an orphan ``</think>`` is treated as the end
    of a server-prefilled reasoning preamble: everything before+including it is
    discarded. After real content has been yielded, an orphan ``</think>`` is
    passed through literally so we don't silently corrupt mid-stream output.
    """

    def __init__(self):
        self._inside = False
        self._buf = ""
        # Track whether we've already emitted real content. Before the first
        # emission we treat an orphan ``</think>`` as a server-prefilled
        # reasoning close; after, we pass it through literally.
        self._yielded_real_content = False

    def feed(self, text: str) -> str:
        """Process a chunk and return only the non-think content."""
        if (
            not self._inside
            and self._yielded_real_content
            and "<" not in text
            and not self._buf
        ):
            return text  # fast path: past preamble window, no tag candidates

        buf = self._buf + text
        self._buf = ""
        out: list[str] = []
        i = 0

        while i < len(buf):
            if self._inside:
                end = buf.find(_THINK_CLOSE, i)
                if end != -1:
                    self._inside = False
                    i = end + len(_THINK_CLOSE)
                else:
                    # still inside — discard rest, but keep potential
                    # partial </think> at the very end
                    for j in range(max(i, len(buf) - len(_THINK_CLOSE) + 1), len(buf)):
                        if buf[j:] == _THINK_CLOSE[:len(buf) - j]:
                            self._buf = buf[j:]
                            return "".join(out)
                    break
            else:
                # Outside both states. Look for either ``<think>`` (paired
                # block) or, while we are still in the preamble window,
                # ``</think>`` (server-prefilled reasoning close).
                open_at = buf.find(_THINK_OPEN, i)
                close_at = (
                    buf.find(_THINK_CLOSE, i)
                    if not self._yielded_real_content
                    else -1
                )

                if open_at != -1 and (close_at == -1 or open_at < close_at):
                    out.append(buf[i:open_at])
                    self._inside = True
                    i = open_at + len(_THINK_OPEN)
                    continue

                if close_at != -1:
                    # Orphan close found in the preamble window — drop
                    # everything before+including it, do NOT add to ``out``.
                    out = []
                    i = close_at + len(_THINK_CLOSE)
                    continue

                # No full tag in this chunk. Look for a partial tail that
                # could complete in a future chunk. While we are still in the
                # preamble window we must defer the *entire* remainder (not
                # just the tail) — yielding the prefix would commit us to
                # output that a forthcoming ``</think>`` could otherwise drop.
                # Once real content has been yielded the preamble window is
                # closed and only the partial-tail prefix needs to be held.
                tail_start = self._partial_tail_start(buf, i)
                if tail_start is not None:
                    if not self._yielded_real_content:
                        self._buf = buf[i:]
                        return "".join(out)
                    out.append(buf[i:tail_start])
                    self._buf = buf[tail_start:]
                    return "".join(out)

                out.append(buf[i:])
                break

        result = "".join(out)
        if result:
            self._yielded_real_content = True
        return result

    def _partial_tail_start(self, buf: str, i: int) -> int | None:
        """Return the index where a partial ``<think>`` or ``</think>`` tag
        could start, or ``None`` if the tail is safe to flush.

        While we are still in the preamble window we must guard both tags;
        once real content has been yielded only a partial ``<think>`` tail
        could matter (orphan ``</think>`` is no longer special).
        """
        candidates: list[str] = [_THINK_OPEN]
        if not self._yielded_real_content:
            candidates.append(_THINK_CLOSE)

        max_len = max(len(c) for c in candidates)
        scan_start = max(i, len(buf) - max_len + 1)
        for j in range(scan_start, len(buf)):
            suffix = buf[j:]
            for tag in candidates:
                if suffix == tag[: len(suffix)]:
                    return j
        return None

    @staticmethod
    def strip(text: str) -> str:
        """One-shot removal of <think>...</think> blocks AND a leading orphan close.

        Used by callers that have the full text in hand. Mirrors the streaming
        feed() behavior: paired strip first, then drop any leading orphan
        ``</think>`` (server-prefilled reasoning preamble).
        """
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "</think>" in cleaned:
            cleaned = cleaned.split("</think>", 1)[1]
        return cleaned


# Default timeout for agent execution (seconds).
# Prevents hung LLM calls from blocking the turn pipeline indefinitely.
AGENT_RUN_TIMEOUT_SECONDS = 90


class AgentRunner:
    """Generic runner for executing agents with proper model configuration."""

    @staticmethod
    def _normalize_tool_settings_for_agent(agent: Agent) -> Agent:
        """Clear tool-related ModelSettings on an agent that has no tools.

        The agents library forwards ``parallel_tool_calls=False`` to the
        underlying chat completions call whenever it appears on the resolved
        ModelSettings, regardless of whether tools are present. Anthropic's
        OpenAI-compatible endpoint then rejects the request with
        "'parallel_tool_calls' can only be set when 'tools' are specified".

        ``RunConfig.model_settings`` cannot override the Agent's value here:
        ``ModelSettings.resolve()`` only overlays non-None fields from the
        override, so a None on RunConfig leaves the Agent's ``False`` intact.
        We therefore clear ``parallel_tool_calls`` and ``tool_choice`` on the
        Agent itself when it has no tools, returning a fresh Agent so callers
        retain their original instance unmutated.
        """
        if getattr(agent, 'tools', None):
            return agent
        settings = getattr(agent, 'model_settings', None)
        if settings is None:
            return agent
        if settings.parallel_tool_calls is None and settings.tool_choice is None:
            return agent
        cleaned = replace(settings, parallel_tool_calls=None, tool_choice=None)
        try:
            return replace(agent, model_settings=cleaned)
        except TypeError:
            agent.model_settings = cleaned
            return agent

    @staticmethod
    def _resolve_tracing(
        trace_group_id: Optional[str],
        trace_metadata: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Fill in tracing params from the contextvar when not explicitly provided."""
        ctx = get_tracing_context()
        if ctx is None:
            return trace_group_id, trace_metadata
        if trace_group_id is None:
            trace_group_id = ctx.group_id
        if trace_metadata is None and ctx.metadata:
            trace_metadata = dict(ctx.metadata)
        # Always merge contextvar tags into trace_metadata so that campaign/env/name
        # tags propagate even when callers provide their own trace_metadata.
        ctx_tags = ctx.metadata.get("tags") if ctx.metadata else None
        if ctx_tags and isinstance(ctx_tags, list):
            if trace_metadata is None:
                trace_metadata = {}
            existing_tags = trace_metadata.get("tags", [])
            if not isinstance(existing_tags, list):
                existing_tags = []
            merged = list(existing_tags)
            for tag in ctx_tags:
                if tag not in merged:
                    merged.append(tag)
            trace_metadata["tags"] = merged
        # Inject user_id into metadata so LangfuseTracingProcessor can
        # promote it to a first-class Langfuse trace field.
        if ctx.user_id:
            if trace_metadata is None:
                trace_metadata = {}
            trace_metadata.setdefault("user_id", ctx.user_id)
        return trace_group_id, trace_metadata

    @staticmethod
    async def run(
        agent: Agent,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        tool_choice: Optional[str] = "auto",
        parallel_tool_calls: bool = False,
        max_turns: Optional[int] = None,
        context: Optional[Any] = None,
        use_fallback: bool = True,
        trace_group_id: Optional[str] = None,
        trace_metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = AGENT_RUN_TIMEOUT_SECONDS,
        provider_override: Optional[str] = None,
        model_provider: Optional[Any] = None,
        output_validators: Optional[list] = None,
        max_corrections: int = 1,
        prompt_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
        model_settings_by_model: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> Any:
        """Run an agent using the Runner pattern with proper model configuration.

        Args:
            agent: The agent to run
            prompt: The prompt to send to the agent
            model: Optional model override (defaults to agent's model if available)
            temperature: Temperature for generation (default 0.7)
            tool_choice: Tool choice behavior (default "auto", can be "required", "none")
            parallel_tool_calls: Whether to allow parallel tool calls
            max_turns: Maximum number of turns for the agent (default None uses Runner's default)
            context: Optional context object to pass to tools and hooks
            use_fallback: Whether to use model fallback on provider failures (default True)
            trace_group_id: Optional group ID for OpenAI tracing (e.g. session/campaign ID)
            trace_metadata: Optional metadata dict for OpenAI tracing
            timeout: Maximum seconds to wait for agent completion (default 90s).
                     Set to None to disable timeout.
            provider_override: Optional provider slug to force for the primary model
                     (e.g. "base10"). Only applies to the direct model call, not
                     fallback models.
            model_provider: Optional pre-resolved model-provider instance. When set
                     (and use_fallback is False) it is used directly instead of
                     re-resolving from model/provider_override — lets a caller whose
                     own retry_with_fallback already chose a (model, provider) pair
                     forward that provider so its failover isn't undone.
            output_validators: Optional list of OutputValidator instances to apply
                     after the agent run succeeds. When None or empty the correction
                     loop is a complete no-op (zero behaviour change for existing callers).
            max_corrections: Maximum number of corrective re-prompts (default 1).
            prompt_name: Optional Langfuse prompt name for the prompt-gap event
                     (falls back to the injected langfuse prompt resolver).
            prompt_version: Optional Langfuse prompt version for the prompt-gap event.
            model_settings_by_model: Optional ModelSettings kwargs keyed by canonical
                     or provider-specific model id. These settings apply only when
                     that exact model is resolved, so provider-specific request
                     bodies do not leak to fallback attempts.
            **kwargs: Additional keyword arguments for ModelSettings

        Returns:
            The result from the agent run
        """
        # Resolve the provider FIRST so any lazy host bootstrap runs before SDK
        # setup. configure_agents_sdk() latches _applied and reads the injected
        # trace-processor factory; if it ran before the bootstrap configured that
        # factory, a CLI/script process with LANGFUSE_SECRET_KEY would
        # permanently skip Langfuse tracing. (BOU-1778 review)
        prov = get_active_provider()
        configure_agents_sdk()

        # Use provided model or try to get from agent
        model_key = model or getattr(agent, 'model', None)
        if not model_key:
            raise ValueError("No model specified and agent has no default model")

        agent = AgentRunner._normalize_tool_settings_for_agent(agent)

        # Build model settings - separate out non-ModelSettings kwargs
        model_settings_kwargs = {
            "temperature": temperature,
        }

        # Only add tool_choice and parallel_tool_calls if agent has tools
        # get_all_tools requires a run_context, so we'll check for tools attribute instead
        if hasattr(agent, 'tools') and agent.tools:
            model_settings_kwargs["tool_choice"] = tool_choice
            model_settings_kwargs["parallel_tool_calls"] = parallel_tool_calls

        # Add any additional kwargs that are valid for ModelSettings
        _non_model_keys = {
            'max_turns',
            'use_fallback',
            'trace_group_id',
            'trace_metadata',
            'model_settings_by_model',
        }
        for key, value in kwargs.items():
            if key not in _non_model_keys:
                model_settings_kwargs[key] = value

        # Resolve tracing from contextvar defaults when not explicitly provided
        trace_group_id, trace_metadata = AgentRunner._resolve_tracing(
            trace_group_id, trace_metadata
        )

        # Build tracing fields for RunConfig
        workflow_name = getattr(agent, 'name', None) or "Agent workflow"
        tracing_kwargs: Dict[str, Any] = {"workflow_name": workflow_name}
        if trace_group_id is not None:
            tracing_kwargs["group_id"] = trace_group_id
        # Inject agent name tag so each trace is tagged with the agent that ran
        if trace_metadata is None:
            trace_metadata = {}
        agent_tag = f"agent:{workflow_name}"
        existing_tags = trace_metadata.get("tags", [])
        if isinstance(existing_tags, list) and agent_tag not in existing_tags:
            trace_metadata["tags"] = existing_tags + [agent_tag]
        tracing_kwargs["trace_metadata"] = trace_metadata

        # Mutable prompt cell so the correction loop can pass a different prompt
        # to run_with_model without restructuring the closure.
        _prompt_cell: List[str] = [prompt]

        # Define the operation to run with a specific model
        async def run_with_model(resolved_model: str, model_provider) -> Any:
            runner = Runner()

            # Inject provider tag into trace metadata
            provider_type = prov.get_provider_for_model(resolved_model)
            provider_tag = f"provider:{provider_type}"
            tags = trace_metadata.get("tags", [])
            # Remove any previous provider tag (in case of fallback retry)
            tags = [t for t in tags if not t.startswith("provider:")]
            tags.append(provider_tag)
            trace_metadata["tags"] = tags

            # Create run config for this specific model
            run_model_settings_kwargs = dict(model_settings_kwargs)
            if model_settings_by_model:
                scoped_settings = model_settings_by_model.get(resolved_model)
                if scoped_settings is None:
                    # Try canonical/provider-specific aliases (e.g. ModelName.value
                    # / .name) via the injected provider — keeps this gaia-free.
                    for alias in prov.get_model_setting_aliases(resolved_model):
                        scoped_settings = model_settings_by_model.get(alias)
                        if scoped_settings is not None:
                            break
                if scoped_settings:
                    run_model_settings_kwargs.update(scoped_settings)
            if "max_tokens" in run_model_settings_kwargs:
                run_model_settings_kwargs["max_tokens"] = prov.clamp_max_tokens(
                    resolved_model,
                    run_model_settings_kwargs["max_tokens"],
                )

            run_config = RunConfig(
                model=resolved_model,
                model_provider=model_provider,
                model_settings=ModelSettings(**run_model_settings_kwargs),
                **tracing_kwargs,
            )

            logger.info(f"[AgentRunner] Running agent '{agent.name if hasattr(agent, 'name') else 'Unknown'}' with model {resolved_model}")
            if max_turns:
                logger.debug(f"  Max turns: {max_turns}")

            # Runner.run expects positional args: agent, prompt, then keyword args
            run_kwargs = {"run_config": run_config}

            # Only add optional parameters if they're provided
            if max_turns is not None:
                run_kwargs["max_turns"] = max_turns
            if context is not None:
                run_kwargs["context"] = context

            return await runner.run(
                agent,
                _prompt_cell[0],
                **run_kwargs
            )

        # Run with or without fallback
        resolved_model = model_key  # Initialize for error handling
        try:
            # Get agent name for logging
            agent_name = getattr(agent, 'name', None)

            async def _execute() -> Any:
                nonlocal resolved_model
                if use_fallback:
                    result, successful_model = await prov.retry_with_fallback(
                        model_key,
                        run_with_model,
                        max_retries_per_model=2,
                        retry_on_validation_failure=True,
                        task_name=agent_name,
                        provider_override=provider_override,
                    )
                    resolved_model = successful_model
                    if successful_model != model_key:
                        logger.info(f"🔄 Agent completed with fallback model: {successful_model} (originally requested: {model_key})")
                    return result
                else:
                    if model_provider is not None:
                        # Caller supplied a pre-resolved provider (e.g. the one an
                        # outer retry_with_fallback already selected). Honor it
                        # directly instead of re-resolving from the model prefix /
                        # provider_override, so the caller's provider-level failover
                        # isn't silently undone. (BOU-1752 review)
                        resolved_model = model_key
                        return await run_with_model(model_key, model_provider)
                    resolved_provider, rm = prov.create_model_provider_for_model(
                        model_key, provider_override=provider_override
                    )
                    resolved_model = rm
                    return await run_with_model(rm, resolved_provider)

            async def _run_with_validation() -> Any:
                """Run the agent, then apply the correction loop if validators are set."""
                if timeout is not None:
                    try:
                        raw_result = await asyncio.wait_for(_execute(), timeout=timeout)
                    except asyncio.TimeoutError:
                        logger.error(
                            "[AgentRunner] Agent '%s' timed out after %ds | model=%s",
                            agent_name or "Unknown",
                            timeout,
                            resolved_model,
                        )
                        raise TimeoutError(
                            f"Agent '{agent_name or 'Unknown'}' timed out after {timeout}s (model={resolved_model})"
                        )
                else:
                    raw_result = await _execute()

                # Fast path: no validators → return immediately (zero-overhead no-op)
                if not output_validators:
                    return raw_result

                # Resolve the Langfuse prompt name/version for the gap event.
                # The SDK Agent passed to run() does NOT carry this metadata (it
                # lives on the wrapper agent before as_openai_agent()), so prefer
                # the explicit args, then fall back to the injected resolver the
                # host wires (gaia reads the Langfuse prompt contextvars).
                eff_prompt_name = prompt_name
                eff_prompt_version = prompt_version
                if eff_prompt_name is None or eff_prompt_version is None:
                    resolver = get_langfuse_prompt_resolver()
                    if resolver is not None:
                        try:
                            _rn, _rv = resolver()
                            eff_prompt_name = eff_prompt_name or _rn
                            eff_prompt_version = eff_prompt_version or _rv
                        except Exception:  # noqa: BLE001
                            pass

                # Extract structured output for validation; fall back to the raw result
                structured = AgentRunner.extract_structured_output(raw_result)
                result_to_validate = structured if structured is not None else raw_result

                # Collect violations across all validators
                all_violations = [
                    v
                    for val in output_validators
                    for v in val.validate(result_to_validate)
                ]
                if not all_violations:
                    return raw_result

                # Emit prompt-gap event on first violation (best-effort)
                try:
                    await AgentRunner._emit_prompt_correction_event(
                        agent=agent,
                        violations=all_violations,
                        attempt=0,
                        prompt_name=eff_prompt_name,
                        prompt_version=eff_prompt_version,
                    )
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "[AgentRunner] _emit_prompt_correction_event failed (non-fatal)",
                        exc_info=True,
                    )

                # Correction loop
                current_result = raw_result
                current_validated = result_to_validate
                for attempt in range(max_corrections):
                    # Build correction prompt
                    violation_lines = []
                    for v in all_violations:
                        line = f"- {v.field}: {v.message}"
                        if v.suggested_fix:
                            line += f" (suggested fix: {v.suggested_fix})"
                        violation_lines.append(line)
                    correction_suffix = (
                        "\n\nYOUR PREVIOUS OUTPUT WAS INVALID:\n"
                        + "\n".join(violation_lines)
                        + "\nReturn a corrected, complete response."
                    )
                    _prompt_cell[0] = prompt + correction_suffix

                    logger.info(
                        "[AgentRunner] Correction attempt %d/%d for agent '%s': %d violation(s)",
                        attempt + 1,
                        max_corrections,
                        agent_name or "Unknown",
                        len(all_violations),
                    )

                    if timeout is not None:
                        try:
                            current_result = await asyncio.wait_for(
                                _execute(), timeout=timeout
                            )
                        except asyncio.TimeoutError:
                            logger.error(
                                "[AgentRunner] Agent '%s' timed out on correction attempt %d",
                                agent_name or "Unknown",
                                attempt + 1,
                            )
                            raise TimeoutError(
                                f"Agent '{agent_name or 'Unknown'}' timed out after {timeout}s (correction attempt {attempt + 1})"
                            )
                    else:
                        current_result = await _execute()

                    structured = AgentRunner.extract_structured_output(current_result)
                    current_validated = structured if structured is not None else current_result

                    all_violations = [
                        v
                        for val in output_validators
                        for v in val.validate(current_validated)
                    ]
                    if not all_violations:
                        logger.info(
                            "[AgentRunner] Correction succeeded on attempt %d for agent '%s'",
                            attempt + 1,
                            agent_name or "Unknown",
                        )
                        return current_result

                # Budget exhausted — apply deterministic normalize() fallbacks
                logger.warning(
                    "[AgentRunner] Agent '%s' still has %d violation(s) after %d correction(s); "
                    "applying normalize() fallbacks: %s",
                    agent_name or "Unknown",
                    len(all_violations),
                    max_corrections,
                    [f"{v.field}: {v.message}" for v in all_violations],
                )
                normalized = current_validated
                for val in output_validators:
                    try:
                        normalized = val.normalize(normalized)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "[AgentRunner] normalize() failed for validator '%s' (non-fatal)",
                            getattr(val, "name", repr(val)),
                            exc_info=True,
                        )
                # Re-validate after normalize() — the deterministic fallback is
                # supposed to clear every violation; warn loudly if it didn't.
                residual = [
                    v for val in output_validators for v in val.validate(normalized)
                ]
                if residual:
                    logger.error(
                        "[AgentRunner] normalize() did NOT clear violations for agent '%s': %s",
                        agent_name or "Unknown",
                        [f"{v.field}: {v.kind}" for v in residual],
                    )
                # Emit exhausted event (best-effort)
                try:
                    await AgentRunner._emit_prompt_correction_event(
                        agent=agent,
                        violations=all_violations,
                        attempt=max_corrections,
                        exhausted=True,
                        prompt_name=eff_prompt_name,
                        prompt_version=eff_prompt_version,
                    )
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "[AgentRunner] _emit_prompt_correction_event (exhausted) failed (non-fatal)",
                        exc_info=True,
                    )
                # Preserve the Runner result wrapper: write the normalized object
                # back onto final_output and return the RESULT, not the bare
                # structured object — callers (e.g. CharacterGeneratorAgent) read
                # result.final_output, which would otherwise be the un-normalized
                # value and re-trip "returned no output"/invalid downstream.
                if hasattr(current_result, "final_output"):
                    try:
                        current_result.final_output = normalized
                        return current_result
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "[AgentRunner] could not set final_output on result; returning normalized object",
                            exc_info=True,
                        )
                return normalized

            return await _run_with_validation()
        except TimeoutError:
            raise  # Let timeout propagate cleanly to callers
        except Exception as e:
            # Prepare error context - be careful not to hide the actual error
            try:
                agent_name = getattr(agent, 'name', 'Unknown agent')
            except Exception:
                agent_name = 'Unknown agent (error getting name)'

            error_type = type(e).__name__
            error_details = {
                'agent': agent_name,
                'model': resolved_model,
                'error_type': error_type,
                'error_message': str(e),
                'has_partial_result': 'result' in locals()
            }

            # Special handling for OutputGuardrail exceptions
            if "OutputGuardrail" in error_type or "OutputGuardrail" in str(e):
                logger.error(
                    "OutputGuardrail validation failed - Agent: '%(agent)s', Model: '%(model)s', Error: %(error_message)s",
                    error_details
                )
                logger.error(
                    "This typically means the agent's output didn't match the expected format or contained disallowed content"
                )

                # Collect and log guardrail context if available
                context = AgentRunner._collect_guardrail_context(e)
                if context:
                    error_details['guardrail_context'] = context
                    logger.error("Guardrail exception context: %s", context)

                # Log prompt preview for debugging
                prompt_preview = AgentRunner._safe_preview(prompt)
                if prompt_preview:
                    error_details['prompt_preview'] = prompt_preview
                    logger.debug("Prompt excerpt when guardrail triggered: %s", prompt_preview)

                # Log the partial result if available for debugging
                if error_details['has_partial_result']:
                    partial_result = AgentRunner._safe_preview(result)
                    error_details['partial_result'] = partial_result
                    logger.debug("Partial result before guardrail: %s", partial_result)

                # Create enhanced exception with all context
                enhanced_error = RuntimeError(
                    f"Agent execution failed with {error_type}: {error_details}"
                )
                enhanced_error.__cause__ = e
                enhanced_error.error_details = error_details
                raise enhanced_error from e
            else:
                # General error handling with structured context
                logger.error(
                    "Agent execution error - Agent: '%(agent)s', Model: '%(model)s', Type: %(error_type)s, Message: %(error_message)s",
                    error_details,
                    exc_info=True
                )

                # Add any partial results to error details
                if error_details['has_partial_result']:
                    error_details['partial_result'] = AgentRunner._safe_preview(result)

                # Re-raise with enhanced context
                enhanced_error = RuntimeError(
                    f"Agent execution failed: {error_details}"
                )
                enhanced_error.__cause__ = e
                enhanced_error.error_details = error_details
                raise enhanced_error from e

    @staticmethod
    async def _emit_prompt_correction_event(
        agent: Any,
        violations: list,
        attempt: int,
        exhausted: bool = False,
        prompt_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
    ) -> None:
        """Record a prompt-gap event for each violation (best-effort, never raises).

        The actual recording (signature + DB write) is host-specific, so it is
        delegated to the injected ``prompt_correction_emitter`` the host wires via
        ``configure_agentrunner``. When no emitter is configured this is a no-op,
        keeping the package free of any storage dependency.

        prompt_name/prompt_version are resolved by the caller (explicit run() args
        or the injected langfuse prompt resolver) because the SDK Agent passed in
        does NOT carry that metadata — it lives on the wrapper agent. Agent
        attributes are a last-resort fallback handled by the emitter.
        """
        emitter = get_prompt_correction_emitter()
        if emitter is None:
            return
        await emitter(
            agent=agent,
            violations=violations,
            attempt=attempt,
            exhausted=exhausted,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
        )

    @staticmethod
    def run_streamed(
        agent: Agent,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        tool_choice: Optional[str] = "auto",
        parallel_tool_calls: bool = False,
        max_turns: Optional[int] = None,
        context: Optional[Any] = None,
        trace_group_id: Optional[str] = None,
        trace_metadata: Optional[Dict[str, Any]] = None,
        provider_override: Optional[str] = None,
        **kwargs,
    ):
        """Run an agent using Runner.run_streamed for incremental consumption."""
        # Provider first so a lazy host bootstrap configures the trace-processor
        # factory before configure_agents_sdk() reads it (BOU-1778 review).
        prov = get_active_provider()
        configure_agents_sdk()
        runner = Runner()

        model_key = model or getattr(agent, 'model', None)
        if not model_key:
            raise ValueError("No model specified and agent has no default model")

        agent = AgentRunner._normalize_tool_settings_for_agent(agent)

        model_provider, resolved_model = prov.create_model_provider_for_model(
            model_key, provider_override=provider_override
        )

        # Set request context for rate limit logging
        agent_name = getattr(agent, 'name', None) or 'unknown'
        set_request_context(agent_name=agent_name, model=resolved_model)

        model_settings_kwargs = {
            "temperature": temperature,
        }

        # Only add tool_choice and parallel_tool_calls if agent has tools
        if hasattr(agent, 'tools') and agent.tools:
            model_settings_kwargs["tool_choice"] = tool_choice
            model_settings_kwargs["parallel_tool_calls"] = parallel_tool_calls

        for key, value in kwargs.items():
            if key not in ['max_turns', 'trace_group_id', 'trace_metadata']:
                model_settings_kwargs[key] = value

        # Resolve tracing from contextvar defaults when not explicitly provided
        trace_group_id, trace_metadata = AgentRunner._resolve_tracing(
            trace_group_id, trace_metadata
        )

        # Build tracing fields
        workflow_name = agent_name if agent_name != 'unknown' else "Agent workflow"
        tracing_kwargs: Dict[str, Any] = {"workflow_name": workflow_name}
        if trace_group_id is not None:
            tracing_kwargs["group_id"] = trace_group_id
        # Inject agent name tag so each trace is tagged with the agent that ran
        if trace_metadata is None:
            trace_metadata = {}
        agent_tag = f"agent:{workflow_name}"
        existing_tags = trace_metadata.get("tags", [])
        if isinstance(existing_tags, list) and agent_tag not in existing_tags:
            trace_metadata["tags"] = existing_tags + [agent_tag]
        # Inject provider tag
        provider_type = prov.get_provider_for_model(resolved_model)
        provider_tag = f"provider:{provider_type}"
        tags = trace_metadata["tags"]
        tags = [t for t in tags if not t.startswith("provider:")]
        tags.append(provider_tag)
        trace_metadata["tags"] = tags
        tracing_kwargs["trace_metadata"] = trace_metadata

        run_config = RunConfig(
            model=resolved_model,
            model_provider=model_provider,
            model_settings=ModelSettings(**model_settings_kwargs),
            **tracing_kwargs,
        )

        run_kwargs = {"run_config": run_config}
        if max_turns is not None:
            run_kwargs["max_turns"] = max_turns
        if context is not None:
            run_kwargs["context"] = context

        logger.info(f"[AgentRunner] Running agent '{agent_name}' with model {resolved_model}")
        return runner.run_streamed(
            agent,
            prompt,
            **run_kwargs,
        )

    @staticmethod
    def extract_text_from_stream_event(event: Any) -> tuple[Optional[str], bool, bool]:
        """Extract text from a streamed event.

        Returns:
            (text, is_delta, is_message_item)
        """
        try:
            from agents import RawResponsesStreamEvent, RunItemStreamEvent, ItemHelpers
        except Exception:  # noqa: BLE001
            RawResponsesStreamEvent = None  # type: ignore[assignment]
            RunItemStreamEvent = None  # type: ignore[assignment]
            ItemHelpers = None  # type: ignore[assignment]

        try:
            from openai.types.responses import ResponseTextDeltaEvent
        except Exception:  # noqa: BLE001
            ResponseTextDeltaEvent = None  # type: ignore[assignment]

        if RawResponsesStreamEvent is not None and isinstance(event, RawResponsesStreamEvent) and hasattr(event, "data"):
            raw_data = event.data
            if ResponseTextDeltaEvent is not None and isinstance(raw_data, ResponseTextDeltaEvent):
                text_delta = getattr(raw_data, "delta", "") or ""
                if text_delta:
                    return text_delta, True, False
            return None, False, False

        if RunItemStreamEvent is not None and isinstance(event, RunItemStreamEvent):
            item = getattr(event, "item", None)
            item_type = getattr(item, "type", "")
            if item_type == "message_output_item" and ItemHelpers is not None:
                try:
                    message_text = ItemHelpers.text_message_output(item)
                except Exception:  # noqa: BLE001
                    message_text = None
                if message_text:
                    return message_text, False, True

        return None, False, False

    @staticmethod
    async def iter_streamed_text(streamed_run):
        """Yield streamed text chunks with a flag indicating delta origin.

        Automatically strips <think>...</think> chain-of-thought blocks.
        """
        streamed_any = False
        think_filter = ThinkTagFilter()
        async for event in streamed_run.stream_events():
            text, is_delta, is_message_item = AgentRunner.extract_text_from_stream_event(event)
            if not text:
                continue
            if is_message_item and streamed_any:
                continue
            text = think_filter.feed(text)
            if not text:
                continue
            streamed_any = True
            yield text, is_delta

    @staticmethod
    async def iter_streamed_text_with_fallback(
        agent: Agent,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        tool_choice: Optional[str] = "auto",
        parallel_tool_calls: bool = False,
        max_turns: Optional[int] = None,
        context: Optional[Any] = None,
        task_name: Optional[str] = None,
        trace_group_id: Optional[str] = None,
        trace_metadata: Optional[Dict[str, Any]] = None,
        provider_override: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[Tuple[str, bool, str]]:
        """Stream text from an agent with automatic model fallback on provider errors.

        This method handles rate limits and provider errors by automatically falling
        back to alternative models. Fallback only happens BEFORE any content is
        streamed - if content has already been sent to the client, the error is
        propagated to avoid duplicate/incoherent output.

        Args:
            agent: The agent to run
            prompt: The prompt to send to the agent
            model: Primary model to use
            temperature: Temperature for generation (default 0.7)
            tool_choice: Tool choice behavior (default "auto")
            parallel_tool_calls: Whether to allow parallel tool calls
            max_turns: Maximum number of turns for the agent
            context: Optional context object to pass to tools and hooks
            task_name: Optional name for logging (defaults to agent name)
            trace_group_id: Optional group ID for OpenAI tracing
            trace_metadata: Optional metadata dict for OpenAI tracing
            provider_override: Optional provider slug to force for the primary model
                     (e.g. "base10"). Only applies to the primary model call, not
                     fallback models.
            **kwargs: Additional keyword arguments for ModelSettings

        Yields:
            Tuples of (text, is_delta, model_used) for each streamed chunk.
            The model_used field indicates which model successfully produced the output.

        Raises:
            Exception: If all models fail or if streaming fails mid-way
        """
        prov = get_active_provider()
        agent_name = task_name or getattr(agent, 'name', None) or 'unknown'
        primary_model = model

        # Build list of models to try: primary + fallbacks
        models_to_try = [primary_model] + prov.get_fallback_models(primary_model)
        failed_models: List[str] = []
        rate_limit_failures = 0
        last_error: Optional[Exception] = None

        for current_model in models_to_try:
            # Log fallback if not the primary model
            if current_model != primary_model:
                logger.info(f"[{agent_name}] Falling back to {current_model}")

            has_streamed = False

            try:
                # Only apply provider_override to the primary model;
                # fallback models use their own default providers.
                effective_override = provider_override if current_model == primary_model else None
                streamed_run = AgentRunner.run_streamed(
                    agent=agent,
                    prompt=prompt,
                    model=current_model,
                    temperature=temperature,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                    max_turns=max_turns,
                    context=context,
                    trace_group_id=trace_group_id,
                    trace_metadata=trace_metadata,
                    provider_override=effective_override,
                    **kwargs,
                )

                async for text, is_delta in AgentRunner.iter_streamed_text(streamed_run):
                    has_streamed = True
                    yield text, is_delta, current_model

                # Success! Log if we used a fallback model
                if failed_models:
                    failures_str = " → ".join(failed_models)
                    logger.info(f"[{agent_name}] ✅ {current_model} (fallback after: {failures_str})")
                else:
                    logger.debug(f"[{agent_name}] ✅ {current_model}")

                return  # Successfully completed

            except Exception as e:
                last_error = e

                # If we already streamed content to the client, we can't retry with a
                # fallback model - that would result in duplicate/incoherent output.
                if has_streamed:
                    logger.error(
                        f"[{agent_name}] Failed mid-stream; cannot fallback",
                        exc_info=True,
                    )
                    raise

                if prov.is_provider_error(e):
                    # Provider error before streaming started - try next model
                    reason = "rate-limited" if prov.is_rate_limit_error(e) else type(e).__name__
                    failed_models.append(f"{current_model}({reason})")
                    logger.exception("[%s] %s failed: %s", agent_name, current_model, reason)

                    # Add delay for rate limit errors
                    if prov.is_rate_limit_error(e):
                        rate_limit_failures += 1
                        delay = min(
                            prov.default_rate_limit_delay_seconds * (2 ** (rate_limit_failures - 1)),
                            prov.max_rate_limit_delay_seconds,
                        )
                        logger.info(f"[{agent_name}] Rate limited; waiting {delay:.2f}s before fallback")
                        await asyncio.sleep(delay)

                    # Continue to next model
                    continue
                else:
                    # Non-provider error (e.g., validation error) - don't fallback, re-raise
                    logger.error(f"[{agent_name}] Non-provider error: {e}", exc_info=True)
                    raise

        # All models failed
        failures_str = " → ".join(failed_models)
        logger.error(f"[{agent_name}] ❌ All models failed: {failures_str}")
        raise Exception(f"All models failed for {agent_name}: {failures_str}") from last_error

    @staticmethod
    def _coerce_item_to_text(item: Any) -> Optional[str]:
        """Attempt to extract human-readable text from varied agent output structures."""
        if item is None:
            return None

        if isinstance(item, str):
            return item

        # Handle lists at top level by recursively processing items
        if isinstance(item, list):
            pieces = [
                piece
                for piece in (AgentRunner._coerce_item_to_text(part) for part in item)
                if piece
            ]
            if pieces:
                return "".join(pieces)

        # Attempt to use agents.ItemHelpers when available
        try:
            from agents.items import ItemHelpers, MessageOutputItem  # type: ignore
        except Exception:  # noqa: BLE001
            ItemHelpers = None  # type: ignore[assignment]
            MessageOutputItem = None  # type: ignore[assignment]

        if MessageOutputItem is not None and isinstance(item, MessageOutputItem):  # type: ignore[arg-type]
            try:
                text = ItemHelpers.text_message_output(item)  # type: ignore[union-attr]
                if text:
                    return text
            except Exception:  # noqa: BLE001
                pass

        raw_item = getattr(item, "raw_item", None)
        if raw_item is not None and raw_item is not item:
            extracted = AgentRunner._coerce_item_to_text(raw_item)
            if extracted:
                return extracted

        if hasattr(item, "model_dump"):
            try:
                dumped = item.model_dump(exclude_unset=True)
                extracted = AgentRunner._coerce_item_to_text(dumped)
                if extracted:
                    return extracted
            except Exception:  # noqa: BLE001
                pass

        if isinstance(item, dict):
            for key in ("text", "content", "output"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            # Recursively inspect nested content structures
            for value in item.values():
                extracted = AgentRunner._coerce_item_to_text(value)
                if extracted:
                    return extracted

        content = getattr(item, "content", None)
        if isinstance(content, str):
            if content.strip():
                return content
        elif isinstance(content, list):
            pieces = [
                piece
                for piece in (AgentRunner._coerce_item_to_text(part) for part in content)
                if piece
            ]
            if pieces:
                return "".join(pieces)

        text_attr = getattr(item, "text", None)
        if isinstance(text_attr, str) and text_attr.strip():
            return text_attr

        # Some response objects expose a 'value' field with text
        value_attr = getattr(item, "value", None)
        if isinstance(value_attr, str) and value_attr.strip():
            return value_attr

        return None

    @staticmethod
    def _safe_preview(value: Any, max_len: int = 800) -> Optional[str]:
        """Generate a safe, truncated string preview for logging."""
        if value is None:
            return None
        try:
            text = repr(value)
        except Exception:
            text = f"<unrepresentable {type(value).__name__}>"
        if len(text) > max_len:
            return f"{text[:max_len]}... (truncated {len(text) - max_len} chars)"
        return text

    @staticmethod
    def _collect_guardrail_context(exception: Exception) -> Dict[str, str]:
        """Collect useful attributes from a guardrail exception for debugging."""
        context: Dict[str, str] = {}
        candidate_attrs = [
            "raw_output",
            "validated_output",
            "output",
            "response",
            "failures",
            "failure",
            "errors",
            "metadata",
            "validator",
            "schema",
        ]
        for attr in candidate_attrs:
            if hasattr(exception, attr):
                value = getattr(exception, attr)
                if value:
                    context[attr] = AgentRunner._safe_preview(value)

        # Include args if present
        if getattr(exception, "args", None):
            context["args"] = AgentRunner._safe_preview(exception.args)

        # Include any additional non-callable public attributes
        if hasattr(exception, "__dict__"):
            for key, value in exception.__dict__.items():
                if key.startswith("_") or key in context:
                    continue
                context[key] = AgentRunner._safe_preview(value)

        return {k: v for k, v in context.items() if v}

    @staticmethod
    def is_structured_output_parse_error(exception: Exception) -> bool:
        """Return whether an exception chain came from structured-output parsing."""
        parse_markers = (
            "invalid json when parsing",
            "type=json_invalid",
            "json_invalid",
            "validate_json",
            "jsondecodeerror",
            "control character",
            # An after-validator that rejects a syntactically-valid but semantically
            # incomplete structured response (e.g. a truncated diff) raises with this
            # phrase, so callers treat it as a recoverable parse/incomplete failure.
            "incomplete structured output",
        )
        current: Optional[BaseException] = exception
        while current is not None:
            message = str(current).lower()
            if any(marker in message for marker in parse_markers):
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def extract_tool_result(result: Any) -> Optional[Any]:
        """Extract tool call result from agent response.
        
        Args:
            result: The result from agent.run()
            
        Returns:
            The extracted tool result content, or None if not found
        """
        if hasattr(result, 'messages') and result.messages:
            for message in result.messages:
                if hasattr(message, 'tool_results') and message.tool_results:
                    for tool_result in message.tool_results:
                        if hasattr(tool_result, 'content'):
                            return tool_result.content
        return None
    
    @staticmethod
    def extract_text_response(result: Any) -> Optional[str]:
        """Extract text response from agent result.
        
        Args:
            result: The result from agent.run()
            
        Returns:
            The extracted text response, or None if not found
        """
        if result is None:
            return None

        # Prefer the structured final_output, if present.
        if hasattr(result, "final_output") and result.final_output is not None:
            extracted = AgentRunner._coerce_item_to_text(result.final_output)
            if extracted:
                return extracted

        # Many agent results expose the generated items list; try harvesting text from it.
        new_items = getattr(result, "new_items", None)
        if new_items:
            try:
                from agents.items import ItemHelpers  # type: ignore

                text = ItemHelpers.text_message_outputs(new_items)  # type: ignore[union-attr]
                if text:
                    return text
            except Exception:  # noqa: BLE001
                pass

            for item in new_items:
                extracted = AgentRunner._coerce_item_to_text(item)
                if extracted:
                    return extracted

        # Fall back to message content lists
        if hasattr(result, "messages") and result.messages:
            for message in result.messages:
                extracted = AgentRunner._coerce_item_to_text(message)
                if extracted:
                    return extracted

        if isinstance(result, str):
            return result

        return None
    
    @staticmethod
    def extract_structured_output(result: Any) -> Optional[Any]:
        """Extract structured output from agent response.

        Checks in order:
        1. final_output (Pydantic model from output_type)
        2. Tool results
        3. Text response parsed as JSON

        Args:
            result: The result from agent.run()

        Returns:
            The extracted structured data (Pydantic model or dict), or None if not found
        """
        from pydantic import BaseModel

        # First check for final_output (set when agent has output_type)
        # This is the preferred path for structured output
        if hasattr(result, "final_output") and result.final_output is not None:
            final = result.final_output
            # If it's already a Pydantic model, return it directly
            if isinstance(final, BaseModel):
                return final
            # If it's a dict, return it as-is
            if isinstance(final, dict):
                return final

        # Try to get tool result
        tool_result = AgentRunner.extract_tool_result(result)
        if tool_result:
            return tool_result

        # If no tool result, try to parse text response as JSON
        text_response = AgentRunner.extract_text_response(result)
        if text_response:
            try:
                # Try to extract JSON from the text
                # Handle cases where model returns explanation with JSON
                if '{' in text_response and '}' in text_response:
                    # Find the JSON part
                    start = text_response.find('{')
                    end = text_response.rfind('}') + 1
                    json_str = text_response[start:end]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse text response as JSON: {text_response[:200]}")

        return None
