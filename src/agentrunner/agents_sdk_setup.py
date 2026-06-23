"""
One-time OpenAI Agents SDK patches applied at process startup.

Consolidates all monkey-patches in a single module so they're easy to
find, audit, and remove when the upstream SDK fixes them.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)

_applied = False
_OPENAI_TRACING_ALLOWED_USAGE_KEYS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "input_tokens_details",
        "output_tokens_details",
    }
)


def configure_agents_sdk() -> None:
    """Apply all one-time SDK patches.  Safe to call multiple times."""
    global _applied
    if _applied:
        return
    _applied = True

    _patch_openai_tracing_payload_compatibility()
    _disable_openai_tracing()
    _configure_trace_processor()
    _patch_json_validation()


def _serialize_trace_tags(tags: Any) -> str | None:
    """Normalize trace tags to the string form expected by OpenAI tracing."""
    if isinstance(tags, str):
        normalized = tags.strip()
        return normalized or None

    if not isinstance(tags, list):
        return None

    normalized_tags: list[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        cleaned = tag.strip()
        if cleaned and cleaned not in normalized_tags:
            normalized_tags.append(cleaned)

    if not normalized_tags:
        return None

    return ",".join(normalized_tags)


def _patch_openai_tracing_payload_compatibility() -> None:
    """Patch SDK payload export to match the current OpenAI tracing schema."""
    try:
        from agents.tracing.processors import BackendSpanExporter
    except ImportError:
        logger.exception("Could not import BackendSpanExporter for tracing patch")
        return
    except Exception:
        logger.exception("Failed to import BackendSpanExporter for tracing patch")
        return

    if getattr(BackendSpanExporter, "_gaia_compat_patch_applied", False):
        return

    original_sanitize = BackendSpanExporter._sanitize_for_openai_tracing_api

    def patched_sanitize_for_openai_tracing_api(self, payload_item: dict[str, Any]) -> dict[str, Any]:
        sanitized = original_sanitize(self, payload_item)

        metadata = sanitized.get("metadata")
        if not isinstance(metadata, dict) or "tags" not in metadata:
            return sanitized

        normalized_tags = _serialize_trace_tags(metadata.get("tags"))
        sanitized_metadata = dict(metadata)
        if normalized_tags is None:
            sanitized_metadata.pop("tags", None)
        else:
            sanitized_metadata["tags"] = normalized_tags

        sanitized_payload = dict(sanitized)
        sanitized_payload["metadata"] = sanitized_metadata
        return sanitized_payload

    BackendSpanExporter._sanitize_for_openai_tracing_api = patched_sanitize_for_openai_tracing_api
    BackendSpanExporter._OPENAI_TRACING_ALLOWED_USAGE_KEYS = _OPENAI_TRACING_ALLOWED_USAGE_KEYS
    BackendSpanExporter._gaia_compat_patch_applied = True
    logger.debug("Patched BackendSpanExporter for tracing payload compatibility")


def _disable_openai_tracing() -> None:
    """Disable the SDK's built-in OpenAI trace export.

    We use ``set_trace_processors`` in ``_configure_trace_processor`` to
    *replace* the default processors (which include ``BackendSpanExporter``)
    when the host injected a trace-processor factory. If none was injected we
    disable tracing entirely so the default exporter never phones home.
    """
    from agentrunner.runtime import get_trace_processor_factory

    if get_trace_processor_factory() is None:
        try:
            from agents import set_tracing_disabled
            set_tracing_disabled(True)
            logger.info("[OK] All Agents SDK tracing disabled (no trace processor configured)")
        except Exception:
            logger.warning("Failed to disable Agents SDK tracing", exc_info=True)
    else:
        logger.info("[OK] OpenAI default tracing will be replaced by the injected processor")


def _configure_trace_processor() -> None:
    """Register the host-injected trace processor, if one was configured.

    The host calls ``configure_agentrunner(trace_processor_factory=...)`` to
    supply, e.g., a Langfuse processor. When no factory is configured, SDK
    tracing has already been disabled in ``_disable_openai_tracing``.
    """
    from agentrunner.runtime import get_trace_processor_factory

    factory = get_trace_processor_factory()
    if factory is None:
        return

    try:
        from agents import set_trace_processors

        # Replace the default processors (which include BackendSpanExporter
        # that sends to OpenAI) with only the injected processor.
        set_trace_processors([factory()])
        logger.info("[OK] Injected trace processor registered (replaced default OpenAI exporter)")
    except Exception:
        logger.warning("Injected trace processor setup failed", exc_info=True)


def _patch_json_validation() -> None:
    """Patch ``agents.util._json.validate_json`` to sanitise malformed JSON.

    Some LLM providers return JSON with control characters or other
    defects.  The upstream SDK rejects these outright; this patch adds a
    sanitisation fallback before raising.
    """
    try:
        import agents.util._json as agents_json
        from pydantic.type_adapter import TypeAdapter
        from agentrunner.json_sanitizer import (
            repair_truncated_json,
            sanitize_json_string,
        )

        if hasattr(agents_json, "_original_validate_json"):
            return  # already patched

        agents_json._original_validate_json = agents_json.validate_json

        def allows_truncated_json_repair(type_adapter: TypeAdapter) -> bool:
            model_type = getattr(type_adapter, "_type", None)
            if getattr(model_type, "__allow_truncated_json_repair__", False):
                return True
            core_schema = getattr(type_adapter, "core_schema", None)
            if isinstance(core_schema, dict):
                schema_cls = core_schema.get("cls")
                return bool(getattr(schema_cls, "__allow_truncated_json_repair__", False))
            return False

        def patched_validate_json(
            json_str: str, type_adapter: TypeAdapter, partial: bool
        ):
            try:
                return agents_json._original_validate_json(
                    json_str, type_adapter, partial
                )
            except Exception as e:
                if "Invalid JSON" not in str(e) and "control character" not in str(e):
                    raise
                logger.debug(
                    "JSON validation failed, attempting recovery: %s",
                    str(e)[:200],
                )
                # Build recovery candidates defensively: a transform that itself
                # raises must never mask the ORIGINAL validation error.
                #
                #  1. Cheap control-char / trailing-comma sanitisation.
                #  2. EOF-truncation repair (whitespace flood / unterminated string
                #     / unbalanced braces — BOU-1145), and 3. the two combined.
                #     Repair fabricates closing tokens, so it runs only for FINAL
                #     validation: on a partial (mid-stream) chunk, completing the
                #     structure would surface a half-emitted value as if complete.
                candidates: list[str] = []
                try:
                    candidates.append(sanitize_json_string(json_str))
                except Exception:
                    pass
                if not partial and allows_truncated_json_repair(type_adapter):
                    try:
                        repaired = repair_truncated_json(json_str)
                        candidates.append(repaired)
                        candidates.append(sanitize_json_string(repaired))
                    except Exception:
                        pass
                for candidate in candidates:
                    if candidate == json_str:
                        continue
                    try:
                        return agents_json._original_validate_json(
                            candidate, type_adapter, partial
                        )
                    except Exception:
                        continue
                # Every candidate failed — surface the original error.
                raise e

        agents_json.validate_json = patched_validate_json
        logger.debug("Patched agents library JSON validation with sanitization")

    except ImportError:
        logger.exception("Could not import agents.util._json for patching")
    except Exception as e:
        logger.exception('Failed to patch JSON validation')
