# agentrunner

Host-agnostic LLM **agent execution** on top of the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python).

`AgentRunner` wraps the Agents SDK `Runner` with the production concerns you'd
otherwise rewrite per project:

- **Model fallback** on rate-limit / provider errors (policy supplied by you)
- **Streaming** with automatic `<think>…</think>` chain-of-thought filtering
- **Structured-output extraction** from varied SDK result shapes, with tolerant
  JSON sanitization/repair
- **Self-correction**: validate output, re-prompt on violations, then a
  deterministic `normalize()` fallback
- **Tracing hooks** (e.g. Langfuse) and per-request context — all injected, no
  hard dependency
- **Token clamping**, timeouts, and structured error enrichment

The package depends only on `openai-agents`, `openai`, and `pydantic`. It has
**no knowledge of any host application** — you inject the model layer and
optional hooks via a small protocol.

## Install

```bash
pip install boundless-agentrunner
```

The distribution is published as `boundless-agentrunner` (the `agentrunner` name
on PyPI was taken); the **import name is still `agentrunner`**:

```python
import agentrunner
```

## Quick start

`AgentRunner` resolves its model layer from an injected
`ModelClientProvider`. Configure it once at startup:

```python
from agentrunner import AgentRunner, configure_agentrunner, ModelClientProvider

class MyProvider:  # implements ModelClientProvider (a typing.Protocol)
    default_rate_limit_delay_seconds = 1.0
    max_rate_limit_delay_seconds = 8.0

    def create_model_provider_for_model(self, model_key, provider_override=None):
        # return (agents-SDK ModelProvider, resolved_model_id)
        ...
    async def retry_with_fallback(self, model_key, run_with_model, **kw):
        # run `run_with_model(resolved_model, provider)` with your fallback policy;
        # return (result, successful_model)
        ...
    def get_fallback_models(self, model): return []
    def get_provider_for_model(self, model): return "myprovider"
    def get_model_setting_aliases(self, resolved_model): return []
    def clamp_max_tokens(self, model, max_tokens): return max_tokens
    def is_rate_limit_error(self, exc): ...
    def is_provider_error(self, exc): ...

configure_agentrunner(model_provider=MyProvider())

# then anywhere:
from agents import Agent
result = await AgentRunner.run(Agent(name="demo", model="...", instructions="..."), "hello")
```

### Batteries-included: OpenRouter

Don't want to write a provider? Use the bundled OpenRouter one — a single
`OPENROUTER_API_KEY` reaches every model family (Claude, GPT, Llama, Qwen,
DeepSeek, Gemini, …) through OpenRouter's OpenAI-compatible endpoint, with model
fallback built in:

```python
import os
from agentrunner import configure_agentrunner
from agentrunner.providers import OpenRouterModelClientProvider

configure_agentrunner(
    model_provider=OpenRouterModelClientProvider(
        # api_key defaults to $OPENROUTER_API_KEY
        fallback_models=["anthropic/claude-sonnet-4.5", "openai/gpt-4o-mini"],
    )
)

from agents import Agent
result = await AgentRunner.run(
    Agent(name="demo", model="anthropic/claude-sonnet-4.5", instructions="..."),
    "hello",
)
```

Models are addressed by their OpenRouter slug (`vendor/model`). The provider
forces Chat Completions (OpenRouter doesn't fully implement the Responses API),
honors the validation-retry budget, and classifies rate-limit/provider errors
for fallback.

### Lazy configuration

If you can't configure at startup, register a bootstrap that runs on first use:

```python
from agentrunner.runtime import register_bootstrap
register_bootstrap(lambda: configure_agentrunner(model_provider=MyProvider()))
```

### Optional hooks

`configure_agentrunner` also accepts:

- `trace_processor_factory` — returns an Agents-SDK trace processor (e.g. Langfuse)
- `prompt_correction_emitter` — async callback to record self-correction events
- `langfuse_prompt_resolver` — returns `(prompt_name, prompt_version)` for events

## Output validation / self-correction

```python
from agentrunner.output_validation import BoundedTextValidator

result = await AgentRunner.run(
    agent, prompt,
    output_validators=[BoundedTextValidator("title", 255)],
    max_corrections=1,
)
```

On a validation failure the agent is re-prompted up to `max_corrections` times;
if it still fails, each validator's deterministic `normalize()` bounds the value.

## Status

`0.2.0` ships the execution engine, the `ModelClientProvider` seam, and the
batteries-included **OpenRouter** provider (single `OPENROUTER_API_KEY`, all
model families) so adopters can start without writing a provider.

## License

MIT.
