"""LAMaS as a LiteLLM provider: one adapter that reaches every framework built on LiteLLM.

    import litellm, lamas
    from lamas.adapters.litellm_provider import register

    register(lamas.Runner(policy="auto", backend="http://localhost:8000/v1"))
    reply = litellm.completion(model="lamas/auto", messages=[{"role": "user", "content": "What is 17 * 24?"}])

After ``register`` any library that calls LiteLLM can use the model name ``lamas/<policy>``;
DSPy, for example: ``dspy.LM("lamas/auto")``. No server is involved: the runner lives in the
same process. (With a running ``lamas serve`` no adapter is needed at all: point the framework's
OpenAI-compatible client at it.)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import litellm
from litellm import CustomLLM, ModelResponse

from ..api import Runner


def _query(messages: List[Dict[str, Any]]) -> str:
    def text(content: Any) -> str:
        if isinstance(content, list):
            return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        return str(content or "")

    system = "\n".join(text(m.get("content")) for m in messages if m.get("role") == "system").strip()
    user = [text(m.get("content")) for m in messages if m.get("role") == "user"]
    question = user[-1].strip() if user else ""
    return f"{system}\n\n{question}" if system else question


class LamasLLM(CustomLLM):
    """LiteLLM custom handler that answers with a LAMaS runner."""

    def __init__(self, runner: Runner):
        super().__init__()
        self.runner = runner
        self._runners: Dict[str, Runner] = {}
        self.last_trace = None

    def _runner_for(self, model: str) -> Runner:
        name = model.split("/", 1)[1] if "/" in model else model
        if name in ("auto", "", self.runner.policy.name):
            return self.runner
        if name not in self._runners:
            self._runners[name] = Runner(policy=name if "/" not in name else name, backend=self.runner.backend)
        return self._runners[name]

    def _response(self, model: str, result: Any) -> ModelResponse:
        trace = result.trace
        self.last_trace = trace
        return ModelResponse(
            id=f"lamas-{int(time.time() * 1000)}", model=model, created=int(time.time()),
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": result.answer}}],
            usage={"prompt_tokens": trace.usage.prompt_tokens, "completion_tokens": trace.usage.completion_tokens,
                   "total_tokens": trace.usage.prompt_tokens + trace.usage.completion_tokens},
        )

    async def acompletion(self, model: str, messages: list, *args: Any, **kwargs: Any) -> ModelResponse:
        result = await self._runner_for(model).arun(_query(messages))
        return self._response(model, result)

    def completion(self, model: str, messages: list, *args: Any, **kwargs: Any) -> ModelResponse:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._response(model, asyncio.run(self._runner_for(model).arun(_query(messages))))
        # called from inside an event loop: run on a helper thread so we do not block it
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(lambda: asyncio.run(self._runner_for(model).arun(_query(messages)))).result()
        return self._response(model, result)


def register(runner: Runner, provider: str = "lamas") -> LamasLLM:
    """Make ``<provider>/<policy>`` a model name LiteLLM understands. Returns the handler."""
    handler = LamasLLM(runner)
    existing = [m for m in (litellm.custom_provider_map or []) if m.get("provider") != provider]
    litellm.custom_provider_map = existing + [{"provider": provider, "custom_handler": handler}]
    return handler
