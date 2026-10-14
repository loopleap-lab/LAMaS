"""Client for any OpenAI-compatible chat endpoint: vLLM, SGLang, Ollama, OpenAI, OpenRouter, ..."""

from __future__ import annotations

import asyncio
import os
import random
import re
import time
from typing import Optional

import httpx

from ..registry import register_backend
from ..types import Completion
from .base import Backend, Prompt, lookup_price, to_messages

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
_KEY_LIKE = re.compile(r"\b(?:sk|hf|gsk|xai)[-_][\w\-\*\.]{8,}")      # providers echo part of a bad key in 401 bodies


@register_backend(name="openai")
class OpenAICompatBackend(Backend):
    """Talks ``POST {base_url}/chat/completions``.

    Args:
        base_url: e.g. ``http://localhost:8000/v1`` (vLLM, SGLang), ``http://localhost:11434/v1``
            (Ollama) or ``https://api.openai.com/v1``.
        model: the model name the server expects.
        api_key: defaults to ``$LAMAS_API_KEY`` then ``$OPENAI_API_KEY``; local servers need none.
        price: USD per million (prompt, completion) tokens; looked up for known models.
        max_concurrency: cap on simultaneous requests from this process.
        temperature, max_tokens, system: request settings. The defaults (temperature 1, 4096 tokens,
            the system message "You are a helpful assistant.") are the ones the released policy
            was trained and checked with.
    """

    name = "openai"

    def __init__(self, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None, temperature: Optional[float] = 1.0,
                 max_tokens: Optional[int] = 4096, price: Optional[tuple] = None,
                 max_concurrency: int = 64, timeout: float = 300.0, max_retries: int = 6,
                 extra_body: Optional[dict] = None, system: Optional[str] = "You are a helpful assistant."):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.price = tuple(price) if price else lookup_price(model)
        self.max_retries = max_retries
        self.extra_body = dict(extra_body or {})
        self.system = system
        key = api_key or os.environ.get("LAMAS_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self._key = key
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=httpx.Timeout(timeout, connect=30.0),
            limits=httpx.Limits(max_connections=max_concurrency, max_keepalive_connections=max_concurrency),
        )
        self._sem = asyncio.Semaphore(max_concurrency)

    async def complete(self, prompt: Prompt, *, temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> Completion:
        messages = to_messages(prompt)
        if self.system and isinstance(prompt, str):
            messages = [{"role": "system", "content": self.system}] + messages
        body = {"model": self.model, "messages": messages}
        temp = self.temperature if temperature is None else temperature
        if temp is not None:
            body["temperature"] = temp
        limit = max_tokens or self.max_tokens
        if limit:
            body["max_tokens"] = limit
        body.update(self.extra_body)

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                async with self._sem:
                    resp = await self._client.post("/chat/completions", json=body)
                if resp.status_code in _RETRY_STATUS:
                    raise httpx.HTTPStatusError(f"retryable status {resp.status_code}", request=resp.request, response=resp)
                resp.raise_for_status()
                data = resp.json()
                text = (data["choices"][0]["message"].get("content") or "").strip()
                usage = data.get("usage") or {}
                return Completion(
                    text=text,
                    prompt_tokens=int(usage.get("prompt_tokens") or _estimate_tokens(body["messages"])),
                    completion_tokens=int(usage.get("completion_tokens") or max(1, len(text) // 4)),
                    latency=time.perf_counter() - started,
                )
            except asyncio.CancelledError:
                raise
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError, KeyError, ValueError) as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status not in _RETRY_STATUS:
                    detail = self._scrub(getattr(exc.response, "text", "")[:300])
                    raise RuntimeError(f"backend returned {status}: {detail}") from exc
                await asyncio.sleep(min(20.0, 0.5 * 2 ** attempt) + random.random() * 0.25)
        raise RuntimeError(f"backend failed after {self.max_retries} attempts: {last_error!r}")

    def _scrub(self, text: str) -> str:
        """Error text ends up in traces and reports: keep credentials out of it."""
        if self._key:
            text = text.replace(self._key, "***")
        return _KEY_LIKE.sub("***", text)

    async def aclose(self) -> None:
        await self._client.aclose()


def _estimate_tokens(messages) -> int:
    return max(1, sum(len(m.get("content") or "") for m in messages) // 4)
