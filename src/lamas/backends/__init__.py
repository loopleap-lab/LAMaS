"""Backends: where LLM calls go."""

from __future__ import annotations

from typing import Any, Optional, Union

from ..registry import BACKENDS
from .base import PRICES, Backend, Prompt, lookup_price
from .mock import MockBackend
from .openai_compat import OpenAICompatBackend

__all__ = ["Backend", "Prompt", "PRICES", "lookup_price", "MockBackend", "OpenAICompatBackend", "make_backend"]


def make_backend(spec: Union[None, str, dict, Backend] = None, **overrides: Any) -> Backend:
    """Build a backend from a config value.

    * ``None``            -> OpenAI-compatible, from ``$LAMAS_BASE_URL`` / ``$LAMAS_MODEL`` (or OpenAI defaults)
    * ``"mock"``          -> the simulated backend
    * ``"http://..."``    -> OpenAI-compatible at that base URL
    * ``{"type": ..., ...}`` or ``{"base_url": ..., "model": ...}``
    """
    import os

    if isinstance(spec, Backend):
        return spec
    if spec is None:
        spec = {}
    if isinstance(spec, str):
        spec = {"type": spec} if spec in BACKENDS else {"base_url": spec}
    cfg = dict(spec)
    cfg.update(overrides)
    kind = cfg.pop("type", None) or ("openai" if ("base_url" in cfg or "model" in cfg or not cfg) else "openai")
    if kind == "openai":
        cfg.setdefault("base_url", os.environ.get("LAMAS_BASE_URL", "https://api.openai.com/v1"))
        cfg.setdefault("model", os.environ.get("LAMAS_MODEL", "gpt-4o-mini"))
    return BACKENDS.get(kind)(**cfg)
