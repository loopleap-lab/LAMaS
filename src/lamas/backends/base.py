"""The backend interface: anything that turns a prompt into a completion."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from ..types import Completion, Usage

Prompt = Union[str, List[Dict[str, str]]]

# USD per one million tokens (prompt, completion). Unknown models cost 0 (e.g. local engines).
PRICES: Dict[str, tuple] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
}


def lookup_price(model: str) -> tuple:
    """Longest-prefix match on the model name, so dated snapshots share their family's price."""
    best = ""
    for name in PRICES:
        if model.startswith(name) and len(name) > len(best):
            best = name
    return PRICES.get(best, (0.0, 0.0))


def to_messages(prompt: Prompt) -> List[Dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return list(prompt)


class Backend:
    """Subclass and implement :meth:`complete`. Everything else has a default."""

    name: str = "backend"
    model: str = ""
    price: tuple = (0.0, 0.0)   # USD per million prompt / completion tokens

    async def complete(self, prompt: Prompt, *, temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> Completion:
        raise NotImplementedError

    def cost(self, usage: Usage) -> float:
        return (usage.prompt_tokens * self.price[0] + usage.completion_tokens * self.price[1]) / 1e6

    async def aclose(self) -> None:  # pragma: no cover - nothing to close by default
        return None

    def describe(self) -> str:
        return f"{self.name}:{self.model}" if self.model else self.name
