"""A simulated LLM for tests and offline demos.

It does not understand language. It recognises which kind of prompt it was given (solve, refine,
ensemble), sleeps for a realistic amount of time and returns a plausible reply. The "right"
answer comes from ``answers`` (substring of the question -> answer; an answer that is a piece
of code comes back as a code block) or, failing that, from the first arithmetic expression
found in the prompt.
"""

from __future__ import annotations

import ast
import asyncio
import operator as op
import random
import re
import time
from collections import Counter
from typing import Callable, Dict, Optional

from ..registry import register_backend
from ..types import Completion
from .base import Backend, Prompt, to_messages

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
        ast.Mod: op.mod, ast.FloorDiv: op.floordiv}
_EXPR = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:\s*[-+*/x×]\s*\d+(?:\.\d+)?)+)")
_FINAL_NUMBER = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_LETTERED = re.compile(r"^([A-Z]):\s*$", re.M)


def _safe_eval(expr: str) -> Optional[float]:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError("unsupported expression")

    try:
        return ev(ast.parse(expr.replace("×", "*").replace("x", "*"), mode="eval"))
    except Exception:
        return None


def _fmt(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


@register_backend(name="mock")
class MockBackend(Backend):
    """Simulated backend.

    Args:
        answers: maps a substring of the question to its correct answer.
        answer_fn: alternatively, a function from the prompt to the correct answer.
        error_rate: probability that a "solve" reply carries a wrong answer.
        tokens_per_second, ttft: the latency model: ``ttft + completion_tokens / tokens_per_second``.
        time_scale: multiply every sleep by this (0 = no sleeping, for unit tests).
        seed: makes replies and latencies reproducible.
    """

    name = "mock"
    model = "mock-llm"
    price = (0.15, 0.60)

    def __init__(self, answers: Optional[Dict[str, str]] = None,
                 answer_fn: Optional[Callable[[str], Optional[str]]] = None,
                 error_rate: float = 0.15, tokens_per_second: float = 70.0, ttft: float = 0.45,
                 time_scale: float = 1.0, seed: Optional[int] = 0, model: Optional[str] = None):
        self.answers = dict(answers or {})
        self.answer_fn = answer_fn
        self.error_rate = error_rate
        self.tokens_per_second = tokens_per_second
        self.ttft = ttft
        self.time_scale = time_scale
        self._rng = random.Random(seed)
        self.calls = 0
        if model:
            self.model = model

    # ---- what would the right answer be? -------------------------------------------------
    def _truth(self, prompt: str) -> str:
        if self.answer_fn is not None:
            got = self.answer_fn(prompt)
            if got is not None:
                return str(got)
        for needle, answer in self.answers.items():
            if needle in prompt:
                return str(answer)
        m = _EXPR.search(prompt)
        if m:
            value = _safe_eval(m.group(1))
            if value is not None:
                return _fmt(value)
        return "42"

    def _wrong(self, truth: str) -> str:
        try:
            value = float(truth.replace(",", ""))
            delta = self._rng.choice([-10, -2, -1, 1, 2, 10])
            return _fmt(value + delta)
        except ValueError:
            return truth + "?"

    # ---- prompt kinds ----------------------------------------------------------------------
    def _reply(self, prompt: str) -> tuple:
        """Returns (text, completion_tokens)."""
        low = prompt.lower()
        rng = self._rng
        if "most consistent" in low or "appears most frequently" in low or "select the best solution" in low:
            letters = _LETTERED.findall(prompt)
            chunks = re.split(r"^[A-Z]:\s*$", prompt, flags=re.M)[1:]
            finals = []
            for chunk in chunks:
                nums = _FINAL_NUMBER.findall(chunk)
                finals.append(nums[-1].replace(",", "") if nums else chunk.strip()[-20:])
            pick = "A"
            if letters and finals:
                best = Counter(finals).most_common(1)[0][0]
                pick = letters[finals.index(best)]
            return (f"The candidates mostly agree with option {pick}.\nAnswer: {pick}", rng.randint(35, 70))
        truth = self._truth(prompt)
        if "\n" in truth or truth.lstrip().startswith("def "):
            # the configured "answer" is a piece of code: return it as a code block
            return (f"```python\n{truth}\n```", rng.randint(110, 200))
        answer = self._wrong(truth) if rng.random() < self.error_rate else truth
        if "refin" in low or "review the above" in low:
            return (f"I re-checked each step of the solution.\nThe final answer is \\boxed{{{truth}}}\nThe answer is {truth}",
                    rng.randint(140, 240))
        steps = rng.randint(3, 5)
        body = "\n".join(f"Step {i + 1}: work through part {i + 1} of the problem." for i in range(steps))
        return (f"{body}\nThe final answer is \\boxed{{{answer}}}\nThe answer is {answer}", rng.randint(150, 330))

    async def complete(self, prompt: Prompt, *, temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> Completion:
        text_in = "\n".join(m.get("content") or "" for m in to_messages(prompt))
        self.calls += 1
        text, n_out = self._reply(text_in)
        jitter = 1.0 + (self._rng.random() - 0.5) * 0.5
        latency = (self.ttft + n_out / self.tokens_per_second) * jitter
        started = time.perf_counter()
        if self.time_scale > 0:
            await asyncio.sleep(latency * self.time_scale)
        else:
            await asyncio.sleep(0)
        elapsed = time.perf_counter() - started
        return Completion(text=text, prompt_tokens=max(1, len(text_in) // 4), completion_tokens=n_out,
                          latency=elapsed if self.time_scale > 0 else latency)
