"""Operator: one agent step. Context: everything an operator may touch while it runs."""

from __future__ import annotations

import asyncio
import re
from typing import Any, ClassVar, Dict, List, Optional, Union

from ..backends.base import Backend, Prompt
from ..types import NEEDS_PROBLEM, Completion, Usage

_FIELD = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def fill(template: str, **fields: Any) -> str:
    """Substitute ``{name}`` for the names given; every other brace is left alone.

    Prompts are full of LaTeX and code, so ``str.format`` is the wrong tool.
    """
    return _FIELD.sub(lambda m: str(fields[m.group(1)]) if m.group(1) in fields else m.group(0), template)


class Context:
    """Handed to :meth:`Operator.run`. One context per node, so token counts are per node."""

    def __init__(self, *, problem: str, task: Any, backend: Backend, item: Optional[dict] = None,
                 solution: str = "", solutions: Optional[List[str]] = None, usage: Optional[Usage] = None,
                 options: Optional[Dict[str, Any]] = None):
        self.problem = problem
        self.task = task
        self.backend = backend
        self.item = item if item is not None else {"problem": problem}
        self.solution = solution
        self.solutions = list(solutions or [])
        self.usage = usage if usage is not None else Usage()
        self.options = dict(options or {})

    async def llm(self, prompt: Prompt, **kwargs: Any) -> Completion:
        """One LLM call, counted against this node."""
        out = await self.backend.complete(prompt, **kwargs)
        self.usage.add(Usage(out.prompt_tokens, out.completion_tokens, 1))
        return out

    async def ask(self, template: str, **fields: Any) -> str:
        """Fill ``template`` with the context fields plus ``fields``, call the LLM, return its text."""
        values = {"problem": self.problem, "solution": self.solution,
                  "solutions": self.task.format_candidates(self.solutions) if self.solutions else ""}
        values.update(fields)
        return (await self.llm(fill(template, **values))).text

    async def gather(self, *aws: Any) -> list:
        """Run several calls at once. Cancels the rest if one fails."""
        tasks = [asyncio.ensure_future(a) for a in aws]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for t in tasks:
                t.cancel()
            raise

    async def call(self, operator: "Operator", *, solution: Optional[str] = None,
                   solutions: Optional[List[str]] = None) -> Any:
        """Run another operator inside this node (its LLM calls are counted here)."""
        sub = Context(problem=self.problem, task=self.task, backend=self.backend, item=self.item,
                      solution=self.solution if solution is None else solution,
                      solutions=self.solutions if solutions is None else solutions,
                      usage=self.usage, options=self.options)
        return await operator.run(sub)


class Operator:
    """Subclass, set the three class attributes and implement :meth:`run`.

    ``needs`` tells the scheduler where the node goes:

    * ``"problem"``   reads only the problem; runs in the first wave, in parallel with its peers
    * ``"solution"``  reads the latest upstream solution; waits for that one node
    * ``"solutions"`` reads every solution so far; a merge node that waits for all of them

    ``description`` is what the controller sees: it is embedded with the same sentence encoder
    as the query, so a new operator can be selected without retraining.

    :meth:`run` returns the solution text, or a list of texts for operators that produce several.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    needs: ClassVar[str] = NEEDS_PROBLEM

    def __init__(self, **options: Any):
        self.options = options

    async def run(self, ctx: Context) -> Union[str, List[str]]:
        raise NotImplementedError

    @classmethod
    def embedding_text(cls) -> str:
        return f"{cls.name}: {cls.description}"
