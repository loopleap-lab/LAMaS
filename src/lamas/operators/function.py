"""An operator from one async function, for tools and other small steps.

    @lamas.operator(needs="problem")
    async def Calculator(ctx):
        \"\"\"Writes the arithmetic as one Python expression and computes it.\"\"\"
        expression = await ctx.ask(WRITE_EXPRESSION)
        ok, value = await run_python(f"def main():\\n    return {expression}", call="main")
        return f"Final answer: {value}" if ok else ""

The function is ``run``; its name is the operator's name and its docstring the description the
controller reads, unless ``name`` or ``description`` are given.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from ..registry import OPERATORS
from ..types import NEEDS_PROBLEM
from .base import Operator


def operator(fn: Optional[Callable] = None, *, name: Optional[str] = None, description: Optional[str] = None,
             needs: str = NEEDS_PROBLEM, register: bool = True) -> Any:
    """Turn ``async def fn(ctx) -> str | list[str]`` into a registered :class:`Operator` class.

    Use as ``@operator`` or ``@operator(needs="solution", description="...")``.
    """

    def wrap(f: Callable) -> type:
        if not inspect.iscoroutinefunction(f):
            raise TypeError(f"{f.__name__}: an operator function must be `async def`")
        op_name = name or f.__name__
        text = description or inspect.getdoc(f) or ""
        if not text:
            raise ValueError(f"{op_name}: give the operator a description (the docstring or description=...); "
                             "it is what the controller reads")

        async def run(self, ctx):
            return await f(ctx)

        cls = type(op_name, (Operator,), {"name": op_name, "description": text, "needs": needs, "run": run,
                                          "__module__": f.__module__, "__doc__": f.__doc__})
        if register:
            OPERATORS.register(cls)
        return cls

    return wrap(fn) if fn is not None else wrap
