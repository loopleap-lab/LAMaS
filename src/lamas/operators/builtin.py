"""The operator library from the paper: generators, self-consistency, refinement.

Operators return raw model text; the executor passes it through ``task.clean`` afterwards.
"""

from __future__ import annotations

from typing import List

from .. import answers
from ..registry import register_operator
from ..types import NEEDS_NONE, NEEDS_PROBLEM, NEEDS_SOLUTION, NEEDS_SOLUTIONS
from .base import Context, Operator


@register_operator
class Generate(Operator):
    name = "Generate"
    description = "Generates anything based on customized input and instruction."
    needs = NEEDS_PROBLEM

    async def run(self, ctx: Context) -> str:
        return await ctx.ask(ctx.task.generate_template, instruction=ctx.task.instruction(self.name))


@register_operator
class GenerateCoT(Operator):
    name = "GenerateCoT"
    description = ("Generates anything using a chain-of-thought approach, providing step-by-step reasoning "
                   "before producing the final solution. If you need to solve mathematical or difficult "
                   "reasoning problems, you can use this operator.")
    needs = NEEDS_PROBLEM

    async def run(self, ctx: Context) -> str:
        return await ctx.ask(ctx.task.cot_template, instruction=ctx.task.instruction(self.name))


@register_operator
class MultiGenerateCoT(Operator):
    name = "MultiGenerateCoT"
    description = ("Generates multiple solutions using diverse chain-of-thought reasoning processes to "
                   "increase solution variety and robustness.")
    needs = NEEDS_PROBLEM
    samples = 3

    async def run(self, ctx: Context) -> List[str]:
        n = int(self.options.get("samples", self.samples))
        instruction = ctx.task.instruction(self.name)
        return await ctx.gather(*[ctx.ask(ctx.task.cot_template, instruction=instruction) for _ in range(n)])


@register_operator
class ScEnsemble(Operator):
    name = "ScEnsemble"
    description = ("Uses self-consistency to select the solution that appears most frequently in the solution "
                   "list, improve the selection to enhance the choice of the best solution.")
    needs = NEEDS_SOLUTIONS

    async def run(self, ctx: Context) -> str:
        solutions = [s for s in ctx.solutions if s]
        if not solutions:
            return ctx.solution
        if len(solutions) == 1:
            return solutions[0]
        reply = await ctx.ask(ctx.task.ensemble_template, solutions=ctx.task.format_candidates(solutions))
        idx = ctx.task.parse_choice(reply, len(solutions))
        if idx is None:
            # unreadable verdict: fall back to a plain majority vote over the extracted answers
            return answers.majority(solutions, lambda s: ctx.task.answer_key(s, ctx.item)) or solutions[0]
        return solutions[idx]


@register_operator
class SelfRefine(Operator):
    name = "SelfRefine"
    description = ("Refines the generated solution by analyzing errors or suboptimal aspects and making "
                   "iterative improvements to enhance correctness and efficiency.")
    needs = NEEDS_SOLUTION

    async def run(self, ctx: Context) -> str:
        return await ctx.ask(ctx.task.refine_template)


@register_operator
class EarlyStop(Operator):
    """A control token, never executed: when the controller picks it, no further layer is added."""

    name = "EarlyStop"
    description = ("Immediately terminates the workflow when selected, preventing any further operators "
                   "from being invoked.")
    needs = NEEDS_NONE

    async def run(self, ctx: Context) -> str:  # pragma: no cover - the scheduler skips it
        return ctx.solution
