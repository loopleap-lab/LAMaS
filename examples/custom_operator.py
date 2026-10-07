"""Add an operator in one file, then offer it to the trained controller.

    python examples/custom_operator.py

The controller reads the operator's description, so it can pick an operator it was never trained with.
"""
from collections import Counter

import lamas
from lamas import Operator, register_operator

PRO = "Argue FOR the solution below, and fix any small slip you notice.\n\nProblem:\n{problem}\n\nSolution:\n{solution}"
CON = "Argue AGAINST the solution below: find the error that matters most and repair it.\n\nProblem:\n{problem}\n\nSolution:\n{solution}"
JUDGE = "Problem:\n{problem}\n\nAdvocate:\n{pro}\n\nCritic:\n{con}\n\nDecide who is right and write the complete final solution."


@register_operator
class Debate(Operator):
    name = "Debate"
    description = ("Two agents argue for and against the current "
                   "solution; a judge decides.")
    needs = "solution"            # problem | solution | solutions

    async def run(self, ctx):
        pro, con = await ctx.gather(ctx.ask(PRO), ctx.ask(CON))
        return await ctx.ask(JUDGE, pro=pro, con=con)


PROBLEM = '''def median(numbers: list) -> float:
    """ Return the median of the given numbers. """
'''
policy = lamas.load_policy()                                    # lamas/humaneval
pool = policy.operators + ["Debate"]
picks = Counter(name for seed in range(200) for layer in lamas.plan(PROBLEM, operators=pool, seed=seed).layers for name in layer)
total = sum(picks.values())
for name in pool:
    print(f"{name:18s} picked in {picks[name] / total:5.1%} of all picks")
