"""Two tools for a shop's order questions, as operators. orders.yaml loads this file as a plugin.

* PriceLookup: retrieval. It finds the items of the order in the shop's catalog and answers with their prices.
* Calculator: code execution. The LLM writes the total as one expression; Python computes it.

Each is one operator: the controller (or a hand-written workflow) can pick it like any other.
"""
import json
import os
import re

import lamas
from lamas import Operator, register_operator
from lamas.sandbox import run_python

CATALOG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.json")))


def lookup(problem):
    """The retrieval step: every catalog item in the order, with its quantity and price."""
    found = []
    for item, price in CATALOG.items():
        match = re.search(rf"(\d+)\s+{item}s?\b", problem)
        if match:
            found.append((int(match.group(1)), item, price))
    return found


WITH_PRICES = """{instruction}{problem}

Prices from the catalog:
{prices}

The bill: {bill}
Add it up, then end with a line `Final answer: <amount>`."""


@register_operator
class PriceLookup(Operator):
    name = "PriceLookup"
    description = "Looks up the price of every item of the order in the shop's catalog, then answers with those prices."
    needs = "problem"              # reads only the question, so it runs at the same time as other generators

    async def run(self, ctx):
        found = lookup(ctx.problem)
        prices = "\n".join(f"- {item}: {price:.2f} dollars each" for _, item, price in found) or "- none found"
        bill = " + ".join(f"{n} x {price:.2f}" for n, _, price in found) or "unknown"
        return await ctx.ask(WITH_PRICES, instruction=ctx.task.instruction(self.name), prices=prices, bill=bill)


WRITE_EXPRESSION = """{problem}

Prices from the catalog:
{prices}

Write the total as one Python arithmetic expression that uses these prices.
Reply with the expression only, in a ```python block."""
_EXPRESSION = re.compile(r"^[\d\s.+\-*/()]+$")


def last_expression(reply):
    """The last line of the reply's code block (or of the reply) that is pure arithmetic."""
    code = re.findall(r"```(?:python)?\s*(.*?)```", reply, re.S)
    lines = [ln.strip() for ln in (code[-1] if code else reply).splitlines() if _EXPRESSION.match(ln.strip())]
    return lines[-1] if lines else None


@lamas.operator(needs="problem")
async def Calculator(ctx):
    """Writes the order total as one arithmetic expression from the catalog prices, then computes it with Python."""
    prices = "\n".join(f"- {item}: {price:.2f} dollars each" for _, item, price in lookup(ctx.problem))
    reply = await ctx.ask(WRITE_EXPRESSION, prices=prices)
    expression = last_expression(reply)
    if not expression:
        return reply                                    # no expression: pass the reply on as it is
    ok, value = await run_python(f"def total():\n    return round({expression}, 2)", call="total")
    return f"{expression} = {value}\nFinal answer: {value}" if ok else reply
