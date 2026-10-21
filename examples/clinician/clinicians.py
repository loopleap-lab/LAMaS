"""The agents: three clinicians read the case, and the attending decides when they disagree.

Each agent is one LAMaS operator: an async function. `needs` says what it reads, so LAMaS knows
what it waits for. clinician.yaml loads this file with `plugins: [clinicians.py]`.
"""
import lamas

ASK = """You are {role}. {focus}

{problem}

Think it through in a few sentences, then end with a line `Final answer: <letter>`."""

DECIDE = """You are the attending physician. Your team read this case:

{problem}

Their answers:
{solutions}

They do not all agree. Weigh their reasoning, then end with a line `Final answer: <letter>`."""


@lamas.operator(needs="problem")                 # reads only the case: starts at once
async def Internist(ctx):
    """An internist reads the whole case: the history, the findings and the tests."""
    return await ctx.ask(ASK, role="an internist",
                         focus="Weigh the history, the findings and the tests.")


@lamas.operator(needs="problem")
async def Pathologist(ctx):
    """A pathologist reasons about the mechanism of disease behind the findings."""
    return await ctx.ask(ASK, role="a pathologist",
                         focus="Reason about the mechanism of disease behind the findings.")


@lamas.operator(needs="problem")
async def Pharmacist(ctx):
    """A clinical pharmacist checks the drugs: their effects, side effects and interactions."""
    return await ctx.ask(ASK, role="a clinical pharmacist",
                         focus="Check the drugs: their effects, side effects and interactions.")


@lamas.operator(needs="solutions")               # reads the team's answers: waits for them
async def Attending(ctx):
    """The attending physician reads the team's answers and decides."""
    return await ctx.ask(DECIDE)
