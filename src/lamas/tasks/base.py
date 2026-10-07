"""A Task tells LAMaS how to pose a problem, how to read an answer and how to score it."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .. import answers
from ..registry import register_task

DEFAULT_OPERATORS = ["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "SelfRefine", "EarlyStop"]


def data_dir() -> str:
    """Where benchmark files live: ``$LAMAS_DATA`` or ``<repository>/data``."""
    env = os.environ.get("LAMAS_DATA")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "data"))


def read_jsonl(path: str, limit: Optional[int] = None) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


@register_task(name="general")
class Task:
    """The interface, with defaults for a plain question that has a short final answer.

    Subclass it for a kind of problem: the prompts the operators use, how a reply becomes a
    solution, when two solutions agree, how an answer is scored, and how the final answer is chosen.
    """

    name = "general"
    operators: List[str] = DEFAULT_OPERATORS
    files: Dict[str, str] = {}      # split name -> file name inside data_dir()

    # ---- prompt pieces (operators read these) --------------------------------------------
    solve_instruction = (
        "Answer the question below. Be concise.\n"
        "End your reply with a line of the form `Final answer: <answer>`.\n\nQuestion:\n"
    )
    generate_template = "{instruction}{problem}"
    cot_template = (
        "{instruction}{problem}\n\n"
        "Work through the problem step by step, checking each step, before you state the final answer."
    )
    refine_template = (
        "You are reviewing a proposed answer.\n\nQuestion:\n{problem}\n\nProposed answer:\n{solution}\n\n"
        "Look for mistakes or weak points, fix them, and give the complete improved answer.\n"
        "End your reply with a line of the form `Final answer: <answer>`."
    )
    ensemble_template = (
        "Question:\n{problem}\n\nSeveral candidate answers were written independently:\n{solutions}\n"
        "Pick the candidate whose final answer agrees with most of the others. If they all differ, "
        "pick the best-reasoned one.\nExplain briefly, then end with a line of the form `Answer: <letter>`."
    )

    def __init__(self, **options: Any):
        self.options = options

    # ---- prompts -------------------------------------------------------------------------
    def instruction(self, operator: str) -> str:
        """The task instruction a generating operator should use."""
        return self.solve_instruction

    def format_candidates(self, solutions: List[str]) -> str:
        return "".join(f"{chr(65 + i)}: \n{s}\n\n\n" for i, s in enumerate(solutions))

    def parse_choice(self, reply: str, n: int) -> Optional[int]:
        """Index of the candidate the ensemble picked, or None if the reply is unreadable."""
        import re

        hits = re.findall(r"[Aa]nswer\s*[:：]\s*\(?\[?([A-Za-z])\]?\)?", reply or "")
        if not hits:
            hits = re.findall(r"\b([A-Z])\b\s*\.?\s*$", (reply or "").strip())
        if hits:
            idx = ord(hits[-1].upper()) - 65
            if 0 <= idx < n:
                return idx
        return None

    # ---- data ----------------------------------------------------------------------------
    def load(self, split: str = "test", limit: Optional[int] = None) -> List[dict]:
        if split not in self.files:
            raise ValueError(f"task {self.name!r} has no {split!r} split (available: {list(self.files) or 'none'})")
        path = os.path.join(data_dir(), self.files[split])
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Set $LAMAS_DATA to the folder holding the benchmark files.")
        return read_jsonl(path, limit)

    def problem(self, item: dict) -> str:
        return item.get("problem") or item.get("question") or ""

    # ---- answers -------------------------------------------------------------------------
    def clean(self, text: str, item: Optional[dict] = None) -> str:
        """Turn a raw model reply into the solution that is passed on."""
        return (text or "").strip()

    def answer_key(self, solution: str, item: Optional[dict] = None) -> Optional[str]:
        """A normalised final answer, used to decide whether two solutions agree.

        Return None when no short answer can be read; the solution then only agrees with
        byte-identical ones.
        """
        return answers.extract_final_line(solution)

    async def akey(self, solution: str, item: Optional[dict] = None) -> Optional[str]:
        """Async form of :meth:`answer_key`, for tasks that must run something to compare answers."""
        return self.answer_key(solution, item)

    def final_text(self, solution: str, item: Optional[dict] = None) -> str:
        """What the user sees as the answer."""
        return solution

    def score(self, item: dict, answer: str) -> float:
        gold = item.get("answer")
        if gold is None:
            raise ValueError("this item has no reference answer")
        key = self.answer_key(answer, item)
        gold_key = answers.extract_final_line(f"Final answer: {gold}")
        return 1.0 if key is not None and key == gold_key else 0.0

    async def ascore(self, item: dict, answer: str) -> float:
        """Async form of :meth:`score`, for tasks that must run something to grade."""
        return self.score(item, answer)

    # ---- the final step ------------------------------------------------------------------
    async def finalize(self, ctx, solutions: List[str], current: str) -> str:
        """Reduce the surviving solutions to one answer. Runs after the workflow (or after an early stop)."""
        if not solutions:
            return current
        if len(solutions) == 1:
            return solutions[0]
        keys = [await self.akey(s, ctx.item) for s in solutions]
        if all(k is not None for k in keys) and len(set(keys)) == 1:
            return solutions[-1]            # they already agree: no LLM call needed
        from ..operators.builtin import ScEnsemble

        return await ctx.call(ScEnsemble(), solutions=solutions)
