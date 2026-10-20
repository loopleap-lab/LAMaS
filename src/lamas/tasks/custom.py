r"""A task set up from a config: your data, your prompts, your answer format, no subclass.

    task:
      name: custom
      data:     {train: faq_train.jsonl, test: faq_test.jsonl}    # JSONL, one problem per line
      fields:   {problem: question, answer: label}                 # which keys hold them
      prompts:  {solve_instruction: "You answer questions about our product.\n"}
      instructions: {GenerateCoT: "Think it through, then answer.\n"}
      answer_format: final_line                                    # or boxed, or a regular expression

In a config, relative ``data`` paths are relative to the config file; from Python, to the working
directory. Everything not set behaves as the ``general`` task.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from .. import answers
from ..registry import register_task
from .base import Task, read_jsonl

PROMPTS = ("solve_instruction", "generate_template", "cot_template", "refine_template", "ensemble_template")


def extract(text: str, answer_format: str = "final_line") -> Optional[str]:
    r"""The normalised answer in ``text``.

    * ``"final_line"``  a ``Final answer: ...`` line (or a short ``\boxed{}``), as in the ``general`` task
    * ``"boxed"``       the last ``\boxed{...}``
    * anything else     a regular expression: its first group (or the whole match) in the last match
    """
    if answer_format == "final_line":
        return answers.extract_final_line(text)
    if answer_format == "boxed":
        value = answers.extract_boxed(text)
    else:
        hits = list(re.finditer(answer_format, text or "", re.M))
        value = (hits[-1].group(1) if hits[-1].groups() else hits[-1].group(0)) if hits else None
    return answers._normalise(value) if value else None


@register_task(name="custom")
class CustomTask(Task):
    """Args:
        data: split name -> JSONL file.
        fields: ``{"problem": key, "answer": key}``: the keys of a row that hold the problem and the
            reference answer. Default: ``problem`` (or ``question``) and ``answer``.
        prompts: replaces any of ``solve_instruction``, ``generate_template``, ``cot_template``,
            ``refine_template``, ``ensemble_template`` (``{problem}``, ``{solution}``, ``{solutions}``
            and ``{instruction}`` are filled in).
        instructions: operator name -> the instruction that operator gets instead of ``solve_instruction``.
        answer_format: how the short answer is read, for agreement and for scoring (see :func:`extract`).
        operators: the operator pool to offer when the policy does not name one.
    """

    name = "custom"

    def __init__(self, data: Optional[Dict[str, str]] = None, fields: Optional[Dict[str, str]] = None,
                 prompts: Optional[Dict[str, str]] = None, instructions: Optional[Dict[str, str]] = None,
                 answer_format: str = "final_line", operators: Optional[List[str]] = None, **options: Any):
        super().__init__(**options)
        self.files = {str(k): str(v) for k, v in (data or {}).items()}
        self.fields = dict(fields or {})
        for key, text in (prompts or {}).items():
            if key not in PROMPTS:
                raise ValueError(f"unknown prompt {key!r}. Known: {list(PROMPTS)}")
            setattr(self, key, text)
        self.instructions = dict(instructions or {})
        self.answer_format = answer_format
        if operators:
            self.operators = list(operators)

    def instruction(self, operator: str) -> str:
        return self.instructions.get(operator, self.solve_instruction)

    def load(self, split: str = "test", limit: Optional[int] = None) -> List[dict]:
        if split not in self.files:
            raise ValueError(f"task {self.name!r} has no {split!r} split (data: {self.files or 'none'})")
        path = os.path.abspath(self.files[split])
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        return read_jsonl(path, limit)

    def problem(self, item: dict) -> str:
        if "problem" in self.fields:
            return str(item.get(self.fields["problem"]) or "")
        return super().problem(item)

    def answer_key(self, solution: str, item: Optional[dict] = None) -> Optional[str]:
        return extract(solution, self.answer_format)

    def score(self, item: dict, answer: str) -> float:
        gold = item.get(self.fields.get("answer", "answer"))
        if gold is None:
            raise ValueError("this item has no reference answer")
        key = self.answer_key(answer, item)
        gold_key = (answers.extract_final_line(f"Final answer: {gold}") if self.answer_format == "final_line"
                    else answers._normalise(str(gold)))
        return 1.0 if key is not None and key == gold_key else 0.0
