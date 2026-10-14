"""HumanEval: write a Python function from its signature and docstring.

This task follows the research implementation of the paper step by step, because the trained
controller and stop gate only reproduce their numbers on the workflow they were trained with:

* the prompts are the ones the operators were trained with (generators get the bare problem);
* a reply is reduced to code the same way (:func:`sanitize`);
* one task-specific operator, ``Test``, runs the public test cases and repairs the code;
* after the workflow, the answer is chosen by the public tests, and rewritten if nothing passes;
* two solutions agree when they get the same pass/fail outcomes on the public test cases, the same
  execution feedback the ``Test`` operator uses.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import re
from typing import Dict, List, Optional, Set

from .. import answers
from ..operators.base import Context, Operator
from ..registry import register_operator, register_task
from ..sandbox import run_python
from ..types import NEEDS_SOLUTION
from .base import Task, data_dir

# ---- prompts, verbatim from the research code (operators and gate were trained with these) ---------
COT_TEMPLATE = """
Code Generation with Chain-of-Thought Reasoning

{instruction}

Problem: {problem}

Demonstration Examples (HumanEval style):

1. Problem: Write a function that returns the sum of two numbers.
   Analysis: 
   Function takes two parameters and returns their sum. Direct addition handles all numeric types.
   
   def add(a, b):
       return a + b

2. Problem: Write a function to find the maximum element in a list.
   Analysis:
   Handle empty list edge case. Iterate through list once, tracking the maximum value.
   
   def find_max(lst):
       if not lst:
           raise ValueError("List is empty")
       max_val = lst[0]
       for item in lst[1:]:
           if item > max_val:
               max_val = item
       return max_val

Solution Protocol:
1. Analyze the problem requirements carefully
2. Identify edge cases and constraints
3. Think through the algorithmic approach step by step
4. Consider data structures and methods needed
5. Write clean, efficient, well-documented code

Step-by-Step Analysis:
"""

REFINE_TEMPLATE = """
You are an assistant specialized in refining solutions to problems.

Problem:
{problem}

Solution:
{solution}

Instruction:
Analyze the above solution for any errors or suboptimal aspects. Make iterative improvements to enhance its correctness and efficiency. Provide the refined solution below.
"""

REFLECT_TEMPLATE = """
Given a code problem and a python code solution which failed to pass test or execute, you need to analyze the reason for the failure and propose a better code solution.: 
### problem
{problem}

### Code Solution
{solution}

### Execution Result
{exec_pass}

#### Failed Test Case
{test_fail}

Please provide a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases.
"""

ENSEMBLE_TEMPLATE = """
Given the question described as follows: {problem}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.

### Response format (must be strictly followed): All content must be enclosed in the given XML tags, ensuring each opening <tag> has a corresponding closing </tag>, with no incomplete or self-closing tags allowed.

<thought>content</thought>
<solution_letter>content</solution_letter>
"""

IMPROVE_TEMPLATE = """
The previous solution failed some test cases in the HumanEval benchmark. Please conduct a thorough analysis of the problem statement, identifying all edge cases and potential pitfalls. Then, provide an improved solution that not only fixes the issues but also optimizes performance and adheres to industry-standard coding practices. Ensure your revised code includes clear, concise comments that explain your logic and design choices, and that it robustly handles all specified requirements.
{problem}"""

# What the hidden tests of three problems call besides the function itself.
_HELPERS = {
    "decode_cyclic": (
        'def encode_cyclic(s: str):\n    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)]\n'
        '    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups]\n    return "".join(groups)\n\n'),
    "decode_shift": (
        'def encode_shift(s: str):\n    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])\n\n'),
    "find_zero": "def poly(xs: list, x: float):\n    return sum(coeff * (x ** i) for i, coeff in enumerate(xs))\n\n",
}
_PRELUDE = "import math, hashlib, re\nfrom typing import List, Dict, Tuple, Optional, Any\n\n"

# Entry points without public tests: two problems share the name, or the examples do not run as given.
_NO_PUBLIC_TESTS = frozenset({"find_zero", "decode_cyclic", "decode_shift", "by_length", "add", "triangle_area",
                              "correct_bracketing", "solve", "sum_squares", "starts_one_ends"})
_TEST_TIMEOUT = 15


# ---- from a model reply to code -----------------------------------------------------------------
def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError, ValueError, RecursionError):
        return False


def code_extract(text: str) -> str:
    """The longest run of lines that is valid Python (longest = most non-empty lines)."""
    lines = text.split("\n")
    filled = [1 if line.strip() else 0 for line in lines]
    best, longest = (0, 0), 0
    for i in range(len(lines)):
        if sum(filled[i:]) <= longest:              # nothing from here on can be longer
            break
        for j in range(len(lines) - 1, i, -1):      # the longest valid run that starts at line i
            if _parses("\n".join(lines[i:j + 1])):
                size = sum(filled[i:j + 1])
                if size > longest:
                    longest, best = size, (i, j)
                break
    return "\n".join(lines[best[0]:best[1] + 1])


def _names(node: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr)
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(sub.name)
        elif isinstance(sub, ast.arg):
            out.add(sub.arg)
        elif isinstance(sub, ast.keyword) and sub.arg:
            out.add(sub.arg)
    return out


@functools.lru_cache(maxsize=4096)
def sanitize(code: str, entry_point: Optional[str] = None) -> str:
    """Keep the imports and the first definition of every name; with an entry point, only what it uses.

    A function counts only if it returns something. This is the routine of the research code
    (after evalplus), written with ``ast``.
    """
    code = code_extract(code)
    try:
        tree = ast.parse(code)
    except (SyntaxError, MemoryError, ValueError, RecursionError):
        return ""
    imports: List[ast.AST] = []
    definitions: List[tuple] = []
    seen: Set[Optional[str]] = set()
    for node in tree.body:
        name: Optional[str]
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
                imports.append(node)
            continue
        if isinstance(node, ast.ClassDef) and not node.decorator_list:
            name = node.name
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.decorator_list:
            if not any(isinstance(sub, ast.Return) for sub in ast.walk(node)):
                continue
            name = node.name
        elif isinstance(node, ast.Assign):
            target = node.targets[0]
            name = target.id if isinstance(target, ast.Name) else (node.value.id if isinstance(node.value, ast.Name) else None)
        elif isinstance(node, ast.AnnAssign):
            name = node.target.id if isinstance(node.target, ast.Name) else None
        else:
            continue
        if name in seen:
            continue
        seen.add(name)
        definitions.append((name, node))

    wanted: Optional[Set[str]] = None
    if entry_point:
        uses = {name: _names(node) for name, node in definitions}
        wanted, queue = {entry_point}, [entry_point]
        while queue:
            for dep in uses.get(queue.pop(0), ()):
                if dep not in wanted:
                    wanted.add(dep)
                    queue.append(dep)
    parts = [ast.get_source_segment(code, node) or "" for node in imports]
    parts += [ast.get_source_segment(code, node) or "" for name, node in definitions if wanted is None or name in wanted]
    return "\n".join(parts)


# ---- public tests -------------------------------------------------------------------------------
_public: Optional[Dict[str, List[str]]] = None


def public_tests(entry_point: str) -> List[str]:
    """The example-based test cases that may be used while solving (not the hidden ones)."""
    global _public
    if entry_point in _NO_PUBLIC_TESTS:
        return []
    if _public is None:
        _public = {}
        path = os.path.join(data_dir(), "humaneval_public_test.jsonl")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        tests = row.get("test") or []
                        # two problems can share an entry point: the first one in the file counts
                        _public.setdefault(row["entry_point"], [tests] if isinstance(tests, str) else list(tests))
    return _public.get(entry_point, [])


async def run_public_tests(code: str, entry_point: str) -> Dict[str, object]:
    """Run the public tests one by one.

    Returns ``{"passed": bool, "fail_cases": [...], "error": str | None}``: ``fail_cases`` lists the
    failed assertions; ``error`` is set when the code could not be run at all (it stops at the first one).
    """
    tests = public_tests(entry_point)
    if not tests:
        return {"passed": True, "fail_cases": [], "error": None}
    harness = _PRELUDE + _HELPERS.get(entry_point, "") + f"""
def __lamas_public_tests__():
    import json, signal, sys, traceback

    class _Timeout(Exception):
        pass

    def _alarm(signum, frame):
        raise _Timeout()

    signal.signal(signal.SIGALRM, _alarm)
    solution, entry, base = {code!r}, {entry_point!r}, dict(globals())
    fail_cases = []
    for test_case in {tests!r}:
        program = "\\n" + solution + "\\n\\n\\ndef check(candidate):\\n    " + test_case + "\\n\\ndef test_check():\\n    check(" + entry + ")\\n\\ntest_check()\\n"
        signal.alarm({_TEST_TIMEOUT})
        try:
            exec(program, dict(base))
        except _Timeout:
            return json.dumps({{"error": "Execution timed out after {_TEST_TIMEOUT} seconds"}})
        except AssertionError as exc:
            fail_cases.append({{"test_fail_case": {{"test_case": test_case, "error_type": "AssertionError", "error_message": str(exc),
                                                  "traceback": traceback.format_exception(*sys.exc_info())}}}})
        except Exception as exc:
            return json.dumps({{"error": str(exc)}})
        finally:
            signal.alarm(0)
    return json.dumps({{"fail_cases": fail_cases}})
"""
    ok, text = await run_python(harness, call="__lamas_public_tests__", timeout=_TEST_TIMEOUT * len(tests) + 10)
    if not ok:
        return {"passed": False, "fail_cases": [], "error": text}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"passed": False, "fail_cases": [], "error": text}
    if "error" in data:
        return {"passed": False, "fail_cases": [], "error": data["error"]}
    return {"passed": not data["fail_cases"], "fail_cases": data["fail_cases"], "error": None}


@register_operator
class Test(Operator):
    name = "Test"
    description = ("Tests the solution using public test cases. If the solution fails, it reflects on the errors "
                   "and attempts to modify the solution. Returns True and the solution if all tests pass after "
                   "modifications. Returns False and the current solution if it still fails after modifications.")
    needs = NEEDS_SOLUTION
    loops = 3

    async def check(self, ctx: Context, solution: str) -> tuple:
        """Test, and repair up to ``loops`` times. Returns ``(passed, solution)``."""
        entry = ctx.item.get("entry_point", "")
        for _ in range(int(self.options.get("loops", self.loops))):
            result = await run_public_tests(solution, entry)
            if result["passed"]:
                return True, solution
            if result["error"] is not None:
                reply = await ctx.ask(REFLECT_TEMPLATE, solution=solution, test_fail="executed unsucessfully",
                                      exec_pass=f"executed unsuccessfully, error: \n {result['error']}")
            else:
                reply = await ctx.ask(REFLECT_TEMPLATE, solution=solution, exec_pass="executed successfully",
                                      test_fail=result["fail_cases"])
            solution = sanitize(reply, None)
        result = await run_public_tests(solution, entry)
        return bool(result["passed"]), solution

    async def run(self, ctx: Context) -> str:
        _, solution = await self.check(ctx, ctx.solution)
        return solution


@register_task(name="humaneval")
class HumanEval(Task):
    name = "humaneval"
    files = {"train": "humaneval_train.jsonl", "test": "humaneval_test.jsonl"}
    operators = ["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Test", "SelfRefine", "EarlyStop"]
    vote_before_ensemble = False

    solve_instruction = ""                      # generators get the bare problem, as in training
    generate_template = "{instruction}{problem}"
    cot_template = COT_TEMPLATE
    refine_template = REFINE_TEMPLATE
    ensemble_template = ENSEMBLE_TEMPLATE

    def problem(self, item: dict) -> str:
        return item.get("prompt") or item.get("problem") or ""

    def clean(self, text: str, item: Optional[dict] = None) -> str:
        entry = (item or {}).get("entry_point")
        code = sanitize(text or "", entry)
        if entry and not re.search(rf"\b(def|class)\s+{re.escape(entry)}\b", code):
            code = sanitize(text or "", None) or code     # a reply that renamed the function: keep its code
        return code

    def answer_key(self, solution: str, item: Optional[dict] = None) -> Optional[str]:
        return None             # comparing programs means running them: see akey

    async def akey(self, solution: str, item: Optional[dict] = None) -> Optional[str]:
        """Two programs agree when they get the same pass/fail outcomes on the public tests.

        A problem without public tests has no execution feedback; its programs agree only when
        their text is identical.
        """
        entry = (item or {}).get("entry_point")
        if not entry or not public_tests(entry):
            return None
        result = await run_public_tests(solution, entry)
        if result["error"]:
            return "tests:error"
        failed = sorted(case["test_fail_case"]["test_case"] for case in result["fail_cases"])
        return "tests:" + (answers.text_hash("\n".join(failed)) if failed else "pass")

    def parse_choice(self, reply: str, n: int) -> Optional[int]:
        match = re.search(r"<solution_letter>(.*?)</solution_letter>", reply or "", re.DOTALL)
        if match:
            letter = match.group(1).strip().upper()
            if len(letter) == 1 and 0 <= ord(letter) - 65 < n:
                return ord(letter) - 65
        return super().parse_choice(reply, n)

    async def finalize(self, ctx, solutions: List[str], current: str) -> str:
        """Choose the answer with the public tests; rewrite the function if nothing passes.

        With several candidates the first one that passes (after repair) wins. With one, it is
        tested and repaired. Either way a failure ends in one more attempt from scratch.
        """
        tester = Test()
        for candidate in (solutions if len(solutions) > 1 else [current]):
            passed, repaired = await tester.check(ctx, candidate)
            if passed:
                return repaired
        return self.clean(await ctx.ask(IMPROVE_TEMPLATE), ctx.item)

    def score(self, item: dict, answer: str) -> float:
        import asyncio

        return asyncio.run(self.ascore(item, answer))

    async def ascore(self, item: dict, answer: str) -> float:
        entry = item["entry_point"]
        program = (_PRELUDE + _HELPERS.get(entry, "") + sanitize(answer, entry) + "\n\n" + item["test"]
                   + f"\n\ndef __lamas_score__():\n    check({entry})\n    return 'passed'\n")
        ok, out = await run_python(program, call="__lamas_score__", timeout=float(_TEST_TIMEOUT))
        return 1.0 if ok and out == "passed" else 0.0
