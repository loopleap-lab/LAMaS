import pytest

from lamas.sandbox import run_python
from lamas.tasks import HumanEval
from lamas.tasks.humaneval import public_tests, run_public_tests, sanitize

GOOD = "def has_close_elements(numbers, threshold):\n    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i + 1:])\n"
BAD = "def has_close_elements(numbers, threshold):\n    return False\n"


async def test_sandbox_runs_and_times_out():
    assert await run_python("def solve():\n    return 6 * 7\n", call="solve") == (True, "42")
    ok, msg = await run_python("def solve():\n    return 1 / 0\n", call="solve")
    assert not ok and "ZeroDivisionError" in msg
    ok, msg = await run_python("def solve():\n    while True: pass\n", call="solve", timeout=1.0)
    assert not ok and "timed out" in msg
    ok, msg = await run_python("import subprocess\ndef solve():\n    return 1\n", call="solve")
    assert not ok and "Prohibited" in msg


async def test_public_tests_run_and_report():
    if not public_tests("has_close_elements"):
        pytest.skip("benchmark data is not available")
    good, bad = await run_public_tests(GOOD, "has_close_elements"), await run_public_tests(BAD, "has_close_elements")
    assert good["passed"] and not bad["passed"] and bad["fail_cases"] and bad["error"] is None
    broken = await run_public_tests("def has_close_elements(numbers, threshold):\n    return undefined_name\n", "has_close_elements")
    assert not broken["passed"] and "undefined_name" in broken["error"]
    # two programs agree when they get the same pass/fail outcomes on the public tests
    item, task = {"entry_point": "has_close_elements"}, HumanEval()
    assert await task.akey(GOOD, item) == "tests:pass"
    assert await task.akey(BAD, item) not in (None, "tests:pass")
    assert await task.akey("def has_close_elements(numbers, threshold):\n    return undefined_name\n", item) == "tests:error"
    # no public tests, no execution feedback: only identical text agrees
    assert await task.akey("def triangle_area(a, h):\n    return a * h / 2\n", {"entry_point": "triangle_area"}) is None


async def test_names_shared_by_two_problems_have_no_public_tests():
    # HumanEval has two problems called triangle_area and two called solve; their examples must not be mixed up
    assert public_tests("triangle_area") == [] and public_tests("solve") == []
    assert (await run_public_tests("def triangle_area(a, h):\n    return a * h / 2\n", "triangle_area"))["passed"]


async def test_an_entry_point_called_solve_is_graded():
    item = {"entry_point": "solve", "test": "def check(candidate):\n    assert candidate(3) == 9\n"}
    assert await HumanEval().ascore(item, "def solve(n):\n    return n * n\n") == 1.0
    assert await HumanEval().ascore(item, "def solve(n):\n    return n\n") == 0.0


def test_sanitize_keeps_the_first_definition_and_what_it_uses():
    reply = ("Here is the plan.\n```python\nimport math\n\ndef helper(x):\n    return x + 1\n\ndef unused(x):\n    return x\n\n"
             "def f(x):\n    return helper(x) * 2\n\ndef f(x):\n    return 0\n\ndef show(x):\n    print(x)\n\nprint(f(1))\n```\nDone.")
    code = sanitize(reply, "f")
    assert code == "import math\ndef helper(x):\n    return x + 1\ndef f(x):\n    return helper(x) * 2"
    assert "def unused" in sanitize(reply, None) and "def show" not in sanitize(reply, None)


async def test_hidden_tests_grade_the_answer():
    task = HumanEval()
    try:
        item = next(i for i in task.load("train") + task.load("test") if i["entry_point"] == "has_close_elements")
    except (FileNotFoundError, StopIteration):
        pytest.skip("benchmark data is not available")
    assert await task.ascore(item, GOOD) == 1.0
    assert await task.ascore(item, BAD) == 0.0
