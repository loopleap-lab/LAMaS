"""Extending LAMaS without a subclass or training: a task from a config, operators from functions,
plugins from files, a hand-written workflow and a rule-based stop policy (docs/extending.md)."""
import json
import os

import pytest

import lamas
from lamas.backends import MockBackend
from lamas.config import load_config
from lamas.evaluation import aevaluate
from lamas.registry import CONTROLLERS, OPERATORS, STOP_POLICIES
from lamas.stop import StopState

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(HERE, "..", "examples", "extend", "orders.yaml")


def test_a_task_from_options(tmp_path):
    rows = [{"q": "What is 6 * 7?", "label": "42"}, {"q": "What is 2 + 2?", "label": "4"}]
    path = tmp_path / "quiz.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    task = lamas.get_task({"name": "custom", "data": {"test": str(path)}, "fields": {"problem": "q", "answer": "label"},
                           "prompts": {"solve_instruction": "Quiz.\n"}, "instructions": {"GenerateCoT": "Think.\n"}})
    items = task.load("test")
    assert [task.problem(i) for i in items] == ["What is 6 * 7?", "What is 2 + 2?"]
    assert task.instruction("Generate") == "Quiz.\n" and task.instruction("GenerateCoT") == "Think.\n"
    assert task.score(items[0], "6 * 7 = 42.\nFinal answer: 42") == 1.0 and task.score(items[1], "Final answer: 5") == 0.0
    with pytest.raises(ValueError):
        lamas.get_task({"name": "custom", "prompts": {"no_such_prompt": "x"}})


@pytest.mark.parametrize("fmt,text,key", [("boxed", r"so \boxed{12}", "12"), (r"ANSWER=(\w+)", "ANSWER=b ... ANSWER=C", "c"),
                                          ("final_line", "Final answer: 1,000", "1000")])
def test_answer_formats(fmt, text, key):
    task = lamas.get_task({"name": "custom", "answer_format": fmt})
    assert task.answer_key(text) == key


def test_the_general_task_is_unchanged():
    general = lamas.get_task("general")
    assert general.answer_key("Final answer: 1,000") == "1000" and general.instruction("GenerateCoT") == general.solve_instruction


async def test_an_operator_from_a_function():
    @lamas.operator(needs="solution")
    async def Shout(ctx):
        """Repeats the solution in capitals."""
        return ctx.solution.upper()

    assert OPERATORS.get("Shout") is Shout and Shout.needs == "solution" and Shout.description == "Repeats the solution in capitals."
    with pytest.raises(TypeError):
        lamas.operator(lambda ctx: "not async")
    with pytest.raises(ValueError):
        @lamas.operator
        async def Silent(ctx):
            return ""
    policy = lamas.load_policy({"controller": {"type": "fixed", "layers": [["Generate"], ["Shout"]]}})
    result = await lamas.Runner(policy=policy, backend=MockBackend(time_scale=0, error_rate=0.0)).arun("What is 6 * 7?")
    assert result.trace.spans["Shout#0"].deps == ["Generate#0"] and "THE ANSWER IS 42" in result.answer


def test_the_fixed_controller():
    assert "fixed" in CONTROLLERS and "agreement" in STOP_POLICIES
    plan = lamas.Runner(policy={"controller": {"type": "fixed", "layers": [["GenerateCoT", "Generate"], ["ScEnsemble"]]}},
                        backend="mock").plan("What is 6 * 7?")
    assert plan.layers == [["GenerateCoT", "Generate"], ["ScEnsemble"]]
    assert plan.nodes["ScEnsemble#0"].deps == ["GenerateCoT#0", "Generate#0"]
    with pytest.raises(KeyError):
        lamas.Runner(policy={"controller": {"type": "fixed", "layers": [["NoSuchOperator"]]}}, backend="mock").plan("q")
    with pytest.raises(ValueError):
        lamas.load_policy({"controller": {"type": "fixed", "layers": []}})


def test_the_agreement_stop():
    stop = lamas.load_policy({"controller": {"type": "fixed", "layers": [["Generate"]]},
                              "stop": {"type": "agreement", "min_solutions": 2, "consensus": 1.0}}).stop
    state = dict(query="q", n_total=3, n_completed=2, n_pending=1, elapsed=1.0)
    assert stop.should_stop(StopState(n_solutions=2, consensus=1.0, **state))
    assert not stop.should_stop(StopState(n_solutions=2, consensus=0.5, **state))
    assert not stop.should_stop(StopState(n_solutions=1, consensus=1.0, **state))


async def test_agreeing_answers_prune_the_vote():
    policy = {"controller": {"type": "fixed", "layers": [["GenerateCoT", "Generate"], ["ScEnsemble"]]},
              "stop": {"type": "agreement", "min_solutions": 2}}
    trace = (await lamas.Runner(policy=policy, backend=MockBackend(time_scale=0.01, error_rate=0.0)).arun("What is 6 * 7?")).trace
    assert trace.stop_reason == "agreement" and trace.pruned == ["ScEnsemble#0"] and "42" in trace.answer


def test_plugins_load_from_files_and_modules(tmp_path):
    (tmp_path / "my_ops.py").write_text(
        "import lamas\n\n@lamas.operator\nasync def Echo(ctx):\n    \"\"\"Returns the problem.\"\"\"\n    return ctx.problem\n")
    assert lamas.load_plugins(["my_ops.py", "json"], base_dir=str(tmp_path)) == ["lamas_plugin_my_ops", "json"]
    assert "Echo" in OPERATORS
    with pytest.raises(FileNotFoundError):
        lamas.load_plugins("missing.py", base_dir=str(tmp_path))


def test_the_example_config_loads():
    cfg = load_config(EXAMPLE)
    assert "PriceLookup" in OPERATORS and "Calculator" in OPERATORS
    assert os.path.isabs(cfg["task"]["data"]["test"]) and os.path.exists(cfg["task"]["data"]["test"])


async def test_the_example_runs_offline():
    """The shop example end to end, on a simulated LLM that does the arithmetic it is shown."""
    import sys

    cfg = load_config(EXAMPLE)
    lookup = sys.modules["lamas_plugin_tools"].lookup

    def llm(prompt):            # asked for an expression, it writes one; otherwise it adds up the bill it was given
        if "Python arithmetic expression" in prompt:
            return " + ".join(f"{n} * {price}" for n, _, price in lookup(prompt)) + "\n"
        return None

    report = await aevaluate(cfg["task"], cfg["policy"], MockBackend(time_scale=0, error_rate=0.0, answer_fn=llm), concurrency=3)
    assert report.n == 6 and report.errors == 0 and report.accuracy == 1.0
    assert all(r["layers"] == [["PriceLookup", "Calculator"], ["ScEnsemble"]] for r in report.rows)
    assert report.stopped_early == 1.0 and report.llm_calls_mean == 2.0       # the tools agree, so the vote never runs


async def test_the_calculator_computes_with_python():
    load_config(EXAMPLE)
    calc = OPERATORS.get("Calculator")()
    backend = MockBackend(time_scale=0, answer_fn=lambda prompt: "3 * 4.50 + 2 * 1.20\n")     # a reply in a code block
    ctx = lamas.Context(problem="A customer orders 3 notebooks and 2 pens. What is the total, in dollars?",
                        task=lamas.get_task("custom"), backend=backend)
    out = await calc.run(ctx)
    assert out.endswith("Final answer: 15.9")
