"""The clinician tutorial (docs/tutorial.md), offline: the data, the agents, the config, and
lamas.compare with LAMaS and without."""
import json
import os
import runpy

import lamas
from lamas.config import load_config, runner_from_config

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, "..", "examples", "clinician")
CONFIG = os.path.join(FOLDER, "clinician.yaml")
with open(os.path.join(FOLDER, "data", "medqa_sample.jsonl")) as f:
    CASES = [json.loads(line) for line in f]


class Recording(lamas.MockBackend):
    """The simulated LLM, keeping every prompt."""

    def __init__(self, **kwargs):
        super().__init__(time_scale=0, error_rate=0, **kwargs)
        self.prompts = []

    async def complete(self, prompt, **kwargs):
        self.prompts.append("\n".join(m["content"] for m in prompt) if isinstance(prompt, list) else prompt)
        return await super().complete(prompt, **kwargs)


def test_the_sample_is_ten_medqa_cases_with_their_options():
    assert len(CASES) == 10
    for case in CASES:
        assert set(case) == {"id", "question", "answer"} and case["answer"] in "ABCD"
        assert [line[:3] for line in case["question"].splitlines()[-4:]] == ["A) ", "B) ", "C) ", "D) "]


def test_get_medqa_converts_a_row():
    convert = runpy.run_path(os.path.join(FOLDER, "get_medqa.py"))["convert"]
    row = {"question": "Which one? ", "options": {"A": "x", "B": "y", "C": "z", "D": "w"}, "answer_idx": "C"}
    assert convert(row, 7) == {"id": "medqa-test-0007", "question": "Which one?\nA) x\nB) y\nC) z\nD) w", "answer": "C"}


def test_the_config_reads_the_data_and_scores_the_letter():
    runner = runner_from_config(load_config(CONFIG), backend=Recording())
    item = runner.task.load("test")[1]
    assert runner.task.problem(item) == CASES[1]["question"]
    assert runner.task.score(item, "The pathologist is right.\nFinal answer: D") == 1.0
    assert runner.task.score(item, "**Final answer:** (A)") == 0.0
    assert list(runner.plan("any case").nodes) == ["Internist#0", "Pathologist#0", "Pharmacist#0", "Attending#0"]


def test_compare_when_the_team_agrees():
    backend = Recording(answers={c["question"]: c["answer"] for c in CASES})
    report = lamas.compare(CONFIG, backend=backend, limit=3, concurrency=3)
    assert len(report.rows) == 3 and report.simulated
    for row in report.rows:
        without, with_ = row["without"], row["with"]
        assert without["llm_calls"] == 4 and not without["stopped"] and without["pruned"] == []
        assert with_["llm_calls"] == 3 and with_["stopped"] and with_["pruned"] == ["Attending#0"]
        assert without["key"] == with_["key"] == CASES[row["index"]]["answer"].lower()
    assert "not measured on the simulated LLM" in report.table()


def test_without_lamas_each_agent_waits_for_the_one_before():
    backend = Recording(answers={c["question"]: c["answer"] for c in CASES})
    trace = lamas.compare(CONFIG, backend=backend, limit=1).rows[0]["without"]["trace"]
    deps = {node_id: node.deps for node_id, node in trace.plan.nodes.items()}
    assert deps["Internist#0"] == [] and deps["Pathologist#0"] == ["Internist#0"]
    assert deps["Pharmacist#0"] == ["Pathologist#0"]
    assert set(deps["Attending#0"]) == {"Internist#0", "Pathologist#0", "Pharmacist#0"}


def test_the_attending_reads_the_team_when_it_disagrees():
    def letter(prompt):
        case = next(c for c in CASES if c["question"] in prompt)
        return "A" if "clinical pharmacist" in prompt else case["answer"]

    backend = Recording(answer_fn=letter)
    with_ = lamas.compare(CONFIG, backend=backend, limit=1).rows[0]["with"]
    assert with_["llm_calls"] == 4 and not with_["stopped"] and with_["pruned"] == []
    attending = [p for p in backend.prompts if p.startswith("You are the attending physician")]
    assert attending and CASES[0]["question"] in attending[-1] and "The answer is A" in attending[-1]


def test_the_compare_script_gives_both_sides_the_same_replies():
    SimulatedLLM = runpy.run_path(os.path.join(FOLDER, "compare.py"))["SimulatedLLM"]
    backend = SimulatedLLM()
    backend.time_scale = 0
    report = lamas.compare(CONFIG, backend=backend, concurrency=10)
    assert len(report.rows) == 10
    for row in report.rows:
        assert row["without"]["llm_calls"] == 4 and row["with"]["llm_calls"] in (3, 4)
        assert all(row["with"]["trace"].spans[k].outputs == row["without"]["trace"].spans[k].outputs
                   for k in ("Internist#0", "Pathologist#0", "Pharmacist#0"))


def test_the_compare_script_puts_backend_and_model_on_top_of_the_config():
    llm = runpy.run_path(os.path.join(FOLDER, "compare.py"))["llm"]
    assert type(llm()).__name__ == "SimulatedLLM"
    assert llm("https://api.openai.com/v1", "gpt-4o-mini") == {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"}
    assert llm(model="qwen3.5-4b") == {"base_url": "http://localhost:8000/v1", "model": "qwen3.5-4b"}
