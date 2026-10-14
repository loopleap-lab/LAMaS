import hashlib
import os

import pytest

import lamas
from lamas.backends import MockBackend
from lamas.policy import available, find_policy_dir, load_policy
from lamas.tasks import HumanEval

needs_policy = pytest.mark.skipif("lamas/humaneval" not in available(), reason="the released policy is not installed")


def _items(split, n):
    try:
        return HumanEval().load(split, n)
    except FileNotFoundError:
        pytest.skip("benchmark data is not available")


def _knows(items, **kwargs):
    """A simulated LLM that answers every problem with its reference solution."""
    return MockBackend(answers={it["prompt"][:120]: it["prompt"] + it["canonical_solution"] for it in items}, **kwargs)


def test_an_unknown_policy_is_an_error():
    with pytest.raises(FileNotFoundError):
        load_policy("lamas/does-not-exist")


@needs_policy
def test_the_released_policy_loads_and_plans():
    policy = load_policy()
    assert policy.name == "lamas/humaneval" and policy.task == "humaneval"
    assert policy.stop.name == "learned" and 0 < policy.stop.tau < 1
    assert policy.operators[0] == "Generate" and "EarlyStop" in policy.operators and "Test" in policy.operators
    problem = _items("test", 1)[0]["prompt"]
    for seed in range(5):
        plan = lamas.plan(problem, seed=seed)
        first = next(iter(plan.nodes.values()))
        assert first.needs == "problem" and not first.deps          # a workflow starts with a generator
    assert lamas.plan(problem, seed=3).layers == lamas.plan(problem, seed=3).layers       # seeded = repeatable


@needs_policy
def test_the_released_files_match_their_checksums():
    folder = find_policy_dir("lamas/humaneval")
    with open(os.path.join(folder, "SHA256SUMS")) as f:
        for line in f:
            digest, name = line.split()
            with open(os.path.join(folder, name), "rb") as g:
                assert hashlib.sha256(g.read()).hexdigest() == digest, name


@needs_policy
async def test_the_released_policy_solves_problems_end_to_end():
    items = _items("test", 4)
    runner = lamas.Runner(backend=_knows(items, seed=5, time_scale=0.01))
    for seed, item in enumerate(items):
        result = await runner.arun(item["prompt"], item=item, seed=seed)
        assert await runner.task.ascore(item, result.answer) == 1.0
        assert result.trace.stop_reason in (None, "learned")
    assert runner.stop.last_probability is None or 0.0 <= runner.stop.last_probability <= 1.0


async def test_training_writes_a_loadable_policy(tmp_path):
    pytest.importorskip("sentence_transformers")
    from lamas.evaluation import aevaluate
    from lamas.training import atrain

    items = _items("train", 6)
    events = []
    policy = await atrain("humaneval", _knows(items, time_scale=0.002, seed=1), limit=6, repeats=1, batch_size=3,
                          out=str(tmp_path / "p"), gate_epochs=20, on_progress=events.append)
    assert os.path.exists(tmp_path / "p" / "policy.yaml") and os.path.exists(tmp_path / "p" / "controller.pth")
    assert [e for e in events if e["phase"] == "controller"]
    assert policy.meta["training"]["accuracy_floor"] > 0
    report = await aevaluate("humaneval", policy, _knows(items, time_scale=0.002, seed=2), split="train", limit=6, concurrency=3)
    assert report.n == 6 and report.accuracy == 1.0 and report.latency_mean > 0 and report.errors == 0
    assert "accuracy" in report.table()


def test_objective_matches_the_paper():
    from lamas.objective import Objective

    obj = Objective(latency_weight=0.005, cost_weight=3.0, accuracy_floor=0.9, lambda_acc=1.0, dual_lr=0.01)
    assert abs(obj.reward(1.0, 10.0, 0.001) - (1 / 0.9 - 0.05 - 0.003)) < 1e-9
    assert obj.update_dual(0.8) > 1.0          # below the floor: correctness gets more weight
    assert obj.update_dual(1.0) < obj.lambda_acc + 1e-9 + 0.01
