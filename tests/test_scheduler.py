from lamas.critical_path import analyse
from lamas.scheduler import schedule, waves

LAYERS = [["GenerateCoT", "SelfRefine", "Generate"], ["GenerateCoT", "ScEnsemble"], ["ScEnsemble", "EarlyStop"]]


def test_only_data_dependencies_are_kept():
    nodes = schedule(LAYERS)
    assert list(nodes) == ["GenerateCoT#0", "SelfRefine#0", "Generate#0", "GenerateCoT#1", "ScEnsemble#0", "ScEnsemble#1"]
    assert nodes["GenerateCoT#0"].deps == [] and nodes["Generate#0"].deps == [] and nodes["GenerateCoT#1"].deps == []
    assert nodes["SelfRefine#0"].deps == ["GenerateCoT#0"]
    assert nodes["ScEnsemble#0"].deps == ["GenerateCoT#0", "SelfRefine#0", "Generate#0", "GenerateCoT#1"]
    assert nodes["ScEnsemble#1"].deps == ["ScEnsemble#0"]
    assert waves(nodes) == [["GenerateCoT#0", "Generate#0", "GenerateCoT#1"], ["SelfRefine#0"], ["ScEnsemble#0"], ["ScEnsemble#1"]]


def test_early_stop_is_never_scheduled():
    assert all(n.operator != "EarlyStop" for n in schedule(LAYERS).values())


def test_critical_path_and_weights():
    # a -> c, b -> c ; a is slow, b is fast
    deps = {"a": [], "b": [], "c": ["a", "b"]}
    path, total, w = analyse(deps, {"a": 4.0, "b": 1.0, "c": 2.0})
    assert path == ["a", "c"] and total == 6.0
    assert w["a"] == 1.0 and w["c"] == 1.0 and abs(w["b"] - 0.5) < 1e-9   # l(b) = 1 + 2 = 3, T = 6


def test_critical_path_ignores_nodes_that_did_not_run():
    path, total, w = analyse({"a": [], "b": ["a"], "c": ["b"]}, {"a": 1.0, "b": 2.0})
    assert path == ["a", "b"] and total == 3.0 and "c" not in w
