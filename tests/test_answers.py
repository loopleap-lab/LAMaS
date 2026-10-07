from lamas import Task, answers
from lamas.operators import fill


def test_final_line_is_read_and_normalised():
    assert answers.extract_final_line("Step 1 ...\nFinal answer: 408.") == "408"
    assert answers.extract_final_line("so the answer is 1,250.50") == "1250.5"
    assert answers.extract_final_line("no statement here") is None


def test_consensus_and_majority():
    key = Task().answer_key
    sols = ["... Final answer: 12", "I get. Final answer: 13", "the result. Final answer: 12.0"]
    assert answers.consensus([key(s) for s in sols]) == 2 / 3
    assert answers.majority(sols, key) == sols[2]                  # the most recent one with the majority answer


def test_fill_leaves_latex_and_code_alone():
    assert fill(r"Solve {problem}: \boxed{} and {x}", problem="P") == r"Solve P: \boxed{} and {x}"


def test_plain_task_scoring():
    assert Task().score({"answer": "69"}, "He can buy 69 gumballs.\nFinal answer: 69") == 1.0
    assert Task().score({"answer": "69"}, "Final answer: 68") == 0.0
