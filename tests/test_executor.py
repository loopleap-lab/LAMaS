import lamas
from lamas import Operator, StopPolicy, register_operator
from lamas.backends import MockBackend
from util import CHAIN, WIDE, fixed_policy

Q = "What is 17 * 24?"


class StopAfterFirst(StopPolicy):
    """A stop policy for the tests: stop as soon as one operator has finished."""

    name = "after-first"

    def should_stop(self, state):
        return state.n_completed >= 1


async def test_independent_operators_run_in_parallel(mock):
    trace = (await lamas.Runner(policy=fixed_policy(WIDE), backend=mock).arun(Q)).trace
    work = sum(s.latency for s in trace.spans.values())
    assert trace.usage.llm_calls == 6                           # 1 + 1 + 3 samples, then the merge
    assert trace.latency < 0.75 * work                          # less waiting than one operator at a time
    assert len(trace.critical_path) == 2 and trace.critical_path[-1] == "ScEnsemble#0"


async def test_a_chain_is_all_critical_path(mock):
    trace = (await lamas.Runner(policy=fixed_policy(CHAIN), backend=mock).arun(Q)).trace
    assert trace.critical_path == list(trace.spans) and not trace.pruned and trace.stop_reason is None


async def test_a_stop_policy_prunes_what_has_not_started(mock):
    result = await lamas.Runner(policy=fixed_policy(WIDE, stop=StopAfterFirst()), backend=mock).arun(Q)
    trace = result.trace
    assert trace.stop_reason == "after-first"
    assert trace.pruned == ["ScEnsemble#0"] and trace.spans["ScEnsemble#0"].status == "pruned"
    assert all(s.status == "done" for s in trace.spans.values() if s.id != "ScEnsemble#0")      # running operators finish
    assert "408" in result.answer


async def test_latency_budget_returns_the_best_answer_so_far():
    backend = MockBackend(seed=1, time_scale=0.05, error_rate=0.0)
    full = await lamas.Runner(policy=fixed_policy(CHAIN), backend=backend).arun(Q)
    cut = await lamas.Runner(policy=fixed_policy(CHAIN), backend=backend, budget={"latency_s": full.trace.latency * 0.4}).arun(Q)
    assert cut.trace.stop_reason == "deadline"
    assert cut.trace.latency < full.trace.latency * 0.8
    assert "408" in cut.answer and cut.trace.pruned


async def test_a_failing_operator_does_not_take_the_query_down(mock):
    @register_operator
    class Broken(Operator):
        name = "Broken"
        description = "Always fails."
        needs = "problem"

        async def run(self, ctx):
            raise ValueError("boom")

    result = await lamas.Runner(policy=fixed_policy([["Broken", "GenerateCoT"], ["ScEnsemble"]]), backend=mock).arun(Q)
    assert result.trace.spans["Broken#0"].status == "error" and "boom" in result.trace.spans["Broken#0"].error
    assert "408" in result.answer


async def test_events_tell_the_whole_story(mock):
    events = []
    await lamas.Runner(policy=fixed_policy(WIDE, stop=StopAfterFirst()), backend=mock).arun(Q, on_event=events.append)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "plan" and kinds[-1] == "final"
    assert kinds.count("start") == kinds.count("end") and "stop" in kinds and "prune" in kinds
    assert events[-1]["usage"]["llm_calls"] == 5 and events[-1]["pruned"] == ["ScEnsemble#0"]


def test_blocking_api_and_gantt(mock):
    result = lamas.run(Q, policy=fixed_policy([["GenerateCoT"]]), backend=mock)
    assert "408" in str(result)
    assert "GenerateCoT#0" in result.trace.gantt() and result.trace.to_dict()["usage"]["llm_calls"] == 1


async def test_a_new_operator_is_one_class(mock):
    """An operator declares what it needs; the scheduler places it and the executor counts its LLM calls."""
    @register_operator
    class Debate(Operator):
        name = "Debate"
        description = "Two agents argue for and against the current solution; a judge decides."
        needs = "solution"

        async def run(self, ctx):
            pro, con = await ctx.gather(ctx.ask("Argue for:\n{solution}"), ctx.ask("Argue against:\n{solution}"))
            return await ctx.ask("Problem: {problem}\nFor: {pro}\nAgainst: {con}\nDecide.", pro=pro, con=con)

    result = await lamas.Runner(policy=fixed_policy([["GenerateCoT"], ["Debate"]]), backend=mock).arun(Q)
    span = result.trace.spans["Debate#0"]
    assert span.deps == ["GenerateCoT#0"] and span.llm_calls == 3 and span.status == "done"
