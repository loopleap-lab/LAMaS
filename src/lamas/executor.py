"""Executor: runs a plan as a DAG. Every node starts the moment its inputs are ready.

After each completion the stop policy is asked whether the remaining nodes are still worth
running. If not, nodes that have not started are pruned; nodes that are running finish. A
latency budget is a deadline: when it passes, running nodes are cancelled as well.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import answers
from .backends.base import Backend
from .critical_path import analyse
from .operators.base import Context, Operator
from .registry import OPERATORS
from .stop import StopPolicy, StopState
from .types import NEEDS_SOLUTIONS, Node, Plan, Span, Trace, Usage

FINALIZE = "Finalize"
Event = Dict[str, Any]


async def execute(plan: Plan, backend: Backend, *, task: Any, item: Optional[dict] = None,
                  stop: Optional[StopPolicy] = None, budget: Optional[Mapping[str, Any]] = None,
                  operators: Optional[Mapping[str, Operator]] = None,
                  on_event: Optional[Callable[[Event], None]] = None,
                  started: Optional[float] = None) -> Trace:
    """Run ``plan`` against ``backend`` and return the trace.

    Args:
        task: supplies prompts, answer extraction and the final aggregation step.
        item: the data record behind the query, for tasks that need more than the text.
        stop: consulted after every completed node. Default: never stop.
        budget: ``{"latency_s": 15}`` sets a deadline; when it passes, nothing new is launched,
            running nodes are cancelled and the best answer so far is returned.
        operators: operator instances by name, to override the registry (options, mocks).
        on_event: called with a small dict on plan / start / end / stop / final, for live views.
        started: ``time.perf_counter()`` taken before planning, so latency includes the controller.
    """
    stop = stop or StopPolicy()
    item = item if item is not None else {"problem": plan.query}
    t0 = started if started is not None else time.perf_counter()
    now = lambda: time.perf_counter() - t0  # noqa: E731
    emit = on_event or (lambda event: None)

    nodes = plan.nodes
    spans: Dict[str, Span] = {n.id: Span(id=n.id, operator=n.operator, wave=n.wave, deps=list(n.deps)) for n in nodes.values()}
    done: Dict[str, asyncio.Event] = {n: asyncio.Event() for n in nodes}
    outputs: Dict[str, List[str]] = {}
    running: Dict[str, asyncio.Task] = {}
    instances: Dict[str, Operator] = {}

    def instance(name: str) -> Operator:
        if operators and name in operators:
            return operators[name]
        if name not in instances:
            instances[name] = OPERATORS.get(name)()
        return instances[name]

    halted = asyncio.Event()
    halt: Dict[str, Any] = {"reason": None, "time": None, "first_only": False}
    table: Counter = Counter()          # answer key -> count, for the solutions currently on the table
    counters = {"completed": 0}
    query_embedding: Dict[str, Any] = {}

    def embed_query() -> Any:
        if "v" not in query_embedding:
            from .encoder import embed_one

            query_embedding["v"] = embed_one(plan.query)
        return query_embedding["v"]

    key_cache: Dict[str, str] = {}

    async def key_of(solution: str) -> str:
        if solution not in key_cache:
            key_cache[solution] = (await task.akey(solution, item)) or "h:" + answers.text_hash(solution)
        return key_cache[solution]

    async def majority(candidates: List[str]) -> Optional[str]:
        """The most recent candidate that carries the most common answer."""
        if not candidates:
            return None
        keys = [await key_of(c) for c in candidates]
        best = Counter(keys).most_common(1)[0][0]
        for candidate, key in zip(reversed(candidates), reversed(keys)):
            if key == best:
                return candidate
        return candidates[-1]

    def do_halt(reason: str, cancel: bool, except_id: Optional[str] = None) -> None:
        if halted.is_set():
            return
        halted.set()
        halt["reason"], halt["time"] = reason, now()
        emit({"type": "stop", "reason": reason, "t": round(halt["time"], 3)})
        if cancel:
            _cancel_others(except_id)

    async def run_node(node: Node) -> None:
        span = spans[node.id]
        try:
            if node.deps:
                await asyncio.gather(*(done[d].wait() for d in node.deps))
            if halted.is_set():
                span.status = "pruned"
                emit({"type": "prune", "id": node.id, "t": round(now(), 3)})
                return
            solution = ""
            if node.solution_from and outputs.get(node.solution_from):
                solution = outputs[node.solution_from][-1]
            merged: List[str] = []
            for src in node.solutions_from:
                merged.extend(outputs.get(src, []))
            usage = Usage()
            ctx = Context(problem=plan.query, task=task, backend=backend, item=item,
                          solution=solution, solutions=merged, usage=usage)
            span.status, span.start = "running", now()
            emit({"type": "start", "id": node.id, "t": round(span.start, 3)})
            try:
                raw = await instance(node.operator).run(ctx)
                texts = raw if isinstance(raw, (list, tuple)) else [raw]
                span.outputs = [task.clean(t, item) for t in texts if t]
                span.status = "done"
            except asyncio.CancelledError:
                span.status = "cancelled"
            except Exception as exc:  # an operator failure must not take the whole query down
                span.status, span.error = "error", f"{type(exc).__name__}: {exc}"
            span.end = now()
            span.prompt_tokens, span.completion_tokens, span.llm_calls = usage.prompt_tokens, usage.completion_tokens, usage.llm_calls
            if span.status == "done":
                outputs[node.id] = span.outputs
            emit({"type": "end", "id": node.id, "t": round(span.end, 3), "status": span.status,
                  "tokens": usage.completion_tokens, "calls": usage.llm_calls})
            if span.status != "done":
                return

            # ---- stop decision ----------------------------------------------------------
            counters["completed"] += 1
            if node.needs == NEEDS_SOLUTIONS and span.outputs:
                table.clear()                      # a merge replaces what it merged
            for s in span.outputs:
                table[await key_of(s)] += 1
            if halted.is_set():
                if halt["first_only"] and span.outputs:     # the deadline passed before anything was ready
                    halt["first_only"] = False
                    _cancel_others(node.id)
                return
            pending = sum(1 for s in spans.values() if s.status in ("pending", "running"))
            n_solutions = sum(table.values())
            if pending > 0 and n_solutions > 0:
                state = StopState(query=plan.query, n_total=len(nodes), n_completed=counters["completed"],
                                  n_pending=pending, n_solutions=n_solutions,
                                  consensus=max(table.values()) / n_solutions, elapsed=now(),
                                  embed_query=embed_query)
                if stop.should_stop(state):
                    do_halt(stop.name, cancel=False)
        except asyncio.CancelledError:
            # cancelled while still waiting for its inputs: it never started
            if span.status == "pending":
                span.status = "pruned"
                emit({"type": "prune", "id": node.id, "t": round(now(), 3)})
        finally:
            running.pop(node.id, None)
            done[node.id].set()

    def _cancel_others(keep: Optional[str]) -> None:
        for node_id, running_task in list(running.items()):
            if node_id != keep:
                running_task.cancel()

    async def deadline(seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds - now()))
        if halted.is_set():
            return
        have_answer = any(outputs.values())
        halt["first_only"] = not have_answer
        do_halt("deadline", cancel=have_answer)

    emit({"type": "plan", "nodes": [n.to_dict() for n in nodes.values()], "layers": plan.layers, "t": round(now(), 3)})
    for node in nodes.values():
        running[node.id] = asyncio.ensure_future(run_node(node))
    watchdog = None
    latency_budget = (budget or {}).get("latency_s")
    if latency_budget:
        watchdog = asyncio.ensure_future(deadline(float(latency_budget)))
    node_tasks = list(running.values())
    try:
        if node_tasks:
            await asyncio.gather(*node_tasks, return_exceptions=True)
    except asyncio.CancelledError:      # the caller went away: stop everything
        for node_task in node_tasks:
            node_task.cancel()
        raise
    finally:
        if watchdog:
            watchdog.cancel()

    # ---- reduce to one answer ----------------------------------------------------------------
    solutions, current = _replay(nodes, outputs)
    answer = current
    final_usage = Usage()
    if halted.is_set() and halt["reason"] == "deadline" and solutions:
        answer = await majority(solutions) or current       # out of time: no final step
    elif solutions:
        sinks = _sinks(nodes, outputs)
        span = Span(id=FINALIZE, operator=FINALIZE, wave=1 + max((n.wave for n in nodes.values()), default=0), deps=sinks)
        ctx = Context(problem=plan.query, task=task, backend=backend, item=item, solution=current,
                      solutions=solutions, usage=final_usage)
        span.status, span.start = "running", now()
        emit({"type": "start", "id": FINALIZE, "t": round(span.start, 3), "deps": sinks})
        try:
            answer = await task.finalize(ctx, solutions, current)
            span.status = "done"
        except Exception as exc:
            span.status, span.error = "error", f"{type(exc).__name__}: {exc}"
            answer = await majority(solutions) or current
        span.end = now()
        span.outputs = [answer] if answer else []
        span.prompt_tokens, span.completion_tokens, span.llm_calls = final_usage.prompt_tokens, final_usage.completion_tokens, final_usage.llm_calls
        emit({"type": "end", "id": FINALIZE, "t": round(span.end, 3), "status": span.status,
              "tokens": final_usage.completion_tokens, "calls": final_usage.llm_calls})
        if span.llm_calls > 0 or span.latency > 0.05:
            spans[FINALIZE] = span          # only keep it when it did real work
        else:
            emit({"type": "drop", "id": FINALIZE})

    # ---- bookkeeping -------------------------------------------------------------------------
    usage = Usage()
    for s in spans.values():
        usage.add(Usage(s.prompt_tokens, s.completion_tokens, s.llm_calls))
    latency = {s.id: s.latency for s in spans.values() if s.status == "done"}
    path, cp_latency, weights = analyse({s.id: s.deps for s in spans.values()}, latency)
    for s in spans.values():
        s.critical = s.id in path
        s.weight = weights.get(s.id, 0.0)
    cp_tokens = sum(_path_tokens(spans[n]) for n in path)
    total = now()
    trace = Trace(
        query=plan.query, plan=plan, spans=spans, answer=task.final_text(answer or "", item), solutions=solutions,
        latency=total, cost=backend.cost(usage), usage=usage, critical_path=path, critical_path_latency=cp_latency,
        critical_path_tokens=int(cp_tokens), pruned=[s.id for s in spans.values() if s.status in ("pruned", "cancelled")],
        stop_reason=halt["reason"], stop_time=halt["time"], backend=backend.describe(),
        meta={"raw_answer": answer or ""},
    )
    emit({"type": "final", "t": round(total, 3), "answer": trace.answer, "latency": round(total, 3),
          "cost": trace.cost, "usage": usage.to_dict(), "critical_path": path, "pruned": trace.pruned,
          "stop_reason": trace.stop_reason})
    return trace


def _replay(nodes: Mapping[str, Node], outputs: Mapping[str, List[str]]) -> tuple:
    """Replay the completed nodes in serial order to recover `solutions` and the current solution."""
    solutions: List[str] = []
    current = ""
    for node in nodes.values():
        outs = outputs.get(node.id)
        if not outs:
            continue
        if node.needs == NEEDS_SOLUTIONS:
            solutions = list(outs)
        else:
            solutions.extend(outs)
        current = outs[-1]
    return solutions, current


def _sinks(nodes: Mapping[str, Node], outputs: Mapping[str, List[str]]) -> List[str]:
    used = {d for n in nodes.values() if n.id in outputs for d in n.deps}
    return [n for n in nodes if n in outputs and n not in used]


def _path_tokens(span: Span) -> float:
    """Output tokens a node puts on the critical path: parallel samples count once."""
    width = len(span.outputs) if len(span.outputs) > 1 else 1
    return span.completion_tokens / width
