"""The public API: ``run`` = ``plan`` + ``execute``.

    import lamas
    result = lamas.run(problem, backend="http://localhost:8000/v1")
    print(result.answer)
    print(result.trace.gantt())
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from .backends import Backend, make_backend
from .executor import execute as _execute
from .policy import Policy, load_policy
from .registry import OPERATORS, TASKS
from .scheduler import schedule
from .stop import StopPolicy
from .types import NEEDS_NONE, Plan, Result, Trace


def get_task(spec: Any = None) -> Any:
    """A task instance from a name, a ``{"name": ..., **options}`` dict or an instance."""
    from . import tasks  # noqa: F401  (registers the built-in tasks)

    if spec is None:
        spec = "general"
    if isinstance(spec, str):
        return TASKS.get(spec)()
    if isinstance(spec, dict):
        cfg = dict(spec)
        return TASKS.get(cfg.pop("name", "general"))(**cfg)
    return spec


class Runner:
    """A configured runtime that can answer many queries.

    Args:
        policy: ``"lamas/<name>"``, a policy folder, or a :class:`Policy`. Default: the released policy.
        backend: a :class:`Backend`, a base URL, or a config dict (see ``make_backend``).
        task: task name or instance; sets prompts, answer extraction and scoring. Default: the policy's task.
        budget: e.g. ``{"latency_s": 15}``: answer with the best solution so far when the time is up.
        operators: operator pool offered to the controller; defaults to the policy's.
        seed: makes the controller's draws repeatable.
    """

    def __init__(self, policy: Any = None, backend: Any = None, task: Any = None,
                 budget: Optional[Mapping[str, Any]] = None, operators: Optional[List[str]] = None,
                 seed: Optional[int] = None):
        self.policy: Policy = load_policy(policy)
        self.task = get_task(task if task is not None else self.policy.task)
        self._owns_backend = not isinstance(backend, Backend)
        self.backend: Backend = make_backend(backend)
        self.stop: StopPolicy = self.policy.stop
        self.budget = dict(budget) if budget else None
        self.operators = list(operators or self.policy.operators or self.task.operators)
        self.seed = seed
        for name in self.operators:
            OPERATORS.get(name)             # fail early on an unknown operator
        if getattr(self.policy.controller, "name", "") == "supernet" or getattr(self.stop, "name", "") == "learned":
            from .encoder import warm_up

            warm_up()                       # load the sentence encoder now, not on the first query

    # ---- plan ----------------------------------------------------------------------------
    def plan(self, query: str, *, seed: Optional[int] = None, track_grad: bool = False) -> Plan:
        started = time.perf_counter()
        selection = self.policy.controller.select(
            query, self.operators, seed=self.seed if seed is None else seed, track_grad=track_grad)
        nodes = schedule(selection.layers)
        plan = Plan(query=query, layers=selection.layers, nodes=nodes,
                    controller_ms=(time.perf_counter() - started) * 1000.0)
        if selection.log_probs:
            # map every pick to its node; control tokens (EarlyStop) have no node of their own
            ids = iter(nodes)
            extra = []
            for names, lps in zip(selection.layers, selection.log_probs):
                for name, lp in zip(names, lps):
                    if getattr(OPERATORS.get(name), "needs", None) == NEEDS_NONE:
                        extra.append(lp)
                    else:
                        plan.log_probs[next(ids)] = lp
            plan.meta["control_log_probs"] = extra
        plan.meta["probs"] = selection.probs
        return plan

    # ---- execute -------------------------------------------------------------------------
    async def aexecute(self, plan: Plan, *, item: Optional[dict] = None, on_event: Optional[Callable] = None,
                       budget: Optional[Mapping[str, Any]] = None, started: Optional[float] = None) -> Trace:
        return await _execute(plan, self.backend, task=self.task, item=item, stop=self.stop,
                              budget=budget if budget is not None else self.budget, on_event=on_event, started=started)

    async def arun(self, query: str, *, item: Optional[dict] = None, on_event: Optional[Callable] = None,
                   budget: Optional[Mapping[str, Any]] = None, seed: Optional[int] = None) -> Result:
        started = time.perf_counter()
        plan = self.plan(query, seed=seed)
        trace = await self.aexecute(plan, item=item, on_event=on_event, budget=budget, started=started)
        trace.meta.update({"policy": self.policy.name, "task": self.task.name, "controller_ms": round(plan.controller_ms, 2)})
        if not trace.answer:
            errors = [s.error for s in trace.spans.values() if s.error]
            if errors:
                raise RuntimeError(f"every operator failed; first error: {errors[0]}")
        return Result(answer=trace.answer, trace=trace)

    def run(self, query: str, **kwargs: Any) -> Result:
        return _sync(self.arun(query, **kwargs))

    async def aclose(self) -> None:
        if self._owns_backend:
            await self.backend.aclose()

    def describe(self) -> Dict[str, Any]:
        return {"policy": self.policy.describe(), "task": self.task.name, "backend": self.backend.describe(),
                "operators": self.operators, "budget": self.budget}


def _sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError("an event loop is already running here; use the async form (`await lamas.arun(...)`)")


# ---- one-shot functions --------------------------------------------------------------------
async def arun(query: str, *, policy: Any = None, backend: Any = None, task: Any = None,
               budget: Optional[Mapping[str, Any]] = None, item: Optional[dict] = None,
               on_event: Optional[Callable] = None, seed: Optional[int] = None,
               operators: Optional[List[str]] = None) -> Result:
    """Answer one query. See :class:`Runner` for the arguments."""
    runner = Runner(policy=policy, backend=backend, task=task, budget=budget, operators=operators, seed=seed)
    try:
        return await runner.arun(query, item=item, on_event=on_event)
    finally:
        await runner.aclose()


def run(query: str, **kwargs: Any) -> Result:
    """Answer one query (blocking). ``result.answer`` is the text, ``result.trace`` the record."""
    return _sync(arun(query, **kwargs))


def plan(query: str, *, policy: Any = None, task: Any = None, operators: Optional[List[str]] = None,
         seed: Optional[int] = None) -> Plan:
    """Only decide what would run: the controller picks operators, the scheduler orders them."""
    from .backends.mock import MockBackend

    return Runner(policy=policy, backend=MockBackend(time_scale=0), task=task, operators=operators, seed=seed).plan(query)


async def aexecute(plan_: Plan, backend: Any = None, *, task: Any = None, stop: Optional[StopPolicy] = None,
                   budget: Optional[Mapping[str, Any]] = None, item: Optional[dict] = None,
                   on_event: Optional[Callable] = None) -> Trace:
    """Run a plan and return its trace."""
    be = make_backend(backend)
    try:
        return await _execute(plan_, be, task=get_task(task), item=item, stop=stop, budget=budget, on_event=on_event)
    finally:
        if not isinstance(backend, Backend):
            await be.aclose()


def execute(plan_: Plan, backend: Any = None, **kwargs: Any) -> Trace:
    return _sync(aexecute(plan_, backend, **kwargs))
