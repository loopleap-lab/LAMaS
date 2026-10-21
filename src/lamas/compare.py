"""With LAMaS and without: the same operators, the same LLM and the same data, run two ways.

Without LAMaS, the workflow runs the way a plain multi-agent loop runs it: one operator after the
other, and every operator runs. With LAMaS, operators that do not wait for each other run at the
same time, and the stop policy prunes what is no longer needed.

    import lamas
    report = lamas.compare("my_task.yaml")          # or task=, policy=, backend= as in lamas.evaluate
    print(report.table())
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .api import Runner, _sync
from .stop import StopPolicy
from .types import Plan

SIDES = ("without", "with")


class OneAfterAnother(Runner):
    """A runner without LAMaS: each operator waits for the one before it, and nothing stops early."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.stop = StopPolicy()                # never stops
        self.budget = None

    def plan(self, query: str, **kwargs: Any) -> Plan:
        plan = super().plan(query, **kwargs)
        previous = None
        for wave, node in enumerate(plan.nodes.values()):      # the nodes are in serial order
            if previous is not None and previous not in node.deps:
                node.deps.append(previous)
            node.wave, previous = wave, node.id
        return plan


@dataclass
class Comparison:
    backend: str
    simulated: bool
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def good(self) -> List[Dict[str, Any]]:
        return [r for r in self.rows if all("error" not in r[side] for side in SIDES)]

    def mean(self, side: str, key: str) -> float:
        values = [r[side][key] for r in self.good()]
        return statistics.fmean(values) if values else 0.0

    def table(self) -> str:
        good = self.good()
        lat = {side: self.mean(side, "latency") for side in SIDES}
        stopped = {side: sum(1 for r in good if r[side]["stopped"]) for side in SIDES}
        faster = f"   {lat['without'] / lat['with']:.1f}x faster" if lat["with"] else ""
        lines = [f"{len(self.rows)} problems on {self.backend}" + (f" ({len(self.rows) - len(good)} errors)" if len(good) < len(self.rows) else ""),
                 f"                    without LAMaS   with LAMaS",
                 f"  latency, mean     {lat['without']:10.2f}s  {lat['with']:10.2f}s{faster}",
                 f"  LLM calls, mean   {self.mean('without', 'llm_calls'):11.1f}  {self.mean('with', 'llm_calls'):11.1f}",
                 f"  stopped early     {stopped['without']:>6} of {len(good):<3} {stopped['with']:>6} of {len(good)}"]
        if self.simulated:
            lines.append("  accuracy          not measured on the simulated LLM")
        else:
            lines.append(f"  accuracy          {self.mean('without', 'score') * 100:10.1f}%  {self.mean('with', 'score') * 100:10.1f}%")
        return "\n".join(lines)


async def acompare(config: Optional[str] = None, *, task: Any = None, policy: Any = None, backend: Any = None,
                   split: str = "test", limit: Optional[int] = None, concurrency: int = 1, seed: Optional[int] = 0,
                   on_problem: Optional[Callable[[Dict[str, Any]], None]] = None) -> Comparison:
    """Run every problem of a split without LAMaS and then with it.

    Args:
        config: a config file, as for ``lamas eval``; ``task``, ``policy`` and ``backend`` override it.
        concurrency: problems in flight at once. 1 keeps each latency free of the others' load.
        on_problem: called with each problem's row as it finishes, for a live view.
    """
    from .config import load_config, runner_from_config

    with_lamas = runner_from_config(load_config(config), task=task, policy=policy, backend=backend)
    without = OneAfterAnother(policy=with_lamas.policy, backend=with_lamas.backend, task=with_lamas.task,
                              operators=with_lamas.operators, seed=with_lamas.seed)
    task_obj = with_lamas.task
    data = task_obj.load(split, limit)
    sem = asyncio.Semaphore(max(1, concurrency))
    rows: List[Optional[Dict[str, Any]]] = [None] * len(data)

    async def side(runner: Runner, i: int, item: dict) -> Dict[str, Any]:
        try:
            result = await runner.arun(task_obj.problem(item), item=item, seed=None if seed is None else seed + i)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        trace = result.trace
        return {"latency": trace.latency, "llm_calls": trace.usage.llm_calls, "cost": trace.cost,
                "score": await task_obj.ascore(item, result.answer), "key": task_obj.answer_key(result.answer, item),
                "stopped": trace.stop_reason is not None, "pruned": list(trace.pruned), "trace": trace}

    async def one(i: int, item: dict) -> None:
        async with sem:
            row: Dict[str, Any] = {"index": i}
            row["without"] = await side(without, i, item)
            row["with"] = await side(with_lamas, i, item)
            rows[i] = row
            if on_problem:
                on_problem(row)

    try:
        await asyncio.gather(*(one(i, item) for i, item in enumerate(data)))
    finally:
        await with_lamas.aclose()
    return Comparison(backend=with_lamas.backend.describe(), simulated=with_lamas.backend.name == "mock",
                      rows=[r for r in rows if r])


def compare(config: Optional[str] = None, **kwargs: Any) -> Comparison:
    """Blocking :func:`acompare`."""
    return _sync(acompare(config, **kwargs))
