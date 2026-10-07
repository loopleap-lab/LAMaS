"""Evaluate a policy on a task: accuracy, latency, cost and stability in one report."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .api import Runner, _sync


@dataclass
class Report:
    task: str
    policy: str
    backend: str
    n: int
    accuracy: float
    latency_mean: float
    latency_p50: float
    latency_p90: float
    cost_mean: float
    llm_calls_mean: float
    cp_tokens_mean: float
    pruned_share: float          # share of planned operators that never ran
    stopped_early: float         # share of queries where the stop policy fired
    errors: int
    wall_time: float
    date: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))
    rows: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    config: Dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self, rows: bool = False) -> Dict[str, Any]:
        d = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items() if k not in ("rows", "config")}
        d["config"] = self.config
        if rows:
            d["rows"] = self.rows
        return d

    def table(self) -> str:
        return (f"{self.task} · {self.policy} · {self.backend}\n"
                f"  n            {self.n}  ({self.errors} errors)\n"
                f"  accuracy     {self.accuracy * 100:.2f}%\n"
                f"  latency      mean {self.latency_mean:.2f}s   p50 {self.latency_p50:.2f}s   p90 {self.latency_p90:.2f}s\n"
                f"  cost         ${self.cost_mean:.6f} per query   {self.llm_calls_mean:.1f} LLM calls   "
                f"{self.cp_tokens_mean:.0f} critical-path tokens\n"
                f"  early stop   fired on {self.stopped_early * 100:.0f}% of queries, pruned {self.pruned_share * 100:.0f}% of operators")


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


async def aevaluate(task: Any = None, policy: Any = None, backend: Any = None, *, split: str = "test",
                    limit: Optional[int] = None, concurrency: int = 4, seed: Optional[int] = 0,
                    out: Optional[str] = None, on_progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
                    items: Optional[List[dict]] = None, budget: Optional[Dict[str, Any]] = None) -> Report:
    """Run ``policy`` on a split of ``task`` (default: the policy's own task).

    Args:
        concurrency: queries in flight at once. Use 1 for the isolated latency track; larger
            values measure latency under load.
        seed: query ``i`` uses ``seed + i`` for the controller's sampling; ``None`` = unseeded.
        out: a folder to write ``report.json`` and ``traces.jsonl`` into.
        budget: an optional latency budget per query, e.g. ``{"latency_s": 15}``.
    """
    runner = Runner(policy=policy, backend=backend, task=task, budget=budget)
    task_obj = runner.task
    data = items if items is not None else task_obj.load(split, limit)
    sem = asyncio.Semaphore(max(1, concurrency))
    rows: List[Optional[Dict[str, Any]]] = [None] * len(data)
    done_count = 0
    traces_file = None
    if out:
        os.makedirs(out, exist_ok=True)
        traces_file = open(os.path.join(out, "traces.jsonl"), "w", encoding="utf-8")

    async def one(i: int, item: dict) -> None:
        nonlocal done_count
        async with sem:
            row: Dict[str, Any] = {"index": i}
            try:
                result = await runner.arun(task_obj.problem(item), item=item, seed=None if seed is None else seed + i)
                trace = result.trace
                planned = len(trace.plan.nodes)
                row.update({
                    "score": await task_obj.ascore(item, result.answer), "latency": trace.latency, "cost": trace.cost,
                    "llm_calls": trace.usage.llm_calls, "cp_tokens": trace.critical_path_tokens,
                    "planned": planned, "pruned": len(trace.pruned), "stopped": trace.stop_reason is not None,
                    "answer": result.answer[:500], "layers": trace.plan.layers,
                })
                if traces_file:
                    traces_file.write(json.dumps({"index": i, "score": row["score"], "trace": trace.to_dict()}, ensure_ascii=False) + "\n")
                    traces_file.flush()          # a long run keeps what it has finished
            except Exception as exc:
                row.update({"score": 0.0, "error": f"{type(exc).__name__}: {exc}"})
            rows[i] = row
            done_count += 1
            if on_progress:
                on_progress(done_count, len(data), row)

    started = time.perf_counter()
    try:
        await asyncio.gather(*(one(i, item) for i, item in enumerate(data)))
    finally:
        if traces_file:
            traces_file.close()
        await runner.aclose()

    good = [r for r in rows if r and "error" not in r]
    lat = [r["latency"] for r in good]
    report = Report(
        task=task_obj.name, policy=runner.policy.name, backend=runner.backend.describe(),
        n=len(data), accuracy=sum(r["score"] for r in rows if r) / max(len(data), 1),
        latency_mean=statistics.fmean(lat) if lat else 0.0, latency_p50=_percentile(lat, 0.5), latency_p90=_percentile(lat, 0.9),
        cost_mean=statistics.fmean([r["cost"] for r in good]) if good else 0.0,
        llm_calls_mean=statistics.fmean([r["llm_calls"] for r in good]) if good else 0.0,
        cp_tokens_mean=statistics.fmean([r["cp_tokens"] for r in good]) if good else 0.0,
        pruned_share=sum(r["pruned"] for r in good) / max(sum(r["planned"] for r in good), 1),
        stopped_early=sum(1 for r in good if r["stopped"]) / max(len(good), 1),
        errors=len(data) - len(good), wall_time=time.perf_counter() - started,
        rows=[r for r in rows if r], config={"split": split, "limit": limit, "concurrency": concurrency, "seed": seed,
                                             **runner.describe()},
    )
    if out:
        with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report.to_dict(rows=True), f, indent=2, ensure_ascii=False)
    return report


def evaluate(task: Any = None, policy: Any = None, backend: Any = None, **kwargs: Any) -> Report:
    """Blocking form of :func:`aevaluate`."""
    return _sync(aevaluate(task, policy, backend, **kwargs))
