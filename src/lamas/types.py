"""Plain data types shared by every part of LAMaS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# What an operator reads. It decides where the scheduler places the node.
NEEDS_PROBLEM = "problem"      # only the problem: runs in the first wave
NEEDS_SOLUTION = "solution"    # the latest upstream solution: waits for one node
NEEDS_SOLUTIONS = "solutions"  # every solution so far: a merge node
NEEDS_NONE = "none"            # control token (EarlyStop); never executed


@dataclass
class Usage:
    """Token and call counters for one or more LLM calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.llm_calls += other.llm_calls

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "llm_calls": self.llm_calls,
        }


@dataclass
class Completion:
    """One LLM response."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0


@dataclass
class Node:
    """One operator invocation in a plan."""

    id: str                      # e.g. "GenerateCoT#0"
    operator: str                # registry name
    needs: str                   # see NEEDS_*
    wave: int = 0                # 0 = no dependencies; larger = later
    layer: int = 0               # controller layer the pick came from
    position: int = 0            # position inside that layer
    deps: List[str] = field(default_factory=list)       # nodes that must finish first
    solution_from: Optional[str] = None                 # node whose output is the input solution
    solutions_from: List[str] = field(default_factory=list)  # nodes whose outputs are merged

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "operator": self.operator, "needs": self.needs, "wave": self.wave,
            "layer": self.layer, "position": self.position, "deps": list(self.deps),
        }


@dataclass
class Plan:
    """The execution graph the controller and scheduler produced for one query."""

    query: str
    layers: List[List[str]]                 # the controller's picks, layer by layer
    nodes: Dict[str, Node]                  # insertion order = serial order
    controller_ms: float = 0.0
    # training-only: log-probability of each node's pick (a torch scalar) keyed by node id
    log_probs: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query, "layers": self.layers,
            "controller_ms": round(self.controller_ms, 2),
            "nodes": [n.to_dict() for n in self.nodes.values()],
        }


@dataclass
class Span:
    """What happened to one node during execution."""

    id: str
    operator: str
    wave: int
    deps: List[str]
    status: str = "pending"      # pending | running | done | pruned | cancelled | error
    start: Optional[float] = None   # seconds since execution began
    end: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    outputs: List[str] = field(default_factory=list)   # the solution(s) this node produced
    error: Optional[str] = None
    critical: bool = False       # on the critical path
    weight: float = 0.0          # path-criticality weight w(o) = l(o) / T

    @property
    def latency(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return self.end - self.start

    def to_dict(self, preview: int = 400) -> Dict[str, Any]:
        outs = [o if len(o) <= preview else o[:preview] + " ..." for o in self.outputs]
        return {
            "id": self.id, "operator": self.operator, "wave": self.wave, "deps": list(self.deps),
            "status": self.status,
            "start": None if self.start is None else round(self.start, 3),
            "end": None if self.end is None else round(self.end, 3),
            "latency": round(self.latency, 3),
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "llm_calls": self.llm_calls, "outputs": outs, "error": self.error,
            "critical": self.critical, "weight": round(self.weight, 3),
        }


@dataclass
class Trace:
    """The full record of one run: the DAG, timings, the critical path and what was pruned."""

    query: str
    plan: Plan
    spans: Dict[str, Span]
    answer: str = ""
    solutions: List[str] = field(default_factory=list)
    latency: float = 0.0            # end-to-end wall clock, seconds
    cost: float = 0.0               # USD
    usage: Usage = field(default_factory=Usage)
    critical_path: List[str] = field(default_factory=list)
    critical_path_latency: float = 0.0
    critical_path_tokens: int = 0
    pruned: List[str] = field(default_factory=list)
    stop_reason: Optional[str] = None
    stop_time: Optional[float] = None
    backend: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, preview: int = 400) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "latency": round(self.latency, 3),
            "cost": self.cost,
            "usage": self.usage.to_dict(),
            "critical_path": list(self.critical_path),
            "critical_path_latency": round(self.critical_path_latency, 3),
            "critical_path_tokens": self.critical_path_tokens,
            "pruned": list(self.pruned),
            "stop_reason": self.stop_reason,
            "stop_time": None if self.stop_time is None else round(self.stop_time, 3),
            "backend": self.backend,
            "plan": self.plan.to_dict(),
            "spans": [s.to_dict(preview) for s in self.spans.values()],
            "meta": self.meta,
        }

    def gantt(self, width: int = 60) -> str:
        """A text Gantt chart of the run, for terminals and logs."""
        total = max(self.latency, 1e-9)
        lines = []
        name_w = max([len(s.id) for s in self.spans.values()] + [8])
        for s in self.spans.values():
            if s.start is None or s.end is None:
                bar = " " * width
                tag = s.status
            else:
                a = int(round(s.start / total * width))
                b = max(a + 1, int(round(s.end / total * width)))
                ch = "#" if s.critical else "="
                bar = " " * a + ch * (b - a) + " " * max(0, width - b)
                tag = f"{s.latency:5.2f}s" + (" CP" if s.critical else "")
            lines.append(f"{s.id:<{name_w}} |{bar}| {tag}")
        lines.append(f"{'total':<{name_w}}  {self.latency:.2f}s, {self.usage.llm_calls} LLM calls, "
                     f"${self.cost:.5f}" + (f", stopped early ({self.stop_reason})" if self.stop_reason else ""))
        return "\n".join(lines)


@dataclass
class Result:
    """What `lamas.run` returns."""

    answer: str
    trace: Trace

    def __str__(self) -> str:
        return self.answer
