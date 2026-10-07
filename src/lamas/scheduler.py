"""Scheduler: turns the controller's layered picks into a dependency graph.

The controller emits layers of operator names. Read left to right, layer by layer, they define
a serial program over two pieces of state: the current solution and the list of solutions so
far. The scheduler replays that program symbolically to find out which earlier node each node
really reads, and keeps only those data dependencies. Everything that does not depend on an
unfinished node can then start at once.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from .registry import OPERATORS
from .types import NEEDS_NONE, NEEDS_PROBLEM, NEEDS_SOLUTION, NEEDS_SOLUTIONS, Node


def _needs(name: str, operators: Optional[Mapping[str, type]]) -> str:
    cls = operators[name] if operators and name in operators else OPERATORS.get(name)
    return getattr(cls, "needs", NEEDS_PROBLEM)


def schedule(layers: List[List[str]], operators: Optional[Mapping[str, type]] = None) -> Dict[str, Node]:
    """Build the node graph. The returned dict is in serial order, which is a topological order."""
    nodes: Dict[str, Node] = {}
    counts: Dict[str, int] = {}
    current: Optional[str] = None          # node holding the current solution
    sources: List[str] = []                # nodes whose outputs make up `solutions`

    for layer_idx, names in enumerate(layers):
        for pos, name in enumerate(names):
            needs = _needs(name, operators)
            if needs == NEEDS_NONE:
                continue
            node_id = f"{name}#{counts.get(name, 0)}"
            counts[name] = counts.get(name, 0) + 1
            node = Node(id=node_id, operator=name, needs=needs, layer=layer_idx, position=pos)

            if needs == NEEDS_SOLUTIONS:
                node.solutions_from = list(sources)
                node.solution_from = current
                node.deps = list(sources)
                sources = [node_id]            # a merge replaces what it merged
            elif needs == NEEDS_SOLUTION:
                node.solution_from = current
                node.deps = [current] if current else []
                sources.append(node_id)
            else:                              # NEEDS_PROBLEM: a generator
                sources.append(node_id)
            current = node_id

            node.wave = 1 + max((nodes[d].wave for d in node.deps), default=-1)
            nodes[node_id] = node
    return nodes


def waves(nodes: Mapping[str, Node]) -> List[List[str]]:
    """Node ids grouped by wave, for display."""
    out: List[List[str]] = []
    for node in nodes.values():
        while len(out) <= node.wave:
            out.append([])
        out[node.wave].append(node.id)
    return out
