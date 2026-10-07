"""Critical-path analysis of an executed graph.

``T = max over paths of the sum of t(o)`` is the end-to-end latency of the graph (Eq. 1 of the
paper). For every operator, ``l(o)`` is the length of the longest path through it and
``w(o) = l(o) / T`` is its path-criticality weight (Eq. 7): 1 on the critical path, close to 1
on near-critical paths, small for operators that could never delay the answer.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple


def analyse(deps: Mapping[str, List[str]], latency: Mapping[str, float]) -> Tuple[List[str], float, Dict[str, float]]:
    """Return ``(critical_path, T, weights)``.

    Args:
        deps: node id -> ids it depends on, in topological order (dependencies come first).
        latency: node id -> execution time; nodes missing here (pruned, failed) are ignored.
    """
    order = [n for n in deps if n in latency]
    if not order:
        return [], 0.0, {}
    live = set(order)
    parents = {n: [d for d in deps[n] if d in live] for n in order}
    children: Dict[str, List[str]] = {n: [] for n in order}
    for n in order:
        for d in parents[n]:
            children[d].append(n)

    finish: Dict[str, float] = {}       # longest path ending at n (inclusive)
    for n in order:
        finish[n] = max((finish[d] for d in parents[n]), default=0.0) + latency[n]
    tail: Dict[str, float] = {}         # longest path starting at n (inclusive)
    for n in reversed(order):
        tail[n] = latency[n] + max((tail[c] for c in children[n]), default=0.0)

    total = max(finish.values())
    weights = {n: ((finish[n] - latency[n] + tail[n]) / total if total > 0 else 0.0) for n in order}

    path: List[str] = []
    cur = max(order, key=lambda n: finish[n])
    while cur is not None:
        path.append(cur)
        cur = max(parents[cur], key=lambda d: finish[d]) if parents[cur] else None
    path.reverse()
    return path, total, weights
