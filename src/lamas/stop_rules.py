"""Stop policies written as rules, for tasks that have no trained stop gate.

    stop: {type: agreement, min_solutions: 2, consensus: 1.0}
"""

from __future__ import annotations

from typing import Any, Dict

from .registry import register_stop_policy
from .stop import StopPolicy, StopState


@register_stop_policy(name="agreement")
class AgreementStop(StopPolicy):
    """Stop once enough solutions agree.

    Args:
        min_solutions: solutions that must be on the table before it may stop.
        consensus: share of them that must carry the most common answer (1.0: all of them).
    """

    name = "agreement"

    def __init__(self, min_solutions: int = 2, consensus: float = 1.0):
        self.min_solutions = int(min_solutions)
        self.consensus = float(consensus)

    def should_stop(self, state: StopState) -> bool:
        return state.n_solutions >= self.min_solutions and state.consensus >= self.consensus

    def describe(self) -> Dict[str, Any]:
        return {"type": self.name, "min_solutions": self.min_solutions, "consensus": self.consensus}
