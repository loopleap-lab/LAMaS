"""Objective: how a run is scored for training.

The constrained problem of the paper (Eq. 4) minimises latency and cost subject to an accuracy
floor ``S0``. Its Lagrangian relaxation gives the reward (Eq. 5)

    R = lambda_S * S / S0  -  lambda_t * T  -  lambda_c * C

where the dual variable ``lambda_S`` rises when accuracy falls below the floor and relaxes when
the floor is met (Eq. 6). For credit assignment each operator pays only its share
``w(o) = l(o) / T`` of the latency penalty (Eq. 8): in full on the critical path, less elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .types import Trace


@dataclass
class Objective:
    latency_weight: float = 0.005       # lambda_t, per second
    cost_weight: float = 3.0            # lambda_c, per USD
    accuracy_floor: float = 1.0         # S0
    lambda_acc: float = 1.0             # lambda_S, the dual variable
    dual_lr: float = 0.1                # eta

    def reward(self, score: float, latency: float, cost: float) -> float:
        """The global reward of one run (Eq. 5)."""
        return self.lambda_acc * score / self._floor() - self.latency_weight * latency - self.cost_weight * cost

    def node_rewards(self, trace: Trace, score: float) -> Dict[str, float]:
        """Per-operator rewards with critical-path-aware latency credit (Eq. 8)."""
        base = self.lambda_acc * score / self._floor() - self.cost_weight * trace.cost
        out: Dict[str, float] = {}
        for span in trace.spans.values():
            if span.status != "done":
                continue
            out[span.id] = base - self.latency_weight * span.weight * trace.latency
        return out

    def update_dual(self, batch_accuracy: float) -> float:
        """Projected gradient ascent on the dual variable (Eq. 6). Returns the new value."""
        violation = 1.0 - batch_accuracy / self._floor()
        self.lambda_acc = max(0.0, self.lambda_acc + self.dual_lr * violation)
        return self.lambda_acc

    def _floor(self) -> float:
        return self.accuracy_floor if self.accuracy_floor > 0 else 1.0

    def to_dict(self) -> Dict[str, float]:
        return {"latency_weight": self.latency_weight, "cost_weight": self.cost_weight,
                "accuracy_floor": round(self.accuracy_floor, 4), "lambda_acc": round(self.lambda_acc, 4),
                "dual_lr": self.dual_lr}
