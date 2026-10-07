"""StopPolicy: after every operator completion, decide whether the rest is still worth running."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .registry import STOP_POLICIES, register_stop_policy


@dataclass
class StopState:
    """What a stop policy may look at. All of it is available at run time."""

    query: str
    n_total: int                    # nodes in the plan
    n_completed: int                # nodes that finished so far
    n_pending: int                  # nodes not finished yet (waiting or running)
    n_solutions: int                # solutions currently on the table
    consensus: float                # share of those solutions that carry the most common answer
    elapsed: float                  # seconds since execution began
    embed_query: Callable[[], Any] = field(default=lambda: None, repr=False)   # lazy 384-d embedding

    @property
    def progress(self) -> float:
        return self.n_completed / max(self.n_total, 1)


class StopPolicy:
    """The interface. This base class never stops; subclass it and override :meth:`should_stop`."""

    name = "none"

    def should_stop(self, state: StopState) -> bool:
        return False

    def describe(self) -> Dict[str, Any]:
        return {"type": self.name}


register_stop_policy(StopPolicy, name="none")


@register_stop_policy(name="learned")
class LearnedGate(StopPolicy):
    """The stop gate of the paper (Section 3.6).

    A small network reads the progress, the answer consensus and a 4-d projection of the query
    embedding, and predicts the control advantage ``delta_k = R_k - R_N``: how much better it
    is to stop now than to run everything. The stop probability is ``sigmoid(delta / tau)``,
    and the decision is drawn from it.

    Args:
        weights: path to a checkpoint holding ``{"gate_state_dict": ..., "tau": ...}``.
        seed: makes the draws repeatable.
    """

    name = "learned"

    def __init__(self, weights: Optional[str] = None, tau: Optional[float] = None, hidden: int = 32,
                 seed: Optional[int] = None, net: Any = None):
        import torch

        self.net = net if net is not None else build_gate_net(hidden)
        self.tau = 0.1 if tau is None else float(tau)
        if weights:
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            state = ckpt.get("gate_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.net.load_state_dict(state)
            if tau is None and isinstance(ckpt, dict) and "tau" in ckpt:
                self.tau = float(ckpt["tau"])
        self.net.eval()
        self._rng = random.Random(seed)
        self.last_probability: Optional[float] = None

    def probability(self, state: StopState) -> float:
        import torch

        features = torch.tensor([state.progress, state.consensus], dtype=torch.float32)
        with torch.no_grad():
            delta = float(self.net(features, state.embed_query()))
        z = max(-60.0, min(60.0, delta / max(self.tau, 1e-6)))
        return 1.0 / (1.0 + math.exp(-z))

    def should_stop(self, state: StopState) -> bool:
        if state.n_solutions < 1:
            return False
        p = self.probability(state)
        self.last_probability = p
        return self._rng.random() < p

    def describe(self) -> Dict[str, Any]:
        return {"type": self.name, "tau": round(self.tau, 4)}


def build_gate_net(hidden: int = 32):
    """The gate network: ``[progress, consensus] + Linear(384 -> 4)(query) -> 32 -> 1``."""
    import torch
    from torch import nn

    class GateNet(nn.Module):
        STATE_DIM, QUERY_DIM, QUERY_COMPRESSED = 2, 384, 4

        def __init__(self) -> None:
            super().__init__()
            self.query_proj = nn.Linear(self.QUERY_DIM, self.QUERY_COMPRESSED)
            self.net = nn.Sequential(nn.Linear(self.STATE_DIM + self.QUERY_COMPRESSED, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 1))

        def forward(self, state, query_emb):
            q = self.query_proj(query_emb)
            return self.net(torch.cat([state, q], dim=-1)).squeeze(-1)

    return GateNet()


def make_stop(spec: Any = None, base_dir: Optional[str] = None) -> StopPolicy:
    """Build a stop policy from ``None``, a config dict or an instance.

    ``{"weights": "gate.pth"}`` is the learned stop gate; ``{"type": <name>, ...}`` is any
    registered stop policy.
    """
    import os

    if isinstance(spec, StopPolicy):
        return spec
    if not spec:
        return StopPolicy()
    cfg = dict(spec)
    kind = cfg.pop("type", "learned" if cfg.get("weights") else "none")
    if cfg.get("weights") and base_dir and not os.path.isabs(cfg["weights"]):
        cfg["weights"] = os.path.join(base_dir, cfg["weights"])
    return STOP_POLICIES.get(kind)(**cfg)
