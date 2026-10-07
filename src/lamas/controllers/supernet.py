"""The learned controller of the paper: a policy over an agentic supernet (Section 3.2).

For each of ``L`` layers a small network scores every operator against the query (cosine
similarity of two 32-d projections) and turns the scores into a distribution. Operators are
drawn without replacement until their cumulative probability passes a threshold, so the
width of every layer, and through the EarlyStop token the depth, adapts to the query.

Checkpoints written by the research code load unchanged: the parameter names match.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .. import encoder
from ..registry import OPERATORS, register_controller
from ..types import NEEDS_PROBLEM
from .base import Controller, Selection

EARLY_STOP = "EarlyStop"


class _Selector(nn.Module):
    """One layer of the policy."""

    def __init__(self, dim: int, hidden: int, first: bool):
        super().__init__()
        self.first = first
        self.operator_encoder = nn.Linear(dim if first else 2 * dim, hidden)
        self.query_encoder = nn.Linear(dim, hidden)

    def forward(self, query: torch.Tensor, ops: torch.Tensor, prev: Optional[torch.Tensor]) -> torch.Tensor:
        q = F.normalize(self.query_encoder(query.unsqueeze(0)), dim=1)
        if not self.first and prev is not None:
            ops = torch.cat([ops, prev.unsqueeze(0).expand(ops.size(0), -1)], dim=1)
        elif not self.first:
            ops = torch.cat([ops, torch.zeros_like(ops)], dim=1)
        o = F.normalize(self.operator_encoder(ops), dim=1)
        return (q @ o.T).squeeze(0)        # one score per operator


class SupernetNet(nn.Module):
    def __init__(self, layers: int = 4, hidden: int = 32, dim: int = encoder.DIM):
        super().__init__()
        self.layers = nn.ModuleList([_Selector(dim, hidden, first=(i == 0)) for i in range(layers)])


@register_controller(name="supernet")
class SupernetController(Controller):
    """Args:
        weights: checkpoint path (a plain ``state_dict``); omit for a freshly initialised policy.
        layers, hidden: network shape; must match the checkpoint.
        threshold: cumulative-probability threshold ``tau`` that sets the width of each layer.
        operator_texts: the exact text each operator was embedded with at training time.
            Operators missing here are embedded as ``"<name>: <description>"``.
    """

    name = "supernet"

    def __init__(self, weights: Optional[str] = None, layers: int = 4, hidden: int = 32,
                 threshold: float = 0.3, operator_texts: Optional[Dict[str, str]] = None):
        self.net = SupernetNet(layers=layers, hidden=hidden)
        self.weights = weights
        if weights:
            self.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.net.eval()
        self.threshold = threshold
        self.operator_texts = dict(operator_texts or {})
        self._emb_cache: Dict[tuple, torch.Tensor] = {}

    # ---- embeddings ----------------------------------------------------------------------
    def _operator_embeddings(self, operators: List[str]) -> torch.Tensor:
        key = tuple(operators)
        if key not in self._emb_cache:
            texts = [self.operator_texts.get(name) or OPERATORS.get(name).embedding_text() for name in operators]
            self._emb_cache[key] = encoder.embed(texts)
        return self._emb_cache[key]

    # ---- selection -----------------------------------------------------------------------
    def select(self, query: str, operators: List[str], *, seed: Optional[int] = None,
               track_grad: bool = False) -> Selection:
        rng = random.Random(seed)
        ops_emb = self._operator_embeddings(operators)
        q_emb = encoder.embed_one(query, max_chars=None)
        is_generator = [getattr(OPERATORS.get(n), "needs", None) == NEEDS_PROBLEM for n in operators]

        layers: List[List[str]] = []
        log_probs: List[List[Any]] = []
        probs_out: List[Dict[str, float]] = []
        prev: Optional[torch.Tensor] = None
        with torch.set_grad_enabled(track_grad):
            for depth, layer in enumerate(self.net.layers):
                scores = layer(q_emb, ops_emb, prev)
                logp = F.log_softmax(scores, dim=0)
                p = logp.exp().detach()
                picked = _pick(p.tolist(), self.threshold, rng)
                names = [operators[i] for i in picked]
                stop_here = False
                if depth == 0:
                    # the first layer must start with something that can produce a solution
                    gens = [i for i in picked if is_generator[i]]
                    if EARLY_STOP in names or not gens:
                        fallback = operators.index("Generate") if "Generate" in operators else _best(p, is_generator)
                        picked = [fallback]
                        stop_here = EARLY_STOP in names
                    else:
                        picked = [gens[0]] + [i for i in picked if i != gens[0]]
                    names = [operators[i] for i in picked]
                elif EARLY_STOP in names:
                    stop_here = True
                layers.append(names)
                log_probs.append([logp[i] for i in picked])
                probs_out.append({operators[i]: round(float(p[i]), 4) for i in range(len(operators))})
                prev = ops_emb[picked[0]]
                if stop_here:
                    break
        return Selection(layers=layers, log_probs=log_probs, probs=probs_out)

    def describe(self) -> Dict[str, Any]:
        return {"type": self.name, "layers": len(self.net.layers), "threshold": self.threshold}


def _pick(p: List[float], threshold: float, rng: random.Random) -> List[int]:
    """Indices drawn without replacement until their cumulative probability reaches the threshold."""
    remaining = list(range(len(p)))
    picked: List[int] = []
    total = 0.0
    while total < threshold and remaining:
        weights = [max(p[i], 1e-12) for i in remaining]
        choice = rng.choices(remaining, weights=weights, k=1)[0]
        remaining.remove(choice)
        picked.append(choice)
        total += p[choice]
    return picked or [max(range(len(p)), key=lambda i: p[i])]


def _best(p: torch.Tensor, mask: List[bool]) -> int:
    candidates = [i for i, ok in enumerate(mask) if ok] or list(range(len(mask)))
    return max(candidates, key=lambda i: float(p[i]))
