"""A hand-written workflow: the same operators for every query, so no training is needed.

    policy:
      controller:
        type: fixed
        layers: [[GenerateCoT, GenerateCoT, Calculator], [ScEnsemble]]
      stop: {type: agreement}

The scheduler, the stop policy, the trace and the endpoint work as with a trained controller:
operators that do not depend on each other still run at the same time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..registry import OPERATORS, register_controller
from .base import Controller, Selection


@register_controller(name="fixed")
class FixedController(Controller):
    """Args:
        layers: the picks, one list of operator names per layer, as a trained controller would make them.
    """

    name = "fixed"

    def __init__(self, layers: List[List[str]]):
        if not layers or not all(isinstance(layer, (list, tuple)) and layer for layer in layers):
            raise ValueError("a fixed controller needs layers: a list of non-empty lists of operator names")
        self.layers = [[str(name) for name in layer] for layer in layers]

    def select(self, query: str, operators: List[str], *, seed: Optional[int] = None,
               track_grad: bool = False) -> Selection:
        for layer in self.layers:
            for name in layer:
                OPERATORS.get(name)             # an unknown name fails here, with the registered ones listed
        return Selection(layers=[list(layer) for layer in self.layers])

    def describe(self) -> Dict[str, Any]:
        return {"type": self.name, "layers": self.layers}
