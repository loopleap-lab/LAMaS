"""Controller: picks the operators for a query, layer by layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..registry import CONTROLLERS


@dataclass
class Selection:
    """What a controller decided for one query."""

    layers: List[List[str]]
    # training only: one log-probability per pick, same shape as `layers`
    log_probs: List[List[Any]] = field(default_factory=list)
    # for inspection: operator -> probability, one dict per layer
    probs: List[Dict[str, float]] = field(default_factory=list)


class Controller:
    """The interface. Subclass it, implement :meth:`select`, and register it with ``@register_controller``."""

    name = "controller"

    def select(self, query: str, operators: List[str], *, seed: Optional[int] = None,
               track_grad: bool = False) -> Selection:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"type": self.name}


def make_controller(spec: Any, base_dir: Optional[str] = None) -> Controller:
    """Build a controller from a config dict or an instance.

    ``{"weights": "controller.pth", ...}`` is the controller of the paper; ``{"type": <name>, ...}``
    is any registered controller.
    """
    import os

    if isinstance(spec, Controller):
        return spec
    cfg = dict(spec or {})
    kind = cfg.pop("type", "supernet")
    if cfg.get("weights") and base_dir and not os.path.isabs(cfg["weights"]):
        cfg["weights"] = os.path.join(base_dir, cfg["weights"])
    if kind == "supernet":
        from . import supernet  # noqa: F401  (registers itself; needs torch)
    return CONTROLLERS.get(kind)(**cfg)
