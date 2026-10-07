"""Policy: a trained controller, its stop gate and an operator pool, saved together as one folder.

A policy folder looks like this::

    policies/humaneval/
      policy.yaml       # task, operators, controller and stop-gate settings, provenance
      controller.pth    # weights of the controller
      gate.pth          # weights of the stop gate

``load_policy`` accepts a folder path or a name such as ``lamas/humaneval``, which is searched in
``$LAMAS_POLICIES``, the repository's ``policies/`` and ``~/.cache/lamas/policies``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .controllers.base import Controller, make_controller
from .stop import StopPolicy, make_stop

DEFAULT_POLICY = "lamas/humaneval"


@dataclass
class Policy:
    name: str
    controller: Controller
    stop: StopPolicy
    operators: List[str]
    task: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "task": self.task, "operators": self.operators,
                "controller": self.controller.describe(), "stop": self.stop.describe()}


def search_path() -> List[str]:
    dirs = [d for d in os.environ.get("LAMAS_POLICIES", "").split(os.pathsep) if d]
    here = os.path.dirname(os.path.abspath(__file__))
    dirs.append(os.path.normpath(os.path.join(here, "..", "..", "policies")))
    dirs.append(os.path.join(os.path.expanduser("~"), ".cache", "lamas", "policies"))
    return dirs


def find_policy_dir(name: str) -> Optional[str]:
    """Resolve ``lamas/humaneval``, ``humaneval`` or a path to a folder containing ``policy.yaml``."""
    if os.path.isfile(os.path.join(name, "policy.yaml")):
        return name
    short = name.split("/", 1)[1] if name.startswith("lamas/") else name
    for root in search_path():
        candidate = os.path.join(root, short)
        if os.path.isfile(os.path.join(candidate, "policy.yaml")):
            return candidate
    return None


def available() -> List[str]:
    found = []
    for root in search_path():
        if os.path.isdir(root):
            for entry in sorted(os.listdir(root)):
                if os.path.isfile(os.path.join(root, entry, "policy.yaml")) and f"lamas/{entry}" not in found:
                    found.append(f"lamas/{entry}")
    return found


def load_policy(spec: Any = None) -> Policy:
    """A policy from a name, a folder, a config dict or an instance. ``None`` is the released policy."""
    if isinstance(spec, Policy):
        return spec
    if isinstance(spec, dict):
        return _from_config(dict(spec), base_dir=None, default_name=spec.get("name", "custom"))
    name = spec or DEFAULT_POLICY
    folder = find_policy_dir(name)
    if folder is None:
        raise FileNotFoundError(f"policy {name!r} not found. Installed: {available() or 'none'}. Searched: {search_path()}")
    with open(os.path.join(folder, "policy.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _from_config(cfg, base_dir=folder, default_name=f"lamas/{os.path.basename(folder.rstrip(os.sep))}")


def _from_config(cfg: Dict[str, Any], base_dir: Optional[str], default_name: str) -> Policy:
    controller_cfg = dict(cfg.get("controller") or {})
    if "operator_texts" in cfg:
        controller_cfg["operator_texts"] = cfg["operator_texts"]
    return Policy(
        name=cfg.get("name", default_name), controller=make_controller(controller_cfg, base_dir=base_dir),
        stop=make_stop(cfg.get("stop"), base_dir=base_dir), operators=list(cfg.get("operators") or []),
        task=cfg.get("task"), meta={k: cfg[k] for k in ("source", "training") if k in cfg},
    )


def save_policy(folder: str, *, name: str, task: str, operators: List[str], controller_state: Any,
                controller_cfg: Dict[str, Any], gate_state: Optional[Any] = None, tau: Optional[float] = None,
                operator_texts: Optional[Dict[str, str]] = None, extra: Optional[Dict[str, Any]] = None) -> str:
    """Write a policy folder. Returns its path."""
    import torch

    os.makedirs(folder, exist_ok=True)
    torch.save(controller_state, os.path.join(folder, "controller.pth"))
    cfg: Dict[str, Any] = {"name": name, "task": task, "operators": list(operators),
                           "controller": dict(controller_cfg, weights="controller.pth")}
    if gate_state is not None:
        torch.save({"gate_state_dict": gate_state, "tau": float(tau if tau is not None else 0.1)},
                   os.path.join(folder, "gate.pth"))
        cfg["stop"] = {"weights": "gate.pth"}
    if operator_texts:
        cfg["operator_texts"] = dict(operator_texts)
    cfg.update(extra or {})
    with open(os.path.join(folder, "policy.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, width=1000)
    return folder
