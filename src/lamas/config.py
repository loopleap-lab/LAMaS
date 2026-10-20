"""One config, one run: a YAML file that names the serving engine and the policy.

    backend:                                # the serving engine
      base_url: http://localhost:8000/v1    # any OpenAI-compatible endpoint
      model:    qwen3.5-9b
    policy:     lamas/humaneval             # a trained controller + stop gate
    budget:     {latency_s: 15}             # optional latency deadline

Only the backend is required; everything else has a default.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

from .api import Runner

RUNNER_KEYS = ("policy", "backend", "task", "budget", "operators", "seed")


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Read a YAML config. ``None`` gives an empty config (all defaults)."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: the config must be a mapping")
    unknown = [k for k in cfg if k not in RUNNER_KEYS + ("train", "plugins")]
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}. Known: {list(RUNNER_KEYS)}")
    cfg["_dir"] = os.path.dirname(os.path.abspath(path))
    if cfg.get("plugins"):              # your own operators, tasks, ... from files (lamas.plugins)
        from .plugins import load_plugins

        load_plugins(cfg["plugins"], base_dir=cfg["_dir"])
    task = cfg.get("task")
    if isinstance(task, dict) and isinstance(task.get("data"), dict):     # data files: relative to the config
        task["data"] = {k: (v if os.path.isabs(str(v)) else os.path.join(cfg["_dir"], str(v))) for k, v in task["data"].items()}
    return cfg


def runner_from_config(cfg: Dict[str, Any], **overrides: Any) -> Runner:
    """Build a :class:`Runner` from a loaded config; keyword arguments win over the file."""
    kwargs = {k: cfg[k] for k in RUNNER_KEYS if k in cfg}
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return Runner(**kwargs)
