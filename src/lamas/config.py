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
    unknown = [k for k in cfg if k not in RUNNER_KEYS + ("train",)]
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}. Known: {list(RUNNER_KEYS)}")
    cfg["_dir"] = os.path.dirname(os.path.abspath(path))
    return cfg


def runner_from_config(cfg: Dict[str, Any], **overrides: Any) -> Runner:
    """Build a :class:`Runner` from a loaded config; keyword arguments win over the file."""
    kwargs = {k: cfg[k] for k in RUNNER_KEYS if k in cfg}
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return Runner(**kwargs)
