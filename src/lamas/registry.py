"""Name-based registries: the single extension point for operators, controllers, stop
policies, schedulers, backends and tasks.

Third-party packages can register without importing LAMaS internals by declaring an entry
point, e.g. in their ``pyproject.toml``::

    [project.entry-points."lamas.operators"]
    debate = "my_pkg.debate:Debate"
"""

from __future__ import annotations

from importlib import metadata
from typing import Any, Callable, Dict, Generic, Iterator, Optional, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str, entry_point_group: Optional[str] = None):
        self.kind = kind
        self._items: Dict[str, T] = {}
        self._group = entry_point_group
        self._loaded_entry_points = False

    def register(self, obj: Optional[T] = None, *, name: Optional[str] = None) -> Any:
        """Use as ``@registry.register`` or ``@registry.register(name="x")``."""

        def _add(o: T) -> T:
            key = name or getattr(o, "name", None) or getattr(o, "__name__", None)
            if not key:
                raise ValueError(f"cannot register a {self.kind} without a name")
            self._items[str(key)] = o
            return o

        return _add if obj is None else _add(obj)

    def _load_entry_points(self) -> None:
        if self._loaded_entry_points or not self._group:
            return
        self._loaded_entry_points = True
        try:
            eps = metadata.entry_points(group=self._group)
        except Exception:  # pragma: no cover - very old importlib.metadata
            return
        for ep in eps:
            if ep.name not in self._items:
                try:
                    self._items[ep.name] = ep.load()
                except Exception as exc:  # pragma: no cover - a broken plugin must not break us
                    import warnings

                    warnings.warn(f"could not load {self.kind} plugin {ep.name!r}: {exc}")

    def get(self, name: str) -> T:
        if name not in self._items:
            self._load_entry_points()
        if name not in self._items:
            known = ", ".join(sorted(self._items)) or "none"
            raise KeyError(f"unknown {self.kind} {name!r}. Registered: {known}")
        return self._items[name]

    def __contains__(self, name: str) -> bool:
        if name not in self._items:
            self._load_entry_points()
        return name in self._items

    def names(self) -> list:
        self._load_entry_points()
        return sorted(self._items)

    def items(self) -> Iterator:
        self._load_entry_points()
        return iter(self._items.items())


OPERATORS: Registry = Registry("operator", "lamas.operators")
CONTROLLERS: Registry = Registry("controller", "lamas.controllers")
STOP_POLICIES: Registry = Registry("stop policy", "lamas.stop_policies")
BACKENDS: Registry = Registry("backend", "lamas.backends")
TASKS: Registry = Registry("task", "lamas.tasks")

register_operator: Callable = OPERATORS.register
register_controller: Callable = CONTROLLERS.register
register_stop_policy: Callable = STOP_POLICIES.register
register_backend: Callable = BACKENDS.register
register_task: Callable = TASKS.register
