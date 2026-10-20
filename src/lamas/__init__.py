"""LAMaS: latency-aware orchestration for multi-agent LLM systems.

    import lamas
    result = lamas.run(problem, backend="http://localhost:8000/v1")
    print(result.answer)
    print(result.trace.gantt())
"""

from . import operators as _operators  # noqa: F401  (registers the built-in operators)
from . import tasks as _tasks  # noqa: F401  (registers the built-in tasks)
from .api import Runner, aexecute, arun, execute, get_task, plan, run
from .backends import Backend, MockBackend, OpenAICompatBackend, make_backend
from .controllers import Controller, Selection
from .operators import Context, Operator
from .operators.function import operator
from .plugins import load_plugins
from .policy import Policy, load_policy
from .registry import (OPERATORS, TASKS, register_backend, register_controller, register_operator,
                       register_stop_policy, register_task)
from .stop import StopPolicy, StopState
from . import stop_rules  # noqa: F401  (registers the agreement stop policy)
from .tasks import Task
from .types import Node, Plan, Result, Span, Trace, Usage

__version__ = "0.1.0"

__all__ = [
    "run", "arun", "plan", "execute", "aexecute", "Runner", "train", "evaluate",
    "Operator", "operator", "load_plugins", "Context", "Controller", "Selection", "StopPolicy", "StopState",
    "Task", "Policy", "load_policy", "get_task", "Backend", "MockBackend", "OpenAICompatBackend", "make_backend",
    "register_operator", "register_controller", "register_stop_policy", "register_backend", "register_task",
    "OPERATORS", "TASKS", "Plan", "Node", "Span", "Trace", "Result", "Usage", "__version__",
]


def train(*args, **kwargs):
    """Fit a controller and a stop gate to a task. See :func:`lamas.training.train`."""
    from .training import train as _train

    return _train(*args, **kwargs)


def evaluate(*args, **kwargs):
    """Run a task's test split and report accuracy, latency and cost. See :func:`lamas.evaluation.evaluate`."""
    from .evaluation import evaluate as _evaluate

    return _evaluate(*args, **kwargs)
