"""Tasks: prompts, answer extraction and scoring for a kind of problem."""

from ..registry import TASKS, register_task
from .base import Task, data_dir, read_jsonl
from .humaneval import HumanEval, Test
from .custom import CustomTask

__all__ = ["Task", "TASKS", "register_task", "data_dir", "read_jsonl", "HumanEval", "Test", "CustomTask"]
