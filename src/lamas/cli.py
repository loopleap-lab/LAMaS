"""Command line: ``lamas run | plan | serve | demo | eval | train | list``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--policy", help="lamas/<name> or a policy folder (default: lamas/humaneval)")
    p.add_argument("--backend", help="base URL of an OpenAI-compatible serving engine, or 'mock' for a simulated one")
    p.add_argument("--model", help="model name the serving engine expects")
    p.add_argument("--budget", type=float, metavar="SECONDS", help="latency budget per query")


def _settings(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The config file, with the command line on top of it."""
    out: Dict[str, Any] = {k: cfg[k] for k in ("policy", "task", "operators", "budget", "seed") if k in cfg}
    backend = cfg.get("backend")
    if getattr(args, "backend", None):
        backend = "mock" if args.backend == "mock" else dict(backend if isinstance(backend, dict) else {}, base_url=args.backend)
    if getattr(args, "model", None):
        backend = dict(backend) if isinstance(backend, dict) else ({} if backend in (None, "mock") else {"base_url": backend})
        backend["model"] = args.model
    if backend is not None:
        out["backend"] = backend
    if getattr(args, "policy", None):
        out["policy"] = args.policy
    if getattr(args, "budget", None):
        out["budget"] = {"latency_s": args.budget}
    return out


def _split_config(positionals: List[str]) -> tuple:
    """`lamas run run.yaml "problem"` or `lamas run "problem"`."""
    if positionals and positionals[0].endswith((".yaml", ".yml")) and os.path.exists(positionals[0]):
        return positionals[0], positionals[1:]
    return None, positionals


def _load(args: argparse.Namespace) -> tuple:
    from .config import load_config

    config_path, rest = _split_config(getattr(args, "args", []) or [])
    cfg = load_config(config_path)
    return cfg, _settings(args, cfg), rest


def _text(rest: List[str]) -> str:
    text = " ".join(rest)
    return text.encode("utf-8").decode("unicode_escape") if "\\n" in text else text     # "def f():\n ..." typed in a shell


# ---- commands --------------------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    from .api import Runner

    _, settings, rest = _load(args)
    if not rest:
        print("usage: lamas run [config.yaml] \"problem\"", file=sys.stderr)
        return 2
    runner = Runner(**settings)

    async def go():
        try:
            return await runner.arun(_text(rest))
        finally:
            await runner.aclose()

    trace = asyncio.run(go()).trace
    print(trace.answer.strip())
    print()
    print(trace.gantt())
    print(f"policy {runner.policy.name} · backend {trace.backend}")
    if trace.critical_path:
        print("critical path: " + " -> ".join(trace.critical_path))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from .api import plan
    from .scheduler import waves

    _, settings, rest = _load(args)
    p = plan(_text(rest), policy=settings.get("policy"), task=settings.get("task"), operators=settings.get("operators"),
             seed=settings.get("seed"))
    print("controller picks: " + "  |  ".join(", ".join(layer) for layer in p.layers))
    for i, wave in enumerate(waves(p.nodes)):
        print(f"wave {i}: " + ", ".join(f"{n} <- [{', '.join(p.nodes[n].deps)}]" if p.nodes[n].deps else n for n in wave))
    print(f"({p.controller_ms:.0f} ms)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import Runner
    from .serve import create_app

    _, settings, _ = _load(args)
    runner = Runner(**settings)
    app = create_app(runner, demo=_demo(runner) if args.demo else None)
    print(f"LAMaS serving {runner.policy.name} on http://{args.host}:{args.port}/v1  (backend {runner.backend.describe()})")
    if args.demo:
        print(f"demo: http://{args.host}:{args.port}/demo")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _demo(runner: Any):
    from .serve import Demo, demo_samples

    return Demo(runner=runner, samples=demo_samples(), source="simulated" if runner.backend.name == "mock" else "live")


def cmd_demo(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import Runner
    from .backends import MockBackend
    from .serve import create_app, demo_samples

    _, settings, _ = _load(args)
    if settings.get("backend") == "mock":
        # the simulated LLM "knows" the reference solution of every sample problem
        answers = {s["prompt"][:120]: s["prompt"] + s["solution"] for s in demo_samples() if s.get("solution")}
        settings["backend"] = MockBackend(answers=answers, seed=None)
    runner = Runner(**settings)
    app = create_app(runner, demo=_demo(runner))
    print(f"LAMaS demo on http://{args.host}:{args.port}/demo  (backend {runner.backend.describe()})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluation import evaluate

    _, settings, _ = _load(args)
    seen = {"ok": 0.0, "latency": 0.0, "errors": 0}
    live = sys.stderr.isatty()

    def progress(done: int, total: int, row: Dict[str, Any]) -> None:
        mark = "ok " if row.get("score") else ("ERR" if "error" in row else "x  ")
        seen["ok"] += row.get("score") or 0.0
        seen["latency"] += row.get("latency", 0.0)
        seen["errors"] += "error" in row
        line = (f"[{done}/{total}] {mark} {row.get('latency', 0):6.1f}s   accuracy {seen['ok'] / done * 100:5.1f}%   "
                f"mean latency {seen['latency'] / max(done - seen['errors'], 1):5.1f}s")
        if "error" in row:
            line += f"   {row['error'][:160]}"
        # a log file gets one line per query; a terminal gets one line that updates
        print(("\r" + line) if live else line, end="" if live else "\n", file=sys.stderr, flush=True)

    report = evaluate(settings.get("task"), settings.get("policy"), settings.get("backend"), limit=args.limit,
                      concurrency=args.concurrency, seed=settings.get("seed", 0), out=args.out,
                      budget=settings.get("budget"), on_progress=progress)
    if live:
        print(file=sys.stderr)
    print(report.table())
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .policy import load_policy
    from .training import train

    cfg, settings, _ = _load(args)
    task = settings.get("task") or load_policy(settings.get("policy")).task

    def progress(event: Dict[str, Any]) -> None:
        if event["phase"] == "controller":
            print(f"step {event['step']:4d}  acc {event['accuracy']:.2f}  latency {event['latency']:6.1f}s  "
                  f"lambda_S {event['lambda_acc']:.3f}  loss {event['loss']:+.3f}", flush=True)
        else:
            print(json.dumps(event), flush=True)

    policy = train(task, settings.get("backend"), out=args.out, operators=settings.get("operators"),
                   seed=settings.get("seed", 0), on_progress=progress, **(cfg.get("train") or {}))
    print(f"saved {policy.name}: {json.dumps(policy.describe())}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    from . import operators as _ops  # noqa: F401
    from . import tasks as _tasks  # noqa: F401
    from .policy import available
    from .registry import OPERATORS, TASKS

    print("operators")
    for name, cls in sorted(OPERATORS.items()):
        print(f"  {name:18s} needs {cls.needs:9s} {cls.description[:80]}")
    print("tasks      " + ", ".join(TASKS.names()))
    print("policies   " + (", ".join(available()) or "none installed"))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="lamas", description="Latency-aware orchestration for multi-agent LLM systems.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="solve one problem and show what ran")
    p.add_argument("args", nargs="+", metavar="[config.yaml] problem")
    _common(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("plan", help="show the workflow the controller and the scheduler would build")
    p.add_argument("args", nargs="+", metavar="[config.yaml] problem")
    p.add_argument("--policy", help="lamas/<name> or a policy folder (default: lamas/humaneval)")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("serve", help="start the OpenAI-compatible endpoint")
    p.add_argument("args", nargs="*", metavar="config.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--demo", action="store_true", help="also serve the demo page at /demo")
    _common(p)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("demo", help="start the demo page")
    p.add_argument("args", nargs="*", metavar="config.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    _common(p)
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("eval", help="evaluate a policy on its task: accuracy, latency, cost")
    p.add_argument("args", nargs="*", metavar="config.yaml")
    p.add_argument("--limit", type=int, help="only the first N problems")
    p.add_argument("--concurrency", type=int, default=3, help="problems in flight at once")
    p.add_argument("--out", help="folder for report.json and traces.jsonl")
    _common(p)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("train", help="fit a controller and a stop gate to a task (settings: the config's train section)")
    p.add_argument("args", nargs="*", metavar="config.yaml")
    p.add_argument("--out", help="policy folder to write")
    _common(p)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("list", help="list operators, tasks and installed policies")
    p.set_defaults(fn=cmd_list)

    args = parser.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
