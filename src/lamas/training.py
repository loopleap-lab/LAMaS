"""Training: fit a controller (phase 1) and a stop gate (phase 2) to a task.

Phase 1 trains the controller by policy gradient with the Lagrangian reward and critical-path
credit (Sections 3.3 to 3.5 of the paper). Phase 2 freezes the controller, collects full
traces on the training split and fits the gate to the control advantage ``R_k - R_N`` by
regression (Section 3.6).
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import statistics
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import answers
from .api import Runner, _sync, get_task
from .backends import Backend, make_backend
from .objective import Objective
from .policy import Policy, load_policy, save_policy
from .stop import StopPolicy, build_gate_net
from .types import NEEDS_SOLUTIONS, Trace

Progress = Optional[Callable[[Dict[str, Any]], None]]


async def atrain(task: Any, backend: Any = None, *, operators: Optional[List[str]] = None,
                 out: Optional[str] = None, name: Optional[str] = None, init: Any = None,
                 layers: int = 4, threshold: float = 0.3, latency_weight: float = 0.005, cost_weight: float = 3.0,
                 accuracy_floor: Optional[float] = None, dual_lr: float = 0.1,
                 repeats: int = 4, batch_size: int = 4, lr: float = 0.01, limit: Optional[int] = None,
                 gate_epochs: int = 100, gate_lr: float = 0.01,
                 seed: int = 0, on_progress: Progress = None) -> Policy:
    """Train a policy for ``task`` and save it to ``out``.

    Args:
        task: task name or instance; needs a ``train`` split and a scorer.
        backend: where the operators' LLM calls go.
        operators: the operator pool; defaults to the task's.
        out: policy folder to write (default ``policies/<task>-trained``).
        init: a policy to start from (its controller weights are reused).
        accuracy_floor: ``S0``. When omitted it is measured: the accuracy of the starting
            controller on the training split, before any latency pressure.
        repeats, batch_size, lr: passes over the training split, queries per update, Adam step size.
        gate_epochs, gate_lr: how long and how fast the stop gate is fitted.
    """
    import torch

    from .controllers.supernet import SupernetController

    task_obj = get_task(task)
    be: Backend = make_backend(backend)
    owns_backend = not isinstance(backend, Backend)
    operators = list(operators or task_obj.operators)
    items = task_obj.load("train", limit)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    notify = on_progress or (lambda event: None)

    controller = SupernetController(layers=layers, threshold=threshold)
    if init is not None:
        source = load_policy(init).controller
        controller.net.load_state_dict(source.net.state_dict())
        controller.operator_texts = dict(getattr(source, "operator_texts", {}))
    controller.net.train()
    policy = Policy(name=name or f"lamas/{task_obj.name}-trained", controller=controller, stop=StopPolicy(),
                    operators=operators, task=task_obj.name)
    runner = Runner(policy=policy, backend=be, task=task_obj)
    optimizer = torch.optim.Adam(controller.net.parameters(), lr=lr)

    async def run_batch(batch: List[dict], track_grad: bool) -> List[Tuple[Any, Optional[Trace], float]]:
        plans = [runner.plan(task_obj.problem(item), seed=rng.randrange(1 << 30), track_grad=track_grad) for item in batch]

        async def one(plan, item):
            try:
                trace = await runner.aexecute(plan, item=item)
                return plan, trace, float(await task_obj.ascore(item, trace.answer))
            except Exception:
                return plan, None, 0.0

        return list(await asyncio.gather(*(one(p, i) for p, i in zip(plans, batch))))

    try:
        # ---- the accuracy floor S0 -----------------------------------------------------------
        if accuracy_floor is None:
            sample = items[: min(len(items), 32)]
            scores: List[float] = []
            for start in range(0, len(sample), batch_size):
                scores += [s for _, _, s in await run_batch(sample[start:start + batch_size], track_grad=False)]
            accuracy_floor = max(statistics.fmean(scores) if scores else 1.0, 0.05)
            notify({"phase": "floor", "accuracy_floor": accuracy_floor, "n": len(sample)})
        objective = Objective(latency_weight=latency_weight, cost_weight=cost_weight, accuracy_floor=accuracy_floor,
                              dual_lr=dual_lr)

        # ---- phase 1: the controller ---------------------------------------------------------
        ema_mean, ema_sq, momentum, step = None, None, 0.95, 0
        for rep in range(repeats):
            order = list(items)
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                results = await run_batch(order[start:start + batch_size], track_grad=True)
                log_probs, rewards, lat, acc = [], [], [], []
                for plan, trace, score in results:
                    if trace is None:
                        continue
                    acc.append(score)
                    lat.append(trace.latency)
                    node_rewards = objective.node_rewards(trace, score)
                    for node_id, reward in node_rewards.items():
                        if node_id in plan.log_probs:
                            log_probs.append(plan.log_probs[node_id])
                            rewards.append(reward)
                    full = objective.reward(score, trace.latency, trace.cost)
                    for lp in plan.meta.get("control_log_probs", []):      # EarlyStop picks
                        log_probs.append(lp)
                        rewards.append(full)
                if not log_probs:
                    continue
                r = torch.tensor(rewards, dtype=torch.float32)
                mean, sq = float(r.mean()), float((r ** 2).mean())
                ema_mean = mean if ema_mean is None else momentum * ema_mean + (1 - momentum) * mean
                ema_sq = sq if ema_sq is None else momentum * ema_sq + (1 - momentum) * sq
                r = (r - ema_mean) / math.sqrt(max(ema_sq - ema_mean ** 2, 1e-8))
                loss = -(torch.stack(log_probs) * r).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_acc = statistics.fmean(acc) if acc else 0.0
                lam = objective.update_dual(batch_acc)
                step += 1
                notify({"phase": "controller", "step": step, "repeat": rep + 1, "loss": float(loss.detach()),
                        "accuracy": batch_acc, "latency": statistics.fmean(lat) if lat else 0.0, "lambda_acc": lam})
        controller.net.eval()

        # ---- phase 2: the gate ---------------------------------------------------------------
        gate_state, tau, gate_stats = None, None, {}
        samples: List[Tuple[List[float], Any, float]] = []
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            for (plan, trace, _), item in zip(await run_batch(batch, track_grad=False), batch):
                if trace is not None:
                    samples += await gate_samples(trace, item, task_obj, be, objective)
            notify({"phase": "traces", "done": min(start + batch_size, len(items)), "total": len(items),
                    "samples": len(samples)})
        if samples:
            net, tau, gate_stats = fit_gate(samples, epochs=gate_epochs, lr=gate_lr, seed=seed)
            gate_state = net.state_dict()
            notify({"phase": "gate", **gate_stats})
    finally:
        if owns_backend:
            await be.aclose()

    folder = out or os.path.join("policies", f"{task_obj.name}-trained")
    save_policy(folder, name=policy.name, task=task_obj.name, operators=operators,
                controller_state=controller.net.state_dict(),
                controller_cfg={"layers": layers, "hidden": 32, "threshold": threshold},
                gate_state=gate_state, tau=tau, operator_texts=controller.operator_texts or None,
                extra={"training": dict(objective.to_dict(), llm=be.describe(), train_items=len(items), repeats=repeats,
                                        batch_size=batch_size, gate=gate_stats)})
    return load_policy(folder)


def train(task: Any, backend: Any = None, **kwargs: Any) -> Policy:
    """Blocking form of :func:`atrain`."""
    return _sync(atrain(task, backend, **kwargs))


# ---- phase 2 helpers ----------------------------------------------------------------------------
async def gate_samples(trace: Trace, item: dict, task: Any, backend: Backend,
                       objective: Objective) -> List[Tuple[List[float], Any, float]]:
    """Turn one full trace into ``(state features, query embedding, delta_k)`` training samples.

    At every completion ``k`` we ask: had we stopped here, what would the reward have been?
    ``delta_k = R_k - R_N`` compares that with the reward of running everything (Eq. 10).
    """
    from .encoder import embed_one

    spans = sorted((s for s in trace.spans.values() if s.status == "done" and s.id in trace.plan.nodes),
                   key=lambda s: s.end or 0.0)
    if len(spans) < 2:
        return []
    n_total = len(trace.plan.nodes)
    table: List[str] = []           # the solutions on the table after each completion
    points = []
    cost = 0.0
    score_cache: Dict[str, float] = {}
    for k, span in enumerate(spans, start=1):
        if trace.plan.nodes[span.id].needs == NEEDS_SOLUTIONS and span.outputs:
            table = []
        table += span.outputs
        cost += (span.prompt_tokens * backend.price[0] + span.completion_tokens * backend.price[1]) / 1e6
        if not table:
            continue
        keys = [(await task.akey(s, item)) or "h:" + answers.text_hash(s) for s in table]
        best = Counter(keys).most_common(1)[0]
        pick = next(s for s, key in zip(reversed(table), reversed(keys)) if key == best[0])
        if pick not in score_cache:
            score_cache[pick] = float(await task.ascore(item, pick))
        points.append({"k": k, "consensus": best[1] / len(keys), "score": score_cache[pick],
                       "time": span.end or 0.0, "cost": cost, "pending": n_total - k})
    if not points:
        return []
    last = points[-1]
    r_full = objective.reward(last["score"], last["time"], last["cost"])
    query = embed_one(trace.query)
    out = []
    for p in points:
        if p["pending"] == 0:
            continue
        delta = objective.reward(p["score"], p["time"], p["cost"]) - r_full
        out.append(([p["k"] / max(n_total, 1), p["consensus"]], query, delta))
    return out


def fit_gate(samples: List[Tuple[List[float], Any, float]], epochs: int = 100, lr: float = 0.01,
             hidden: int = 32, seed: int = 0) -> Tuple[Any, float, Dict[str, Any]]:
    """Fit the gate network to the deltas by mean squared error. Returns ``(net, tau, stats)``."""
    import torch

    torch.manual_seed(seed)
    states = torch.tensor([s for s, _, _ in samples], dtype=torch.float32)
    queries = torch.stack([q for _, q, _ in samples]).float()
    deltas = torch.tensor([d for _, _, d in samples], dtype=torch.float32)
    net = build_gate_net(hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(net(states, queries), deltas)
        loss.backward()
        optimizer.step()
    net.eval()
    with torch.no_grad():
        predicted = net(states, queries)
    positive = deltas[deltas > 0]
    tau = float(positive.std()) if positive.numel() > 1 and float(positive.std()) > 1e-6 else 0.1
    sign_acc = float(((predicted > 0) == (deltas > 0)).float().mean())
    return net, tau, {"samples": len(samples), "mse": float(loss.detach()), "tau": tau, "sign_accuracy": sign_acc,
                      "positive_share": float((deltas > 0).float().mean())}
