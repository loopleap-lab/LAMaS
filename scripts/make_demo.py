#!/usr/bin/env python
"""Turn an evaluation run into the recorded runs of the demo page.

    lamas eval configs/humaneval-gpt4omini.yaml --out runs/humaneval
    python scripts/make_demo.py runs/humaneval

Picks a few solved problems with an interesting workflow (several operators, some pruned by the
stop gate) and writes ``src/lamas/serve/demo_samples.json`` plus one recording per sample. A
recording is the event stream of the real run, so the page can replay it without a serving engine.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVE = os.path.normpath(os.path.join(HERE, "..", "src", "lamas", "serve"))
DATA = os.path.normpath(os.path.join(HERE, "..", "data", "humaneval_test.jsonl"))
SAMPLES = 6


def events(trace: dict, correct: bool) -> list:
    """The executor's event stream, rebuilt from a saved trace."""
    timeline = []
    ends = {s["id"]: s["end"] for s in trace["spans"] if s["end"] is not None}
    for s in trace["spans"]:
        if s["start"] is not None and s["end"] is not None:
            start = {"type": "start", "id": s["id"], "t": s["start"]}
            if s["id"] not in {n["id"] for n in trace["plan"]["nodes"]}:
                start["deps"] = s["deps"]                       # the final step is not part of the plan
            timeline.append((s["start"], 1, start))
            timeline.append((s["end"], 0, {"type": "end", "id": s["id"], "t": s["end"], "status": s["status"],
                                           "tokens": s["completion_tokens"], "calls": s["llm_calls"]}))
        elif s["status"] == "pruned":
            t = max([trace["stop_time"] or 0.0] + [ends.get(d, 0.0) for d in s["deps"]])
            timeline.append((t, 3, {"type": "prune", "id": s["id"], "t": round(t, 3)}))
    if trace["stop_time"] is not None:
        timeline.append((trace["stop_time"], 2, {"type": "stop", "reason": trace["stop_reason"], "t": trace["stop_time"]}))
    out = [{"type": "plan", "nodes": trace["plan"]["nodes"], "layers": trace["plan"]["layers"], "t": 0.0}]
    out += [e for _, _, e in sorted(timeline, key=lambda x: (x[0], x[1]))]
    out.append({"type": "final", "t": trace["latency"], "answer": trace["answer"], "latency": trace["latency"],
                "cost": trace["cost"], "usage": trace["usage"], "critical_path": trace["critical_path"],
                "pruned": trace["pruned"], "stop_reason": trace["stop_reason"], "correct": correct})
    return out


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run = sys.argv[1]
    with open(DATA, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(run, "report.json"), "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(os.path.join(run, "traces.jsonl"), "r", encoding="utf-8") as f:
        traces = {row["index"]: row for row in map(json.loads, f)}

    def interest(row: dict) -> tuple:
        trace = row["trace"]
        ran = sum(1 for s in trace["spans"] if s["status"] == "done")
        return (len(trace["pruned"]) > 0, min(ran, 5), -abs(trace["latency"] - 12.0))

    solved = [r for r in traces.values() if r["score"] == 1.0 and len(items[r["index"]]["prompt"]) <= 700
              and 5.0 <= r["trace"]["latency"] <= 25.0]
    with_gate = sorted((r for r in solved if r["trace"]["pruned"]), key=interest, reverse=True)[:SAMPLES - 2]
    without = sorted((r for r in solved if not r["trace"]["pruned"] and r not in with_gate), key=interest, reverse=True)[:2]
    picked = sorted(with_gate + without, key=lambda r: r["index"])

    os.makedirs(os.path.join(SERVE, "recordings"), exist_ok=True)
    for name in os.listdir(os.path.join(SERVE, "recordings")):
        os.remove(os.path.join(SERVE, "recordings", name))
    samples = []
    for row in picked:
        item = items[row["index"]]
        samples.append({"id": item["task_id"], "entry_point": item["entry_point"], "prompt": item["prompt"],
                        "test": item["test"], "solution": item["canonical_solution"],
                        "seed": report["config"]["seed"] + row["index"]})
        recording = {"sample": item["task_id"], "backend": row["trace"]["backend"], "recorded_at": report.get("date", ""),
                     "events": events(row["trace"], True)}
        with open(os.path.join(SERVE, "recordings", item["task_id"].replace("/", "_") + ".json"), "w", encoding="utf-8") as f:
            json.dump(recording, f, ensure_ascii=False)
        print(f"{item['task_id']:<14} {item['entry_point']:<24} {row['trace']['latency']:5.1f}s  "
              f"{len(row['trace']['spans'])} operators, {len(row['trace']['pruned'])} pruned")
    with open(os.path.join(SERVE, "demo_samples.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
