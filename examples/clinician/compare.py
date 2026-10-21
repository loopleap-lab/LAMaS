"""With LAMaS and without, on the clinician team: the same agents, the same LLM and the same cases.

    python examples/clinician/compare.py                   # a simulated LLM, no model needed
    python examples/clinician/compare.py --backend http://localhost:8000/v1 --model qwen3.5-9b

--backend and --model take the place of those in clinician.yaml; an API key comes from
$LAMAS_API_KEY or $OPENAI_API_KEY (docs/tutorial.md, step 1).

Without LAMaS, the agents run one after another and all of them run, as in a plain multi-agent
loop. With LAMaS, the three clinicians run at the same time, and the attending runs only if they
disagree.
"""
import argparse
import json
import os
import random

import lamas
from lamas.config import load_config

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "clinician.yaml")


class SimulatedLLM(lamas.MockBackend):
    """No model needed: it gives a case's reference letter, and another letter one time in four.
    The same prompt always gets the same reply after the same time, with LAMaS or without."""

    def __init__(self):
        super().__init__(answer_fn=self.letter, error_rate=0)
        with open(os.path.join(HERE, "data", "medqa_sample.jsonl")) as f:
            self.cases = [json.loads(line) for line in f]

    def letter(self, prompt):
        reference = next(c["answer"] for c in self.cases if c["question"] in prompt)
        rng = random.Random(prompt)
        return reference if rng.random() < 0.75 else rng.choice([x for x in "ABCD" if x != reference])

    async def complete(self, prompt, **kwargs):
        self._rng = random.Random(str(prompt))       # the reply's length, and so its time, follow the prompt
        return await super().complete(prompt, **kwargs)


def llm(base_url=None, model=None):
    """The simulated LLM, or with --backend or --model the backend of clinician.yaml with them on top."""
    if not (base_url or model):
        return SimulatedLLM()
    backend = dict(load_config(CONFIG)["backend"])
    backend.update({key: value for key, value in (("base_url", base_url), ("model", model)) if value})
    return backend


def show(row):
    """One line per case as it finishes."""
    without, with_ = row["without"], row["with"]
    note = "attending pruned" if any(n.startswith("Attending") for n in with_["pruned"]) else "attending ran"
    print(f"case {row['index']}   without LAMaS {without['latency']:5.2f}s, {without['llm_calls']} calls"
          f"   with LAMaS {with_['latency']:5.2f}s, {with_['llm_calls']} calls   {note}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", help="base URL of an OpenAI-compatible endpoint, local or hosted")
    parser.add_argument("--model", help="the model name the endpoint expects")
    parser.add_argument("--limit", type=int, help="only the first N cases")
    args = parser.parse_args()
    backend = llm(args.backend, args.model)
    simulated = isinstance(backend, SimulatedLLM)
    concurrency = 10 if simulated else 1        # the simulated LLM has no load, so all cases can run at once

    print("Each case runs without LAMaS, then with LAMaS:\n")
    report = lamas.compare(CONFIG, backend=backend, limit=args.limit, concurrency=concurrency, on_problem=show)
    print()
    print(report.table())
    first = report.rows[0]
    for side in ("without", "with"):
        print(f"\ncase 0, {side} LAMaS")
        print(first[side]["trace"].gantt(width=40))
    if simulated:
        print("\nOn your own model, set up as in docs/tutorial.md, step 1:")
        print("  python examples/clinician/compare.py --backend http://localhost:8000/v1 --model qwen3.5-9b")


if __name__ == "__main__":
    main()
