# lamas/humaneval

The trained LAMaS policy for HumanEval-style code generation: the controller and the stop gate
behind the HumanEval results of the paper
([*Learning Latency-Aware Orchestration for Multi-Agent Systems*](https://arxiv.org/abs/2601.10560)).

A policy is a trained controller plus its stop gate. The controller picks the operators for each
problem. The stop gate decides, after each operator, whether the rest is still worth waiting for.

## Files

| File | What it is |
|---|---|
| `policy.yaml` | operators, controller and stop-gate settings, the text each operator was embedded with |
| `controller.pth` | controller weights (135,424 parameters, 4 layers, selection threshold 0.3) |
| `gate.pth` | stop-gate weights (1,797 parameters) and its temperature `tau` = 0.13 |
| `SHA256SUMS` | checksums of the three files above |

## Use it

```bash
export OPENAI_API_KEY=...                       # or any OpenAI-compatible serving engine, see the config
lamas plan 'def add(a, b):\n    """Return a + b."""'                 # the workflow, without running it
lamas eval configs/humaneval-gpt4omini.yaml --out runs/humaneval     # the whole test split
```

```python
import lamas

runner = lamas.Runner(policy="lamas/humaneval", backend={"model": "gpt-4o-mini"})
result = runner.run(problem)          # problem = the function signature and docstring
print(result.answer)                  # the function
print(result.trace.gantt())           # which operators ran, the critical path, what the gate pruned
```

## Results

HumanEval test split, 131 problems, gpt-4o-mini (paper, Table 1).

| | Accuracy | Latency per problem |
|---|---|---|
| Single LLM call | 84.73% | 10.23 s |
| MaAS | 93.38% | 47.32 s |
| **LAMaS** | **95.42%** | **21.00 s** |

## The workflow

The `humaneval` task of the package runs the workflow the controller and the stop gate were trained with:

- **Operators**: `Generate`, `GenerateCoT`, `MultiGenerateCoT`, `ScEnsemble`, `Test`, `SelfRefine`, `EarlyStop`.
- **Scheduler**: operators that need only the problem start at once; the others wait for their inputs.
- **Final step**: the answer is chosen with the public test cases, repaired up to three times, and written again if nothing passes.
- **Requests**: temperature 1, at most 4096 tokens, the system message "You are a helpful assistant.". These are the defaults of the package.
- **Agreement between solutions**: two programs agree when they get the same pass/fail outcomes on the public test cases. The gate reads this share as one of its two state features.

`Test` executes generated code in a child process with a time limit. For untrusted workloads, run LAMaS in a container.

## Where it comes from

- **Training data**: 33 HumanEval problems (`data/humaneval_train.jsonl`). The 131 test problems were not used.
- **Controller**: trained with the objective of the paper: latency weight 0.005, cost weight 3.0, an accuracy floor enforced with a dual variable (learning rate 0.01), and critical-path credit. LLM during training: gpt-4o-mini.
- **Stop gate**: trained afterwards on the traces of this controller on the training problems.

## Reproduce

```bash
sha256sum -c SHA256SUMS                                   # in this folder
lamas eval configs/humaneval-gpt4omini.yaml --out runs/humaneval
```
