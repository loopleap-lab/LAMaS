# Extending LAMaS to your domain

The runtime does not depend on the task. The scheduler, the stop policy, the trace and the endpoint
work the same for code, math, customer support or anything else. This guide takes it to a new domain:
your data, your prompts, your tools and your workflow. Each step is a config key or one Python
file, and none of it needs training.

| You want | You write | Section |
|---|---|---|
| Your dataset and how answers are scored | `task: {name: custom, data, fields, answer_format}` | [1](#1-your-data) |
| Your prompts | `task: {prompts, instructions}` | [2](#2-your-prompts) |
| A tool: retrieval, code execution, an API | one operator, as a function or a class | [3](#3-tools-are-operators) |
| Your own files in a config | `plugins: [tools.py]` | [4](#4-plugins) |
| A workflow without a trained controller | `controller: {type: fixed}`, `stop: {type: agreement}` | [5](#5-a-workflow-without-training) |
| Answers that need code to compare, a custom last step | a `Task` subclass | [6](#6-a-task-class) |
| Others can install it | entry points | [7](#7-share-it-as-a-package) |
| A controller, stop policy or serving engine of your own | one class each | [8](#8-the-other-slots) |

## The example

[`examples/extend/`](../examples/extend) is a shop that answers questions about orders: "A customer
orders 3 notebooks and 2 pens. What is the total?" It has a catalog of prices, two tools and one config.

```
examples/extend/
  orders.yaml     the config: data, prompts, plugins, workflow, stop policy
  orders.jsonl    six questions with their totals
  catalog.json    the shop's prices
  tools.py        PriceLookup (retrieval) and Calculator (code execution)
```

```bash
lamas run  examples/extend/orders.yaml "A customer orders 3 notebooks and 2 pens. What is the total, in dollars?"
lamas eval examples/extend/orders.yaml               # every question in orders.jsonl
lamas serve examples/extend/orders.yaml --port 8080  # the same, as an OpenAI-compatible endpoint
```

Add `--backend mock` to any of them to see the workflow and the trace offline. The simulated LLM does
not read the question, so its answers are only placeholders.

## 1. Your data

A dataset is a JSONL file with one problem per line. The `custom` task reads it and scores the answers:

```yaml
task:
  name: custom
  data:   {train: orders_train.jsonl, test: orders.jsonl}   # split -> file, relative to the config
  fields: {problem: question, answer: total}                # the keys that hold the problem and the answer
  answer_format: final_line                                 # how the short answer is read
```

`answer_format` decides when two answers are the same. It is used for scoring, for majority votes
and by the stop policy:

| `answer_format` | Reads | Example |
|---|---|---|
| `final_line` (default) | a `Final answer: ...` line | `Final answer: 15.9` |
| `boxed` | the last `\boxed{...}` | `\boxed{15.9}` |
| a regular expression | its first group in the last match | `'Answer: \(([A-J])\)'` (single quotes in YAML) |

Numbers are compared as numbers (`1,000` equals `1000.0`), and text without case or extra spaces.

From Python, the same task is a dict or a class:

```python
import lamas
from lamas.tasks import CustomTask

task = CustomTask(data={"test": "orders.jsonl"}, fields={"problem": "question", "answer": "total"})
policy = {"controller": {"type": "fixed", "layers": [["GenerateCoT", "GenerateCoT"], ["ScEnsemble"]]}}
report = lamas.evaluate(task, policy, backend="http://localhost:8000/v1")
print(report.table())
```

## 2. Your prompts

Every built-in operator builds its prompt from the task. Change any of them in the config:

```yaml
task:
  name: custom
  prompts:
    solve_instruction: "You work at the front desk of a shop. Answer the customer's question.\nEnd with a line `Final answer: <amount>`.\n\n"
  instructions:                       # one operator gets its own instruction
    GenerateCoT: "List every item with its price, add them up, then give the total.\n"
```

| Prompt | Used by | Fields filled in |
|---|---|---|
| `solve_instruction` | `Generate`, `GenerateCoT`, `MultiGenerateCoT` (as `{instruction}`) | none |
| `generate_template` | `Generate` | `{instruction}`, `{problem}` |
| `cot_template` | `GenerateCoT`, `MultiGenerateCoT` | `{instruction}`, `{problem}` |
| `refine_template` | `SelfRefine` | `{problem}`, `{solution}` |
| `ensemble_template` | `ScEnsemble`; its reply ends with `Answer: <letter>` | `{problem}`, `{solutions}` |

`instructions` replaces `solve_instruction` for the operators it names. Only the field names above
are filled in, and every other brace is left as it is, so prompts can hold code, JSON or LaTeX.
When you change how an answer ends, change `answer_format` with it.

## 3. Tools are operators

An operator is one agent step. A tool is an operator that calls something besides the LLM: a
search index, a code interpreter, a database, an internal API. Three things define it:

- `needs` tells the scheduler where it goes. `problem`: it reads only the problem and runs at the
  same time as the other operators that do. `solution`: it waits for the latest solution.
  `solutions`: it waits for all of them, like a vote.
- `description` is what a trained controller reads, so the controller can pick the operator
  without retraining.
- `run(ctx)` returns the solution text, or a list of texts.

A small one is a function. Its name is the operator's name and its docstring is the description:

```python
import lamas
from lamas.sandbox import run_python

@lamas.operator(needs="problem")
async def Calculator(ctx):
    """Writes the order total as one arithmetic expression from the catalog prices, then computes it with Python."""
    prices = "\n".join(f"- {item}: {price:.2f} dollars each" for _, item, price in lookup(ctx.problem))
    reply = await ctx.ask(WRITE_EXPRESSION, prices=prices)
    expression = last_expression(reply)            # e.g. 3 * 4.50 + 2 * 1.20
    if not expression:
        return reply
    ok, value = await run_python(f"def total():\n    return round({expression}, 2)", call="total")
    return f"{expression} = {value}\nFinal answer: {value}" if ok else reply
```

A larger one is a class:

```python
from lamas import Operator, register_operator

@register_operator
class PriceLookup(Operator):
    name        = "PriceLookup"
    description = "Looks up the price of every item of the order in the shop's catalog, then answers with those prices."
    needs       = "problem"

    async def run(self, ctx):
        found = lookup(ctx.problem)                     # the retrieval step: items, quantities, prices
        prices = "\n".join(f"- {item}: {price:.2f} dollars each" for _, item, price in found)
        bill = " + ".join(f"{n} x {price:.2f}" for n, _, price in found)
        return await ctx.ask(WITH_PRICES, instruction=ctx.task.instruction(self.name), prices=prices, bill=bill)
```

What `ctx` gives an operator:

| | |
|---|---|
| `ctx.problem`, `ctx.solution`, `ctx.solutions` | the problem, the latest solution, every solution so far |
| `ctx.item` | the whole data row, for ids or metadata |
| `ctx.task` | the task: its prompts, `ctx.task.instruction(name)`, its answer reader |
| `await ctx.ask(template, **fields)` | fill a prompt, call the LLM, return the text |
| `await ctx.llm(prompt)` | one LLM call with a string or a list of chat messages; returns the completion |
| `await ctx.gather(a, b, ...)` | several calls at the same time |
| `await ctx.call(OtherOperator())` | run another operator inside this one |

LLM calls made through `ctx` are counted in the trace, for the operator that made them. Time spent
in a tool shows in that operator's latency, so the critical path includes it.

`lamas.sandbox.run_python` runs code in a child process with a time limit. It keeps crashes and
endless loops out of the runtime; run untrusted code inside a container.

## 4. Plugins

`plugins` loads your Python files before anything else in the config, so the names they register can
be used in the rest of it:

```yaml
plugins: [tools.py, my_company.lamas_ops]     # files relative to the config, or module names
```

Then `lamas run`, `lamas eval` and `lamas serve` all know `PriceLookup` and `Calculator`. From
Python, import the file or call `lamas.load_plugins(["tools.py"])`.

## 5. A workflow without training

The `fixed` controller runs the same workflow for every query. Each inner list is one layer of
picks, in the form a trained controller produces them:

```yaml
policy:
  controller:
    type:   fixed
    layers: [[PriceLookup, Calculator], [ScEnsemble]]
  stop: {type: agreement, min_solutions: 2}
```

The scheduler turns the picks into a workflow with `needs`. `PriceLookup` and `Calculator` read only
the problem, so they run at the same time. `ScEnsemble` reads every solution, so it waits for both.

The stop policy decides after each operator whether to wait for the rest:

| `stop` | Stops when |
|---|---|
| `{type: agreement, min_solutions: 2, consensus: 1.0}` | that many solutions are in, and that share of them agree |
| `{type: none}` | never: everything planned runs |
| `{weights: gate.pth}` | the trained stop gate says so (a policy folder) |

In the example, when the two tools agree, `ScEnsemble` is pruned and the answer comes back one step
earlier. A latency budget works with any of them: `budget: {latency_s: 10}`.

Two more ways to choose the operators:

- **The released controller, with your operators in the pool.** It reads their descriptions:
  ```yaml
  policy:    lamas/humaneval
  operators: [Generate, GenerateCoT, MultiGenerateCoT, ScEnsemble, SelfRefine, PriceLookup, Calculator, EarlyStop]
  ```
- **A controller and a stop gate fitted to your task.** With a `train` split and a scorer, the same
  config works with `lamas train` (settings in the config's `train:` section).

## 6. A task class

A config covers data, prompts and short answers. Write a task class when answers need code to compare
or to score, or when the last step is special. HumanEval is one ([`tasks/humaneval.py`](../src/lamas/tasks/humaneval.py)):
two programs agree when they pass the same public tests, and the final answer is the first one that passes.

```python
from lamas import Task, register_task

@register_task(name="sql")
class SQL(Task):
    files = {"test": "sql_test.jsonl"}            # in $LAMAS_DATA
    solve_instruction = "Write one SQL query that answers the question.\n\n"

    def clean(self, text, item=None):             # a reply -> the solution passed on
        return extract_sql(text)

    async def akey(self, solution, item=None):    # when do two solutions agree? Run them.
        return hash_rows(await run_query(solution, item["db"]))

    async def ascore(self, item, answer):         # 1.0 when the result matches the reference
        return float(await same_rows(answer, item["gold"], item["db"]))

    async def finalize(self, ctx, solutions, current):   # the last step; the default is a vote
        return await super().finalize(ctx, solutions, current)
```

Load it with `plugins: [sql_task.py]` and use `task: sql`.

## 7. Share it as a package

Others can install your operators, tasks and the rest with `pip`, without a `plugins` line, through
entry points in your `pyproject.toml`:

```toml
[project.entry-points."lamas.operators"]
PriceLookup = "shop_tools:PriceLookup"
Calculator  = "shop_tools:Calculator"

[project.entry-points."lamas.tasks"]
sql = "shop_tools.sql:SQL"
```

The groups are `lamas.operators`, `lamas.tasks`, `lamas.controllers`, `lamas.stop_policies` and
`lamas.backends`.

## 8. The other slots

Each is one class, registered the same way:

```python
from lamas import Controller, Selection, StopPolicy, register_controller, register_stop_policy

@register_controller(name="draft-and-vote")
class DraftAndVote(Controller):
    def select(self, query, operators, *, seed=None, track_grad=False):
        return Selection(layers=[["Generate", "Generate"], ["ScEnsemble"]])

@register_stop_policy(name="after-two")
class AfterTwo(StopPolicy):
    def should_stop(self, state):                 # state: progress, consensus, n_solutions, elapsed, ...
        return state.n_solutions >= 2
```

A serving engine that does not speak the OpenAI API is a `Backend` with one method, `complete`
(see [`backends/base.py`](../src/lamas/backends/base.py)).

## Check it offline

A test can run the whole workflow on the simulated LLM. `answer_fn` decides what it answers:

```python
from lamas.backends import MockBackend

cfg = lamas.config.load_config("examples/extend/orders.yaml")
backend = MockBackend(time_scale=0, answer_fn=lambda prompt: "15.9" if "3 notebooks" in prompt else None)
runner = lamas.Runner(policy=cfg["policy"], task=cfg["task"], backend=backend)
print(runner.run("A customer orders 3 notebooks and 2 pens. What is the total, in dollars?").trace.gantt())
```

[`tests/test_extending.py`](../tests/test_extending.py) does this for the shop example.
