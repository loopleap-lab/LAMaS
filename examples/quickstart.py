"""Solve one problem and look at what happened.

    python examples/quickstart.py            # simulated LLM, no API key needed
    python examples/quickstart.py --live     # $OPENAI_API_KEY (or $LAMAS_BASE_URL and $LAMAS_MODEL)
"""
import sys

import lamas

PROBLEM = '''def running_max(numbers: list) -> list:
    """ Return a list in which element i is the largest of the first i + 1 numbers.
    >>> running_max([1, 3, 2, 5, 4])
    [1, 3, 3, 5, 5]
    """
'''
REFERENCE = "    out = []\n    for n in numbers:\n        out.append(n if not out else max(out[-1], n))\n    return out\n"

# the simulated LLM is told the reference solution; a real one writes its own
backend = None if "--live" in sys.argv else lamas.MockBackend(answers={PROBLEM[:40]: PROBLEM + REFERENCE}, time_scale=0.2)

result = lamas.run(PROBLEM, backend=backend, seed=0)        # the released policy: lamas/humaneval

print(result.answer)
print()
print(result.trace.gantt())
print("critical path:", " -> ".join(result.trace.critical_path))
print("pruned by the stop gate:", result.trace.pruned or "nothing")
