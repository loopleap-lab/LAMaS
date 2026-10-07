"""LAMaS as a LiteLLM provider: frameworks built on LiteLLM (DSPy, for one) can call it by name.

    pip install litellm
    python examples/litellm_provider.py
"""
import litellm

import lamas
from lamas.adapters.litellm_provider import register

PROBLEM = '''def double(x: int) -> int:
    """ Return twice the given number. """
'''
backend = lamas.MockBackend(answers={PROBLEM[:20]: PROBLEM + "    return 2 * x\n"}, time_scale=0.1)    # simulated LLM
register(lamas.Runner(backend=backend))                         # the released policy on that serving engine

reply = litellm.completion(model="lamas/auto", messages=[{"role": "user", "content": PROBLEM}])
print(reply.choices[0].message.content)

# In DSPy the same model name works once the provider is registered:
#   import dspy
#   dspy.configure(lm=dspy.LM("lamas/auto"))
