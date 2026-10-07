"""Use LAMaS from an unmodified OpenAI client. Start the endpoint first:

    lamas serve configs/humaneval-gpt4omini.yaml --port 8080
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

problem = '''def is_palindrome(text: str) -> bool:
    """ Check whether the given string reads the same in both directions. """
'''
reply = client.chat.completions.create(
    model="lamas/auto",                                         # the policy the endpoint was started with
    messages=[{"role": "user", "content": problem}],
    extra_body={"lamas": {"budget": {"latency_s": 20}}},        # optional: a latency budget for this request
)
print(reply.choices[0].message.content)

trace = reply.model_extra["lamas"]                              # the trace rides along in the response
print(f"{trace['latency']} s, {trace['usage']['llm_calls']} LLM calls, critical path: {trace['critical_path']}")
