import json

import pytest

pytest.importorskip("fastapi")
import httpx  # noqa: E402

import lamas  # noqa: E402
from lamas.backends import MockBackend  # noqa: E402
from lamas.serve import Demo, create_app, demo_samples  # noqa: E402
from util import WIDE, fixed_policy  # noqa: E402

Q = "What is 17 * 24?"


@pytest.fixture
def client():
    runner = lamas.Runner(policy=fixed_policy(WIDE), backend=MockBackend(seed=2, time_scale=0.01, error_rate=0.0))
    app = create_app(runner, demo=Demo(runner=runner, samples=demo_samples(), source="simulated"))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_models_and_health(client):
    async with client:
        assert (await client.get("/health")).json()["status"] == "ok"
        assert "lamas/auto" in [m["id"] for m in (await client.get("/v1/models")).json()["data"]]


async def test_chat_completion_is_openai_shaped(client):
    async with client:
        r = await client.post("/v1/chat/completions", json={
            "model": "lamas/auto", "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": Q}]})
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion" and "408" in body["choices"][0]["message"]["content"]
        assert body["usage"]["total_tokens"] > 0 and body["lamas"]["usage"]["llm_calls"] >= 1
        assert int(r.headers["x-lamas-llm-calls"]) == body["lamas"]["usage"]["llm_calls"]
        assert r.headers["x-lamas-critical-path"]


async def test_options_ride_in_the_body(client):
    async with client:
        r = await client.post("/v1/chat/completions", json={
            "model": "lamas/auto", "messages": [{"role": "user", "content": Q}], "lamas": {"trace": False}})
        assert r.status_code == 200 and "lamas" not in r.json() and int(r.headers["x-lamas-llm-calls"]) >= 1


async def test_streaming_ends_with_done(client):
    async with client:
        async with client.stream("POST", "/v1/chat/completions", json={
                "model": "lamas/auto", "stream": True, "messages": [{"role": "user", "content": Q}]}) as r:
            text = "".join([chunk async for chunk in r.aiter_text()])
    data = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
    assert text.strip().endswith("data: [DONE]")
    assert data[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "408" in "".join(d["choices"][0]["delta"].get("content", "") for d in data)
    assert data[-1]["choices"][0]["finish_reason"] == "stop"
    assert ": lamas " in text            # trace events travel as SSE comments


async def test_bad_requests(client):
    async with client:
        assert (await client.post("/v1/chat/completions", json={"model": "lamas/auto", "messages": []})).status_code == 400
        assert (await client.get("/demo/stream", params={"q": " "})).status_code == 400
        assert (await client.get("/demo/stream", params={"q": Q, "source": "recorded"})).status_code == 404


async def test_demo_page_runs_a_problem_live(client):
    async with client:
        assert "LAMaS" in (await client.get("/demo")).text
        cfg = (await client.get("/demo/config")).json()
        assert cfg["source"] == "simulated" and cfg["policy"] == "test"
        r = await client.get("/demo/stream", params={"q": Q})
        events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
        assert events[0]["type"] == "plan" and events[-1]["type"] == "final" and "408" in events[-1]["answer"]


async def test_demo_page_replays_a_recorded_run(client):
    async with client:
        samples = (await client.get("/demo/config")).json()["samples"]
        recorded = [s for s in samples if s["recorded"]]
        if not recorded:
            pytest.skip("no recorded runs are installed")
        r = await client.get("/demo/stream", params={"sample": recorded[0]["id"], "source": "recorded", "speed": 1000})
        events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
        kinds = [e["type"] for e in events]
        assert kinds[0] == "meta" and kinds[1] == "plan" and kinds[-1] == "final"
        assert kinds.count("start") == kinds.count("end") and events[-1]["correct"] is True


async def test_litellm_provider():
    litellm = pytest.importorskip("litellm")
    from lamas.adapters.litellm_provider import register

    handler = register(lamas.Runner(policy=fixed_policy(WIDE), backend=MockBackend(seed=1, time_scale=0.01, error_rate=0.0)))
    reply = await litellm.acompletion(model="lamas/auto", messages=[{"role": "user", "content": Q}])
    assert "408" in reply.choices[0].message.content and reply.usage.total_tokens > 0
    assert handler.last_trace.usage.llm_calls >= 1
