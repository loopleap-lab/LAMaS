"""`lamas serve`: an OpenAI-compatible endpoint in front of a LAMaS runner.

    POST /v1/chat/completions      the same request an OpenAI client sends
    GET  /v1/models                lamas/auto plus every installed policy
    GET  /demo                     the demo page (when enabled)

Per-request options travel in a ``lamas`` object in the body (``extra_body`` in the OpenAI SDK):

    client.chat.completions.create(
        model="lamas/auto", messages=[...],
        extra_body={"lamas": {"budget": {"latency_s": 10}}})

The response is an ordinary chat completion. The trace rides along in a ``lamas`` field and a
few ``x-lamas-*`` headers, where clients that do not know about it will never look.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..api import Runner
from ..policy import available

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDINGS = os.path.join(HERE, "recordings")


def demo_samples() -> List[Dict[str, Any]]:
    """The sample problems of the demo page: ``[{"id", "prompt", "entry_point", "test", "solution", "seed"}]``."""
    path = os.path.join(HERE, "demo_samples.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Demo:
    """What the demo page shows: one problem, the workflow LAMaS runs for it, live."""

    runner: Runner
    samples: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "live"                             # live | simulated
    recordings_dir: str = RECORDINGS                 # recorded runs of the samples, one file per sample id


def _query_from_messages(messages: List[Dict[str, Any]]) -> str:
    def text(content: Any) -> str:
        if isinstance(content, list):               # OpenAI "content parts"
            return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        return str(content or "")

    system = "\n".join(text(m.get("content")) for m in messages if m.get("role") == "system").strip()
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if not turns or turns[-1].get("role") != "user":
        raise HTTPException(status_code=400, detail="the last message must come from the user")
    question = text(turns[-1].get("content")).strip()
    history = "\n".join(f"{m['role']}: {text(m.get('content'))}" for m in turns[:-1]).strip()
    parts = []
    if system:
        parts.append(system)
    if history:
        parts.append("Conversation so far:\n" + history)
    parts.append(question)
    return "\n\n".join(parts)


def _completion(result: Any, model: str, include_trace: bool) -> Dict[str, Any]:
    trace = result.trace
    body: Dict[str, Any] = {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result.answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": trace.usage.prompt_tokens, "completion_tokens": trace.usage.completion_tokens,
                  "total_tokens": trace.usage.prompt_tokens + trace.usage.completion_tokens},
    }
    if include_trace:
        body["lamas"] = trace.to_dict()
    return body


def _headers(trace: Any) -> Dict[str, str]:
    return {
        "x-lamas-latency-ms": str(int(trace.latency * 1000)),
        "x-lamas-llm-calls": str(trace.usage.llm_calls),
        "x-lamas-critical-path": ",".join(trace.critical_path),
        "x-lamas-pruned": ",".join(trace.pruned),
        "x-lamas-stop": trace.stop_reason or "",
    }


def create_app(runner: Runner, demo: Optional[Demo] = None) -> FastAPI:
    """Build the ASGI app around a default runner (served as ``lamas/auto``)."""
    app = FastAPI(title="LAMaS", version="0.1.0", docs_url=None, redoc_url=None)
    runners: Dict[str, Runner] = {"lamas/auto": runner, runner.policy.name: runner}

    def runner_for(model: str) -> Runner:
        if model in runners:
            return runners[model]
        if model in available():                    # another installed policy, on the same serving engine
            runners[model] = Runner(policy=model, backend=runner.backend, budget=runner.budget)
            return runners[model]
        return runner

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "backend": runner.backend.describe(), "policy": runner.policy.name}

    @app.get("/v1/models")
    async def models() -> Dict[str, Any]:
        ids = ["lamas/auto"] + [name for name in available()]
        return {"object": "list", "data": [{"id": i, "object": "model", "created": 0, "owned_by": "lamas"} for i in ids]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Any:
        body = await request.json()
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")
        model = body.get("model") or "lamas/auto"
        opts = body.get("lamas") or {}
        query = _query_from_messages(messages)
        active = runner_for(model)
        kwargs = {"budget": opts.get("budget"), "seed": opts.get("seed")}
        include_trace = bool(opts.get("trace", True))

        if not body.get("stream"):
            try:
                result = await active.arun(query, **kwargs)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc
            return JSONResponse(_completion(result, model, include_trace), headers=_headers(result.trace))

        async def stream() -> AsyncIterator[str]:
            cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

            def chunk(delta: Dict[str, Any], finish: Optional[str] = None, **extra: Any) -> str:
                payload = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                payload.update(extra)
                return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            queue: asyncio.Queue = asyncio.Queue()
            job = asyncio.ensure_future(active.arun(query, on_event=queue.put_nowait, **kwargs))
            yield chunk({"role": "assistant", "content": ""})
            try:
                while not job.done() or not queue.empty():
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.25)
                    except asyncio.TimeoutError:
                        continue
                    if event.get("type") != "final":
                        yield f": lamas {json.dumps(event, ensure_ascii=False)}\n\n"     # an SSE comment: clients skip it
                result = await job
                yield chunk({"content": result.answer})
                tail = {"lamas": result.trace.to_dict()} if include_trace else {}
                yield chunk({}, finish="stop", **tail)
            except Exception as exc:
                yield chunk({"content": f"[lamas error] {type(exc).__name__}: {exc}"}, finish="stop")
            finally:
                if not job.done():
                    job.cancel()
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    if demo is not None:
        _mount_demo(app, demo)
    return app


# ---- demo ----------------------------------------------------------------------------------------
def _mount_demo(app: FastAPI, demo: Demo) -> None:
    by_id = {s["id"]: s for s in demo.samples}

    def recording(sample_id: Optional[str]) -> Optional[str]:
        if not sample_id or sample_id not in by_id:
            return None
        path = os.path.join(demo.recordings_dir, f"{sample_id.replace('/', '_')}.json")
        return path if os.path.exists(path) else None

    @app.get("/demo", response_class=HTMLResponse)
    @app.get("/", response_class=HTMLResponse)
    async def page() -> str:
        with open(os.path.join(HERE, "demo.html"), "r", encoding="utf-8") as f:
            return f.read()

    @app.get("/demo/config")
    async def config() -> Dict[str, Any]:
        samples = [{"id": s["id"], "label": s.get("entry_point") or s["id"], "prompt": s["prompt"],
                    "recorded": recording(s["id"]) is not None} for s in demo.samples]
        return {"backend": demo.runner.backend.describe(), "source": demo.source, "policy": demo.runner.policy.name,
                "samples": samples}

    @app.get("/demo/stream")
    async def stream(q: str = "", sample: Optional[str] = None, source: str = "live", speed: float = 1.0) -> Any:
        known = by_id.get(sample or "")
        problem = known["prompt"] if known else q
        if not problem.strip():
            raise HTTPException(status_code=400, detail="empty problem")
        path = recording(sample)
        if source == "recorded" and not path:
            raise HTTPException(status_code=404, detail="no recording for this problem")

        async def replay() -> AsyncIterator[str]:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            meta = {"type": "meta", "backend": rec.get("backend", ""), "recorded_at": rec.get("recorded_at", ""), "t": 0}
            yield f"data: {json.dumps(meta)}\n\n"
            started = time.perf_counter()
            for event in rec["events"]:
                wait = float(event.get("t", 0.0)) / max(speed, 0.01) - (time.perf_counter() - started)
                if wait > 0:
                    await asyncio.sleep(wait)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        async def live() -> AsyncIterator[str]:
            queue: asyncio.Queue = asyncio.Queue()
            item = {k: known[k] for k in ("prompt", "entry_point", "test")} if known else None
            job = asyncio.ensure_future(demo.runner.arun(problem, item=item, on_event=queue.put_nowait,
                                                         seed=known.get("seed") if known else None))
            last_sent = time.perf_counter()
            try:
                while not job.done() or not queue.empty():
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.2)
                    except asyncio.TimeoutError:
                        if time.perf_counter() - last_sent > 10:     # keep proxies from closing an idle stream
                            last_sent = time.perf_counter()
                            yield ": keep-alive\n\n"
                        continue
                    last_sent = time.perf_counter()
                    if event.get("type") == "final" and item is not None:     # a sample problem: grade it with its hidden tests
                        event = dict(event, correct=bool(await demo.runner.task.ascore(item, event.get("answer") or "")))
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                exc = job.exception()
                if exc is not None:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'{type(exc).__name__}: {exc}'})}\n\n"
            finally:
                if not job.done():
                    job.cancel()

        return StreamingResponse(replay() if source == "recorded" else live(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
