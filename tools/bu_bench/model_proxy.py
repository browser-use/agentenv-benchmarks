#!/usr/bin/env python3
"""The model endpoint for a local AgentEnv run: OpenAI behind one per-run token, with a spend cap and a usage log.

AgentEnv agents call `[model] base_url` as a LiteLLM gateway: Codex at <base>/openai/v1/responses, OpenAI-compatible
agents at <base>/v1/... This forwards both to https://api.openai.com/v1 with the real key, which never enters an
agent's container, logs each response's token usage (JSONL), and answers 429 once the run's recorded spend passes
PROXY_MAX_USD. An upstream rate limit (429) is retried here with backoff (Retry-After, else 2, 4, ... 60 s; up to
PROXY_RETRIES times) before the agent sees it, so a shared org limit slows a run instead of failing its turns.
A LiteLLM proxy does the same job for anyone who has one.

Env: OPENAI_API_KEY, PROXY_TOKEN (the key agents get), PROXY_USAGE_LOG, PROXY_MAX_USD (default 50), PROXY_PORT (4555).
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

UPSTREAM = os.environ.get("PROXY_UPSTREAM", "https://api.openai.com/v1")
PRICES = {"input": 1.0, "cached": 0.10, "output": 5.0}  # $/1M tokens, the bake-off's basis for gpt-6.1-terra
DROP_REQUEST = {"host", "authorization", "x-api-key", "content-length", "accept-encoding", "connection"}
DROP_RESPONSE = {"content-length", "content-encoding", "transfer-encoding", "connection"}
MAX_BUFFER = 64 * 1024 * 1024

state = {"usd": 0.0, "requests": 0, "rate_limited": 0, "spend_capped": 0}
client = httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30))


def usage_of(payload: dict) -> dict | None:
    """Usage from a Responses (response.completed / response) or Chat Completions body."""
    response = payload.get("response") if isinstance(payload.get("response"), dict) else payload
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return None
    if "input_tokens" in usage:
        return {"input_tokens": usage.get("input_tokens") or 0,
                "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0,
                "output_tokens": usage.get("output_tokens") or 0,
                "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0,
                "model": response.get("model")}
    if "prompt_tokens" in usage:
        return {"input_tokens": usage.get("prompt_tokens") or 0,
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                "output_tokens": usage.get("completion_tokens") or 0,
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
                "model": payload.get("model")}
    return None


def usd(usage: dict) -> float:
    cached = usage["cached_tokens"]
    return ((usage["input_tokens"] - cached) * PRICES["input"] + cached * PRICES["cached"]
            + usage["output_tokens"] * PRICES["output"]) / 1e6


def backoff(attempt: int, retry_after: str | None) -> float:
    try:
        wait = float(retry_after) if retry_after else 0.0
    except ValueError:
        wait = 0.0
    return min(60.0, max(wait, 2.0 ** (attempt + 1)))


def record(entry: dict) -> None:
    with open(os.environ["PROXY_USAGE_LOG"], "a") as log:
        log.write(json.dumps(entry) + "\n")


async def forward(request: Request):
    token = os.environ["PROXY_TOKEN"]
    if request.headers.get("authorization") != f"Bearer {token}" and request.headers.get("x-api-key") != token:
        return JSONResponse({"error": {"message": "bad proxy token"}}, status_code=401)
    tags = request.headers.get("x-litellm-tags", "")
    if state["usd"] >= float(os.environ.get("PROXY_MAX_USD", "50")):
        state["spend_capped"] += 1
        record({"t": round(time.time(), 3), "path": request.path_params["path"], "status": 429, "spend_cap": True,
                "tags": tags})
        return JSONResponse({"error": {"message": "run spend cap reached", "type": "spend_cap"}}, status_code=429)
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP_REQUEST}
    headers.update(authorization=f"Bearer {os.environ['OPENAI_API_KEY']}", **{"accept-encoding": "identity"})
    retries = int(os.environ.get("PROXY_RETRIES", "6"))
    for attempt in range(retries + 1):
        upstream = await client.send(client.build_request(
            request.method, f"{UPSTREAM}/{request.path_params['path']}", content=body, headers=headers,
            params=request.query_params), stream=True)
        if upstream.status_code != 429 or attempt == retries:
            break
        await upstream.aread()
        await upstream.aclose()
        state["rate_limited"] += 1
        wait = backoff(attempt, upstream.headers.get("retry-after"))
        record({"t": round(time.time(), 3), "path": request.path_params["path"], "status": 429, "retry": attempt + 1,
                "wait": wait, "tags": tags})
        await asyncio.sleep(wait)
    started = time.time()
    meta = {"path": request.path_params["path"], "status": upstream.status_code, "tags": tags}
    sse = "text/event-stream" in upstream.headers.get("content-type", "")

    async def stream():
        buffer, found = b"", None
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
                buffer += chunk
                if sse:
                    *lines, buffer = buffer.split(b"\n")
                    for line in lines:
                        if line.startswith(b"data:") and b'"usage"' in line:
                            try:
                                found = usage_of(json.loads(line[5:].strip())) or found
                            except ValueError:
                                pass
                elif len(buffer) > MAX_BUFFER:
                    buffer = b""
            if not sse and buffer:
                try:
                    found = usage_of(json.loads(buffer))
                except ValueError:
                    pass
        finally:
            await upstream.aclose()
            state["requests"] += 1
            entry = {"t": round(started, 3), "seconds": round(time.time() - started, 2), **meta}
            if found:
                entry.update(found, usd=round(usd(found), 6))
                state["usd"] += entry["usd"]
            entry["run_usd"] = round(state["usd"], 4)
            record(entry)

    response_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in DROP_RESPONSE}
    return StreamingResponse(stream(), status_code=upstream.status_code, headers=response_headers)


async def health(_request: Request):
    return JSONResponse({"ok": True, **state})


app = Starlette(routes=[
    Route("/healthz", health),
    Route("/openai/v1/{path:path}", forward, methods=["GET", "POST", "DELETE"]),
    Route("/v1/{path:path}", forward, methods=["GET", "POST", "DELETE"]),
    # Clients whose base URL has no /v1 (AgentEnv's stock browser-use agent) call /chat/completions directly.
    Route("/{path:path}", forward, methods=["GET", "POST", "DELETE"]),
])

if __name__ == "__main__":
    for name in ("OPENAI_API_KEY", "PROXY_TOKEN", "PROXY_USAGE_LOG"):
        if not os.environ.get(name):
            raise SystemExit(f"{name} is required")
    uvicorn.run(app, host=os.environ.get("PROXY_HOST", "0.0.0.0"), port=int(os.environ.get("PROXY_PORT", "4555")),
                log_level="warning")
