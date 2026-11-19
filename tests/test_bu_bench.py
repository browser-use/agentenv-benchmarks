import asyncio
import importlib.util
import json
from pathlib import Path

import httpx

from agentenv_benchmarks.bu_bench import cloud, evidence, steps, tasks

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "src" / "agentenv_benchmarks" / "bundles" / "bu-bench"
TOOLS = ROOT / "tools" / "bu_bench"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tasks_decrypt_the_frozen_benchmark():
    assert len(tasks.task_ids()) == 200
    assert tasks.instruction("bu2-140").strip()


def test_bundle_tasks_carry_no_instruction_text():
    mini = json.loads((TOOLS / "reference" / "mini20.json").read_text())["tasks"]
    assert len(mini) == 20
    full = (BUNDLE / "evals" / "bu-bench-v21.toml").read_text()
    assert sorted(p.stem for p in BUNDLE.glob("tasks/*.json")) == tasks.task_ids()
    assert all(f'"{t}"' in full for t in tasks.task_ids())
    for path in BUNDLE.glob("tasks/*.json"):
        text = path.read_text()
        assert tasks.instruction(path.stem)[:40] not in text
        steps_ = json.loads(text)
        assert [s["type"] for s in steps_] == ["deploy_agent", "bu_cloud_browser", "prompt_agent",
                                                "collect_artifacts", "bu_findings_verifier"]
        assert steps_[2]["prompt"] == "<bu_task>"


def test_steps_round_trip():
    for cls, data in ((steps.BuCloudBrowserTaskStep, {"id": "b", "task_id": "bu2-001"}),
                      (steps.BuFindingsVerifierTaskStep, {"id": "g", "task_id": "bu2-001", "prompt_id": "solve"})):
        step = cls.from_dict(data)
        assert cls.from_dict(step.to_dict()).to_dict() == step.to_dict()


def test_codex_steps_keep_tools_and_messages_and_drop_reasoning():
    events = {"format": "codex-exec-json/1", "events": [
        {"type": "thread.started", "thread_id": "t"},
        {"type": "item.completed", "item": {"type": "reasoning", "text": "secret plan"}},
        {"type": "item.completed", "item": {"type": "command_execution", "command": "browser-harness <<'PY'",
                                            "aggregated_output": "title: Example"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "FINAL ANSWER: Example"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 600, "output_tokens": 10}},
    ]}
    fmt, found, usage = evidence.steps_and_usage(events)
    assert fmt == "codex-exec-json/1"
    assert found == ["$ browser-harness <<'PY'\ntitle: Example", "FINAL ANSWER: Example"]
    assert usage["input_tokens"] == 1000 and usage["cached_input_tokens"] == 600
    assert round(evidence.model_usd(usage), 6) == round((400 * 1 + 600 * 0.1 + 10 * 5) / 1e6, 6)


def test_browser_use_steps_pair_calls_with_results():
    messages = [
        {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "task"}]},
        {"kind": "response", "usage": {"input_tokens": 50, "cache_read_tokens": 10, "output_tokens": 5},
         "parts": [{"part_kind": "thinking", "content": "hmm"},
                   {"part_kind": "tool-call", "tool_name": "python", "args": {"code": "1+1"}, "tool_call_id": "c1"}]},
        {"kind": "request", "parts": [{"part_kind": "tool-return", "tool_name": "python", "content": "2",
                                       "tool_call_id": "c1"}]},
        {"kind": "response", "usage": {"input_tokens": 60, "cache_read_tokens": 50, "output_tokens": 7},
         "parts": [{"part_kind": "text", "content": "done"}]},
    ]
    fmt, found, usage = evidence.steps_and_usage({"events": messages})
    assert fmt == "browser-use-messages/1"
    assert found == ["thinking: hmm", 'python: {"code": "1+1"}\nresult: 2', "text: done"]
    assert usage == {"input_tokens": 110, "cached_input_tokens": 60, "output_tokens": 12, "model_requests": 2}


def test_images_never_reach_the_judge_as_base64():
    assert evidence.text_of({"type": "image", "data": "A" * 1000}) == "[image shown to the agent]"
    assert "AAAA" not in evidence.text_of(json.dumps({"data": "A" * 1000}))


def test_output_files_split_the_judge_cap():
    files = evidence.output_files([("a.csv", b"x" * 400_000), ("b.bin", b"\xff\xfe")])
    assert files[0]["clipped"] and len(files[0]["text"]) < 400_000
    assert files[1]["text"] is None


BROWSER_ID = "0f6c1a52-9d0e-4b8a-a1f7-3c2d5e6f7a8b"
WS = f"wss://{BROWSER_ID}.cdp.browser-use.com/devtools/browser/11111111-2222-3333-4444-555555555555"


def test_proxy_reads_responses_and_chat_usage():
    proxy = load_script("model_proxy")
    completed = {"type": "response.completed", "response": {"model": "m", "usage": {
        "input_tokens": 100, "input_tokens_details": {"cached_tokens": 40}, "output_tokens": 3}}}
    assert proxy.usage_of(completed)["cached_tokens"] == 40
    assert proxy.usage_of({"usage": {"prompt_tokens": 9, "completion_tokens": 1}})["output_tokens"] == 1
    assert proxy.usage_of({"type": "response.created"}) is None


def test_proxy_retries_upstream_rate_limits_before_the_agent_sees_them(monkeypatch, tmp_path):
    from starlette.testclient import TestClient

    class Body(httpx.AsyncByteStream):
        def __init__(self, payload):
            self.data = json.dumps(payload).encode()

        async def __aiter__(self):
            yield self.data

    proxy = load_script("model_proxy")
    ok = {"model": "m", "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    replies = iter([httpx.Response(429, headers={"retry-after": "1"}, stream=Body({"error": "rate"})),
                    httpx.Response(200, headers={"content-type": "application/json"}, stream=Body(ok))])
    proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: next(replies)))
    waits = []

    async def no_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(proxy.asyncio, "sleep", no_sleep)
    log = tmp_path / "usage.jsonl"
    monkeypatch.setenv("PROXY_TOKEN", "t")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("PROXY_USAGE_LOG", str(log))
    reply = TestClient(proxy.app).post("/v1/chat/completions", json={}, headers={"authorization": "Bearer t"})
    assert reply.status_code == 200 and waits == [2.0]
    statuses = [json.loads(line)["status"] for line in log.read_text().splitlines()]
    assert statuses == [429, 200] and proxy.state["rate_limited"] == 1


def test_scrub_removes_every_form_of_the_cdp_address():
    browser = cloud.CloudBrowser(id=BROWSER_ID, ws_url=WS, cdp_url=f"https://{BROWSER_ID}.cdp.browser-use.com")
    leaked = {"out": f"BU_CDP_WS={WS}", "nested": [f"connect to {BROWSER_ID}.cdp.browser-use.com failed",
                                                     f"id {BROWSER_ID}"]}
    text = json.dumps(cloud.scrub_json(leaked, browser.secrets))
    assert BROWSER_ID not in text and "cdp.browser-use.com" not in text and browser.label == BROWSER_ID[:8]
    assert BROWSER_ID not in repr(browser)


def test_api_errors_do_not_name_the_browser(monkeypatch):
    def reply(request):
        return httpx.Response(404, json={"detail": "nope"}, request=request)

    real = httpx.AsyncClient
    monkeypatch.setattr(cloud.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(reply), **kw))
    monkeypatch.setenv(cloud.KEY_NAME, "k")
    try:
        asyncio.run(cloud.request("GET", f"/browsers/{BROWSER_ID}"))
    except RuntimeError as error:
        assert BROWSER_ID not in str(error) and "404" in str(error)
    else:
        raise AssertionError("expected an error")


def test_screenshot_errors_are_scrubbed(tmp_path):
    shots = cloud.Screenshots("wss://" + BROWSER_ID + ".cdp.browser-use.com/x", tmp_path)

    async def boom(label):
        raise OSError(f"cannot reach {BROWSER_ID}.cdp.browser-use.com")

    shots._capture = boom
    asyncio.run(shots.capture("t1"))
    assert shots.status["errors"] and BROWSER_ID not in shots.status["errors"][0]


def codex_tools_agent(monkeypatch):
    """The Codex arms' agent module, imported over AgentEnv's own Codex agent sources (the image's base)."""
    import sys

    import agent_env

    codex = Path(agent_env.__file__).parent / "agents" / "agents" / "codex"
    monkeypatch.syspath_prepend(str(codex))
    monkeypatch.syspath_prepend(str(BUNDLE / "artifacts" / "codex-browser-tools"))
    monkeypatch.delitem(sys.modules, "codex_tools_agent", raising=False)
    return importlib.import_module("codex_tools_agent")


def test_every_codex_arm_has_an_agent_on_the_one_image(monkeypatch):
    import tomllib

    agent = codex_tools_agent(monkeypatch)
    tools = {}
    for path in BUNDLE.glob("agents/codex-*/agent.toml"):
        config = tomllib.loads(path.read_text())
        assert config["image"] == "codex-browser-tools", path
        tools[config["default_env_vars"]["BU_BENCH_TOOL"]] = path.parent.name
    assert set(tools) == set(agent.TOOLS) and tools["browser-harness"] == "codex-bu-cli"


def test_mcp_arms_name_the_server_their_prompt_names_and_parse_as_toml(monkeypatch):
    import re
    import tomllib

    agent = codex_tools_agent(monkeypatch)
    monkeypatch.setattr(agent, "BROWSE_HOME", Path("/nonexistent"))
    monkeypatch.setattr(agent, "browse_env", lambda: {"OPENCODE_BROWSER_PORT": "1"})
    for tool in agent.TOOLS:
        overrides = agent.mcp_overrides(agent.mcp_server(tool, WS, "gpt-6.1-terra", "medium"))
        named = re.search(r"the `([a-z_-]+)` MCP tools", agent.compose_prompt("t", tool))
        if named is None:  # a CLI tool: no MCP server
            assert overrides == [], tool
            continue
        config = {}
        for override in overrides:
            key, value = override.split("=", 1)
            config[key] = tomllib.loads(f"v = {value}")["v"]
        assert config[f"mcp_servers.{named.group(1)}.command"], tool


def test_codex_turn_keeps_mcp_config_and_scrubs_the_browser(monkeypatch):
    agent = codex_tools_agent(monkeypatch)
    seen = {}

    async def fake_codex(*, prompt, config_overrides, event_callback, **kwargs):
        seen.update(prompt=prompt, overrides=config_overrides)
        event_callback({"type": "item.completed", "item": {"type": "mcp_tool_call", "arguments": {"url": WS}}})
        return {"response": WS}  # not a dataclass: returned as is

    events = []
    monkeypatch.setattr(agent, "_run_codex", fake_codex)
    monkeypatch.setattr(agent, "_cdp_url", WS)
    agent._mcp[:] = agent.mcp_overrides(agent.mcp_server("playwright-mcp", WS))
    asyncio.run(agent.run_codex(prompt="task", event_callback=events.append))
    assert BROWSER_ID not in json.dumps(events) and seen["prompt"].endswith("Task:\ntask\n")
    assert 'web_search="disabled"' in seen["overrides"]
    assert any(o.startswith("mcp_servers.playwright.args=") and BROWSER_ID in o for o in seen["overrides"])


def test_repository_carries_no_decrypted_task_text():
    """Task text, rubrics and the canary live only in the encrypted benchmark file; no other file may carry them."""
    data = tasks._benchmark()
    needles = {data["canary"]: "canary"} if data.get("canary") else {}
    for task in data["tasks"]:
        needles[task["canary"]] = f"{task['id']} canary"
        for field in ("task", "rubric", "summary", "title"):
            text = " ".join(str(task.get(field) or "").split())
            if len(text) >= 40:
                needles |= {text[i:i + 40]: f"{task['id']} {field}" for i in (0, len(text) // 2 - 20, len(text) - 40)}
    skip = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "node_modules"}
    leaks = []
    ours = [ROOT / "src" / "agentenv_benchmarks" / "bu_bench", BUNDLE, TOOLS, Path(__file__)]
    for path in (p for top in ours for p in ([top] if top.is_file() else top.rglob("*"))):
        if path.is_dir() or skip & set(path.relative_to(ROOT).parts) or path.suffix == ".enc":
            continue
        text = " ".join(path.read_bytes().decode("utf-8", "replace").split())
        leaks += [f"{path.relative_to(ROOT)}: {where}" for needle, where in needles.items() if needle in text]
    assert not leaks, leaks[:5]  # where, never the text itself: CI logs can be public


def test_browser_handoff_sends_the_agents_control_token(monkeypatch):
    """Agents on agentenv-framework-protocol >= 0.1.582 answer 401 on cdp-config without their control token."""
    from types import SimpleNamespace

    from agent_env.a2a_agent.a2a_agent import A2AAgent

    seen = []
    real = httpx.AsyncClient

    def reply(request):
        seen.append(request)
        return httpx.Response(200, json={}, request=request)

    browser = cloud.CloudBrowser(id=BROWSER_ID, ws_url=WS, cdp_url=f"https://{BROWSER_ID}.cdp.browser-use.com")

    async def start(**kwargs):
        return browser

    monkeypatch.setattr(steps.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(reply), **kw))
    monkeypatch.setattr(steps.cloud.CloudBrowser, "start", start)
    monkeypatch.setattr(steps.cloud.Screenshots, "start", lambda self: None)
    card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_CDP_CONFIG, "params": {"endpoint": "/ext/cdp"}}]}}
    step = steps.BuCloudBrowserTaskStep("b", 1, "bu2-140")
    for token, header in [("tok", {"agentenv-control-token": "tok"}), (None, {})]:
        agent = SimpleNamespace(agent_name=step.agent_name, a2a_url="http://agent", a2a_card=card,
                                control_headers=header)
        context = SimpleNamespace(metadata={}, deployed_agents=[agent])
        asyncio.run(step.execute(context))
        assert seen[-1].url.path == "/ext/cdp"
        assert seen[-1].headers.get("agentenv-control-token") == token
        steps._LIVE.clear()


def test_preflight_names_a_missing_key_before_any_agent_or_browser_starts(monkeypatch):
    """A run with no Browser Use or judge key stops at preflight, not after an agent turn and a cloud browser."""
    for name in (cloud.KEY_NAME, *steps.JUDGE_KEY_NAMES):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(steps, "_judge_key", lambda: "")
    monkeypatch.setattr(cloud, "api_key", lambda: (_ for _ in ()).throw(RuntimeError("set BROWSER_USE_API_KEY")))
    browser = steps.BuCloudBrowserTaskStep("browser", 1, "bu2-140")
    verifier = steps.BuFindingsVerifierTaskStep("grade", 1, "bu2-140", "solve")
    assert "BROWSER_USE_API_KEY" in " ".join(browser.preflight())
    assert "BU_BENCH_JUDGE_API_KEY" in " ".join(verifier.preflight())
    monkeypatch.setattr(steps, "_judge_key", lambda: "k")
    monkeypatch.setattr(cloud, "api_key", lambda: "k")
    assert browser.preflight() == [] and verifier.preflight() == []


def test_bundle_tasks_are_what_make_bundle_writes(tmp_path, monkeypatch):
    """The committed task files and evals are make_bundle.py's output, byte for byte."""
    make_bundle = load_script("make_bundle")
    fake = tmp_path / "bu-bench"
    monkeypatch.setattr(make_bundle, "BUNDLE", fake)
    make_bundle.main()
    for kind in ("tasks", "evals"):
        written = {p.name: p.read_text() for p in (fake / kind).iterdir()}
        committed = {p.name: p.read_text() for p in (BUNDLE / kind).iterdir() if not p.name.startswith("ci-")}
        assert written == committed, kind
