"""bu-browser: its example task plays as an OpenEnv episode scored by its own verifier, its key can only be a secret
reference, its env type stops the browser as it closes, and its site starts, serves, scrubs, resets and stops a Browser
Use Cloud browser, here a fake Browser Use API and a fake Playwright MCP, so no key, network or Docker is needed."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import socket
import sys
import threading
import time
import tomllib
from pathlib import Path

import agent_env
import pytest
import uvicorn
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import live_problems
from agent_env.env.registry import get_env_registry
from agent_env.task import EpisodeSpec, Task
from agent_env.task.episode.rules import works_in_sandbox
from agent_env.utils.card_naming import website_source_from_source
from agentenv_protocol import HIDDEN_BY_DEFAULT_META_KEY
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

import agentenv_benchmarks.bundles
from agentenv_benchmarks.bu_bench import cloud as bu_bench_cloud
from agentenv_benchmarks.bu_browser import env as bu_env

BUNDLE = Path(agentenv_benchmarks.bundles.__file__).parent / "bu-browser"
SITE = BUNDLE / "envs" / "bu-browser-site"
TASK = BUNDLE / "tasks" / "pypi-browser-use-wheel-sha256.json"
FAKE_PLAYWRIGHT_MCP = Path(__file__).with_name("fake_playwright_mcp.py")

# The site's modules import each other by name, as the image runs them from one folder; no __pycache__ is left in the
# folder, which is the image's build context.
sys.path.insert(0, str(SITE))
no_bytecode_before, sys.dont_write_bytecode = sys.dont_write_bytecode, True
import browser_site
import cloud as site_cloud
import playwright_mcp

sys.path.remove(str(SITE))
sys.dont_write_bytecode = no_bytecode_before

MCP_HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
READY = "/agentenv/ext/website/ready"
TOOLS = [{"name": name, "description": name, "inputSchema": schema, "annotations": {"readOnlyHint": False}}
         for name, schema in (
             ("browser_navigate", {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}),
             ("browser_environment", {"type": "object", "properties": {}}),
             ("browser_evaluate", {"type": "object", "properties": {"function": {"type": "string"}}}),
             ("browser_run_code_unsafe", {"type": "object", "properties": {"code": {"type": "string"}}}))]


def test_the_example_task_is_an_episode_on_the_browser_alone_scored_by_its_verifier():
    task = Task.from_dict({"id": "pypi", "version": None, "steps": json.loads(TASK.read_text())})
    spec = EpisodeSpec.from_task(task)
    [deploy] = [step for step in task.steps if step.type == "deploy_agent"]
    access = [(step.env_id, step.action, step.role, step.tools) for step in task.steps
              if step.type == "modify_env_tool_access"]
    reward, problem = _reward_of()(task, None)

    assert spec.trainer_agents == (deploy.agent_name,)
    assert deploy.env_ids == ["bu-browser"] and not works_in_sandbox(task.steps, deploy.agent_name)
    assert access == [("bu-browser", "disable", "default", ["list_website_urls"])]
    assert problem is None and reward({"grade": 1.0}) == 1.0


def test_the_bundle_runs_no_bu_bench_step_or_task():
    steps = [step for path in (BUNDLE / "tasks").glob("*.json") for step in json.loads(path.read_text())]

    assert {step["type"] for step in steps}.isdisjoint({"bu_cloud_browser", "bu_findings_verifier"})
    assert not any(path.name.startswith("bu2-") for path in (BUNDLE / "tasks").iterdir())


def test_the_key_is_a_secret_setting_so_only_a_reference_to_it_is_accepted():
    source = website_source_from_source(str(SITE / "Dockerfile"), str(SITE))
    declared = tomllib.loads((SITE / "env.toml").read_text())
    domains = declared["domains"]

    assert source.secret_settings == ["api_key"]
    assert declared["settings"]["api_key"] == "secret:BROWSER_USE_API_KEY"
    assert live_problems(False, domains, [], declared["settings"], source.secret_settings) == []
    assert live_problems(False, domains, [], {**declared["settings"], "api_key": "bu_0123456789"},
                         source.secret_settings)


def test_the_bu_browser_type_is_this_packages_multi_env():
    assert get_env_registry()["bu_browser"] is bu_env.BuBrowserEnv
    assert issubclass(bu_env.BuBrowserEnv, MultiEnv)


def test_the_sites_cloud_client_is_bu_benchs_byte_for_byte():
    assert (SITE / "cloud.py").read_bytes() == Path(bu_bench_cloud.__file__).read_bytes()


@pytest.mark.parametrize("reset_fails", [False, True], ids=["reset", "reset fails"])
def test_closing_resets_the_site_then_closes_the_env(monkeypatch, reset_fails):
    calls = _closing(monkeypatch, reset=RuntimeError("gateway gone") if reset_fails else None)
    env = _env()
    env._deployed = "deployment"

    asyncio.run(env.close())

    assert calls == [("reset", "deployment"), ("close",)]


def test_closing_gives_up_on_a_reset_that_hangs_and_still_closes(monkeypatch):
    calls = _closing(monkeypatch, reset=asyncio.sleep(60))
    monkeypatch.setattr(bu_env, "STOP_TIMEOUT_S", 0.05)
    env = _env()
    env._deployed = "deployment"

    asyncio.run(env.close())

    assert calls == [("reset", "deployment"), ("close",)]


def test_closing_an_env_never_deployed_only_closes(monkeypatch):
    calls = _closing(monkeypatch)

    asyncio.run(_env().close())

    assert calls == [("close",)]


def test_the_site_boots_one_browser_serves_its_tools_and_stops_it_on_reset_and_shutdown(site, browser_use):
    with TestClient(site.create_app()) as client:
        assert _ready(client)["ready"] is True
        [(first, asked)] = browser_use.created
        mcp = _mcp(client)
        listed = {tool["name"]: bool((tool.get("_meta") or {}).get(HIDDEN_BY_DEFAULT_META_KEY))
                  for tool in _rpc(client, mcp, "tools/list")["tools"]}
        navigated = _text(_rpc(client, mcp, "tools/call", {"name": "browser_navigate",
                                                            "arguments": {"url": "https://pypi.org/"}}))
        environment = json.loads(_text(_rpc(client, mcp, "tools/call", {"name": "browser_environment",
                                                                          "arguments": {}})))
        reset = client.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset", "params": {}})
        stopped_by_reset, running_after_reset = list(browser_use.stopped), site.playwright.running
        again = _text(_rpc(client, mcp, "tools/call", {"name": "browser_navigate", "arguments": {"url": "https://x/"}}))
        [_, (second, _)] = browser_use.created

    assert browser_use.keys == {"bu-test-key"}
    assert asked == {"proxyCountryCode": "us", "timeout": 30, "enableRecording": False}
    # browser_evaluate is listed hidden, which the gateway hides until tool access names it; run_code_unsafe isn't.
    assert listed == {"browser_navigate": False, "browser_environment": False, "browser_evaluate": True}
    assert navigated.startswith("navigated to https://pypi.org/ over <cdp>") and first not in navigated
    assert "cdp.test" not in navigated
    assert {"AGENTENV_WEBSITE_SETTINGS", "BROWSER_USE_API_KEY"}.isdisjoint(environment) and "USER" in environment
    assert reset.json().get("result") == {} and stopped_by_reset == [first] and not running_after_reset
    assert again.startswith("navigated to https://x/ over <cdp>")
    assert browser_use.stopped == [first, second] and not site.playwright.running


def test_a_browser_that_fails_to_start_fails_readiness_and_leaves_nothing(site, browser_use):
    browser_use.refuse = True
    with TestClient(site.create_app()) as client:
        ready = _ready(client)

    assert ready["ready"] is False and "returned 429" in " ".join(ready["problems"])
    assert browser_use.created == [] and not site.playwright.running


def test_playwright_mcp_failing_to_start_stops_the_browser_it_was_given(site, browser_use, monkeypatch):
    monkeypatch.setattr(playwright_mcp, "command", lambda endpoint, port: [sys.executable, "-c", "raise SystemExit(3)"])
    with TestClient(site.create_app()) as client:
        ready = _ready(client)

    assert ready["ready"] is False and "exited at start with 3" in " ".join(ready["problems"])
    assert browser_use.stopped == [browser_use.created[0][0]] and not site.playwright.running


def test_a_reset_while_the_browser_is_still_being_created_stops_it_once_created(site, browser_use):
    browser_use.delay_s = 1.0
    with TestClient(site.create_app()) as client:
        while not browser_use.asked:
            time.sleep(0.01)
        client.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset", "params": {}})
        stopped_by_reset = list(browser_use.stopped)

    assert stopped_by_reset == [browser_use.created[0][0]] and len(browser_use.created) == 1
    assert browser_use.stopped == stopped_by_reset


class FakeBrowserUse:
    """Browser Use's API as cloud.py calls it: create a browser, its CDP discovery, stop it; each call recorded."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.asked = 0
        self.created: list[tuple[str, dict]] = []
        self.stopped: list[str] = []
        self.keys: set[str] = set()
        self.refuse = False
        self.delay_s = 0.0

    def app(self) -> Starlette:
        async def create(request):
            self.asked += 1
            self.keys.add(request.headers.get("x-browser-use-api-key"))
            if self.refuse:
                return JSONResponse({"detail": "too many concurrent browsers"}, 429)
            await asyncio.sleep(self.delay_s)
            browser = f"b{len(self.created) + 1}-0123456789abcdef"
            self.created.append((browser, await request.json()))
            return JSONResponse({"id": browser, "cdpUrl": f"{self.url}/cdp/{browser}"})

        async def version(request):
            return JSONResponse({"webSocketDebuggerUrl": f"ws://cdp.test/devtools/{request.path_params['id']}"})

        async def stop(request):
            self.stopped.append(request.path_params["id"])
            return JSONResponse({"id": request.path_params["id"]})

        return Starlette(routes=[Route("/browsers", create, methods=["POST"]),
                                 Route("/cdp/{id}/json/version", version),
                                 Route("/browsers/{id}", stop, methods=["PATCH"])])


@pytest.fixture
def browser_use():
    port = _free_port()
    fake = FakeBrowserUse(f"http://127.0.0.1:{port}")
    server = uvicorn.Server(uvicorn.Config(fake.app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield fake
    server.should_exit = True
    thread.join(10)


@pytest.fixture
def site(browser_use, tmp_path, monkeypatch):
    manifest = tmp_path / "playwright-mcp-tools.json"
    manifest.write_text(json.dumps({"version": "test", "tools": TOOLS}))
    monkeypatch.setattr(playwright_mcp, "TOOLS", manifest)
    monkeypatch.setattr(playwright_mcp, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(playwright_mcp, "LOG", tmp_path / "playwright-mcp.log")
    monkeypatch.setattr(playwright_mcp, "command", lambda endpoint, port: [
        sys.executable, str(FAKE_PLAYWRIGHT_MCP), "--cdp-endpoint", endpoint, "--port", str(port)])
    monkeypatch.setattr(site_cloud, "API", site_cloud.API)  # the site sets both from its settings
    monkeypatch.setattr(site_cloud, "api_key", site_cloud.api_key)
    monkeypatch.setenv("AGENTENV_WEBSITE_SETTINGS", json.dumps({"api_key": "bu-test-key", "api_url": browser_use.url}))
    monkeypatch.setenv("BROWSER_USE_API_KEY", "bu-test-key")
    playwright_mcp.manifest.cache_clear()
    yield browser_site.BuBrowser()
    playwright_mcp.manifest.cache_clear()


def _closing(monkeypatch, reset: BaseException | asyncio.Future | None = None) -> list[tuple]:
    calls: list[tuple] = []

    async def resetting(self, deployed):
        calls.append(("reset", deployed))
        if isinstance(reset, BaseException):
            raise reset
        if reset is not None:
            await reset

    async def closing(self):
        calls.append(("close",))

    monkeypatch.setattr(bu_env.BuBrowserEnv, "reset", resetting)
    monkeypatch.setattr(MultiEnv, "close", closing)
    return calls


def _env() -> bu_env.BuBrowserEnv:
    return bu_env.BuBrowserEnv(id="bu-browser", version=1, mcp_server_envs=[], website_envs=[], browser_envs=[])


def _reward_of():
    """openenv's ``reward_of``, which ``build_app`` scores a task by, read from its file: importing its package needs
    the openenv extra, which this package doesn't install."""
    spec = importlib.util.spec_from_file_location("agentenv_openenv_reward",
                                                  Path(agent_env.__file__).parent / "openenv" / "reward.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reward_of


def _ready(client: TestClient) -> dict:
    """The site's readiness once its boot has said either way."""
    deadline = time.monotonic() + 30
    while True:
        ready = client.get(READY).json()
        if ready["ready"] or any("returned False" not in problem for problem in ready["problems"]):
            return ready
        assert time.monotonic() < deadline, ready
        time.sleep(0.05)


def _mcp(client: TestClient) -> dict:
    first = client.post("/mcp", headers=MCP_HEADERS, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}})
    headers = {**MCP_HEADERS, "mcp-session-id": first.headers["mcp-session-id"]}
    client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    return headers


def _rpc(client: TestClient, headers: dict, method: str, params: dict | None = None) -> dict:
    response = client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": method,
                                                          "params": params or {}})
    text = response.text
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        text = [line[len("data:"):] for line in text.splitlines() if line.startswith("data:")][-1]
    return json.loads(text)["result"]


def _text(result: dict) -> str:
    assert not result.get("isError"), result
    return result["content"][0]["text"]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
