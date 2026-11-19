"""bu-browser: its example task plays as an OpenEnv episode scored by its own verifier, its key can only be a secret
reference, its env type stops the browser as it closes, and its site starts, serves, scrubs, resets and stops a Browser
Use Cloud browser, here a fake Browser Use API and a fake Playwright MCP, so no key, network or Docker is needed."""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import socket
import sys
import threading
import time
import tomllib
from collections.abc import Awaitable
from pathlib import Path
from unittest.mock import MagicMock

import agent_env
import httpx
import pytest
import uvicorn
from agent_env.env.env import DeployedEnv, Env
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv, live_problems
from agent_env.env.registry import get_env_registry
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.task import EpisodeSpec, Task, teardown
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
             ("browser_click", {"type": "object", "properties": {"ref": {"type": "string"}}}),
             ("browser_crash", {"type": "object", "properties": {}}),
             ("browser_close", {"type": "object", "properties": {}}),
             ("browser_run_code_unsafe", {"type": "object", "properties": {"code": {"type": "string"}}}))]
HASH = "3721b75f6c501523694ef58cd7a3911c032cdabddbb38bca02709a249465c68f"


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


@pytest.mark.parametrize("reply, scored", [(HASH, True), (f" {HASH.upper()}\n", True),
                                           (f"The hash is {HASH}.", False),
                                           (f"{HASH} or b1e0d7a9a8e1f2c3d4b5a6f7e8d9c0b1a2f3e4d5c6b7a8f9e0d1c2b3a4f5e6d7", False)],
                         ids=["the hash", "the hash spaced and upper case", "the hash in a sentence", "two hashes"])
def test_the_example_task_scores_only_a_reply_that_is_the_hash(reply, scored):
    task = Task.from_dict({"id": "pypi", "version": None, "steps": json.loads(TASK.read_text())})
    [grade] = [step for step in task.steps if step.type == "agent_prompt_response_verifier"]

    assert [grade._eval_criterion(0, reply, criterion)["passed"] for criterion in grade.criteria] == [scored]


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


@pytest.mark.parametrize("stop_fails", [False, True], ids=["stopped", "stop fails"])
def test_closing_stops_the_sites_browser_for_good_then_closes_the_env(monkeypatch, stop_fails):
    calls, deployed = _closing(monkeypatch, answer=RuntimeError("gateway gone") if stop_fails else None)
    env = _env()
    env._deployed = deployed

    asyncio.run(env.close())

    assert calls == [("invoke", browser_site.CLOSE_URI, "close", "bu-browser-site"), ("close",)]
    # The extension call waits for the site's slowest stop: every request timing out, with the waits between them.
    slowest = site_cloud.STOP_ATTEMPTS * site_cloud.STOP_REQUEST_S + site_cloud.STOP_RETRY_S * sum(
        range(1, site_cloud.STOP_ATTEMPTS))
    assert deployed.timeouts == [int(bu_env.STOP_TIMEOUT_S)] and deployed.timeouts[0] >= slowest


def test_closing_gives_up_on_a_stop_that_hangs_and_still_closes(monkeypatch):
    calls, deployed = _closing(monkeypatch, answer=asyncio.sleep(60))
    monkeypatch.setattr(bu_env, "STOP_TIMEOUT_S", 0.05)
    env = _env()
    env._deployed = deployed

    asyncio.run(env.close())

    assert calls == [("invoke", browser_site.CLOSE_URI, "close", "bu-browser-site"), ("close",)]


def test_closing_an_env_never_deployed_only_closes(monkeypatch):
    calls, _ = _closing(monkeypatch)

    asyncio.run(_env().close())

    assert calls == [("close",)]


def test_the_site_starts_its_browser_at_the_first_call_serves_its_tools_and_stops_it_on_reset_and_shutdown(
        site, browser_use):
    with TestClient(site.create_app()) as client:
        assert _ready(client)["ready"] is True
        created_when_ready = list(browser_use.created)
        mcp = _mcp(client)
        listed = {tool["name"]: bool((tool.get("_meta") or {}).get(HIDDEN_BY_DEFAULT_META_KEY))
                  for tool in _rpc(client, mcp, "tools/list")["tools"]}
        navigated = _text(_call(client, mcp, "browser_navigate", url="https://pypi.org/"))
        [(first, asked)] = browser_use.created
        environment = json.loads(_text(_call(client, mcp, "browser_environment")))
        reset = client.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset", "params": {}})
        stopped_by_reset, running_after_reset = list(browser_use.stopped), site.playwright.running
        again = _text(_call(client, mcp, "browser_navigate", url="https://x/"))
        [_, (second, _)] = browser_use.created

    # No browser before a call: none runs, or counts down its timeout, before the run records the env.
    assert created_when_ready == []
    assert browser_use.keys == {"bu-test-key"}
    assert asked == {"proxyCountryCode": "us", "timeout": 30, "enableRecording": False}
    # browser_evaluate is listed hidden, which the gateway hides until tool access names it; browser_close, which can
    # end the cloud browser, and run_code_unsafe aren't listed.
    assert listed == {"browser_navigate": False, "browser_environment": False, "browser_evaluate": True,
                      "browser_click": False, "browser_crash": False}
    assert navigated.startswith("navigated to https://pypi.org/ over <cdp>") and first not in navigated
    assert "cdp.test" not in navigated
    assert {"AGENTENV_WEBSITE_SETTINGS", "BROWSER_USE_API_KEY"}.isdisjoint(environment) and "USER" in environment
    assert reset.json().get("result") == {} and stopped_by_reset == [first] and not running_after_reset
    assert again.startswith("navigated to https://x/ over <cdp>")
    assert browser_use.stopped == [first, second] and not site.playwright.running


def test_a_browser_browser_use_ended_is_said_to_have_ended_and_the_next_call_gets_a_fresh_one(site, browser_use):
    with TestClient(site.create_app()) as client:
        mcp = _mcp(client)
        _call(client, mcp, "browser_navigate", url="https://pypi.org/")
        [(first, _)] = browser_use.created
        failed_while_running = _call(client, mcp, "browser_click", ref="e1")
        stopped_while_running = list(browser_use.stopped)
        browser_use.ended.add(first)  # its timeout_minutes ran out, say
        failed_once_ended = _call(client, mcp, "browser_click", ref="e1")
        stopped_once_ended = list(browser_use.stopped)
        again = _text(_call(client, mcp, "browser_navigate", url="https://pypi.org/"))
        [_, (second, _)] = browser_use.created

    # A tool's own error on a running browser is the agent's to handle, and the browser stays.
    assert failed_while_running["isError"] and "has been closed (clicking e1)" in failed_while_running["content"][0]["text"]
    assert stopped_while_running == []
    assert failed_once_ended["isError"] and "the browser ended during browser_click" in failed_once_ended["content"][0]["text"]
    assert stopped_once_ended == [first]
    assert again.startswith("navigated to https://pypi.org/ over <cdp>") and second != first


@pytest.mark.parametrize("answer", [503, "hang"], ids=["CDP discovery 503", "CDP discovery timeout"])
def test_a_browser_whose_check_says_nothing_stays_and_the_call_gets_its_own_error(site, browser_use, monkeypatch,
                                                                                  answer):
    monkeypatch.setattr(playwright_mcp, "PROBE_S", 0.5)
    with TestClient(site.create_app()) as client:
        mcp = _mcp(client)
        _call(client, mcp, "browser_navigate", url="https://pypi.org/")
        browser_use.probe = answer
        failed = _call(client, mcp, "browser_click", ref="e1")
        stopped = list(browser_use.stopped)
        again = _text(_call(client, mcp, "browser_navigate", url="https://pypi.org/"))

    # Neither a 5xx nor a probe that times out says Browser Use ended the browser: it stays, and the agent gets the
    # tool's own error.
    assert failed["isError"] and "has been closed (clicking e1)" in failed["content"][0]["text"]
    assert stopped == [] and len(browser_use.created) == 1 and again.startswith("navigated to https://pypi.org/")


def test_a_browser_whose_cdp_discovery_refuses_connections_has_ended(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(playwright_mcp, "PROBE_RETRY_S", 0.0)
    gone = SimpleNamespace(cdp_url=f"http://127.0.0.1:{_free_port()}")  # nothing listens there

    assert asyncio.run(playwright_mcp._ended(gone)) is True



def test_one_refused_connection_is_not_enough_to_call_the_browser_ended(monkeypatch):
    """A connection refused once and then answered is a blip: the browser is left running."""
    from types import SimpleNamespace

    answers = iter([ConnectionRefusedError(), httpx.Response(200, json={})])
    real = httpx.AsyncClient

    def reply(request):
        answer = next(answers)
        if isinstance(answer, BaseException):
            raise httpx.ConnectError("refused", request=request) from answer
        return answer

    monkeypatch.setattr(playwright_mcp, "PROBE_RETRY_S", 0.0)
    monkeypatch.setattr(playwright_mcp.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(reply), **kw))

    assert asyncio.run(playwright_mcp._ended(SimpleNamespace(cdp_url="http://browser.invalid"))) is False

def test_the_first_result_on_the_browser_that_replaced_one_playwright_mcp_lost_says_the_pages_are_gone(site, browser_use):
    with TestClient(site.create_app()) as client:
        mcp = _mcp(client)
        before = _call(client, mcp, "browser_navigate", url="https://pypi.org/")
        _call(client, mcp, "browser_crash")
        time.sleep(2)  # it exits 0.2 s after it answers
        replaced = _call(client, mcp, "browser_navigate", url="https://pypi.org/")
        after = _call(client, mcp, "browser_navigate", url="https://pypi.org/")

    assert [part["text"] for part in replaced["content"]] == [
        playwright_mcp.REPLACED, "navigated to https://pypi.org/ over <cdp>"]
    assert len(before["content"]) == len(after["content"]) == 1 and len(browser_use.created) == 2


def test_a_reset_sends_the_browsers_stop_before_waiting_on_playwright_mcp_to_exit(site, browser_use, monkeypatch):
    monkeypatch.setattr(playwright_mcp, "STOP_S", 2.0)
    monkeypatch.setattr(playwright_mcp, "command", lambda endpoint, port: [
        sys.executable, str(FAKE_PLAYWRIGHT_MCP), "--cdp-endpoint", endpoint, "--port", str(port), "--ignore-term"])
    with TestClient(site.create_app()) as client:
        _call(client, _mcp(client), "browser_navigate", url="https://pypi.org/")
        asked = time.monotonic()
        client.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset", "params": {}})
        answered = time.monotonic()

    [(browser, stopped_at)] = browser_use.stopped_at
    # Playwright MCP ignores SIGTERM, so the reset waits STOP_S for it; the stop was sent before that wait.
    assert answered - asked >= 2.0 and stopped_at - asked < 1.0 and browser == browser_use.created[0][0]


def test_a_browser_that_fails_to_start_is_refused_to_the_call_and_leaves_nothing(site, browser_use):
    browser_use.refuse = True
    with TestClient(site.create_app()) as client:
        ready = _ready(client)
        refused = _call(client, _mcp(client), "browser_navigate", url="https://pypi.org/")

    assert ready["ready"] is True
    assert refused["isError"] and "returned 429" in refused["content"][0]["text"]
    assert browser_use.created == [] and not site.playwright.running


def test_an_empty_key_fails_readiness(site, browser_use, monkeypatch):
    monkeypatch.setenv("AGENTENV_WEBSITE_SETTINGS", json.dumps({"api_url": browser_use.url}))
    with TestClient(browser_site.BuBrowser().create_app()) as client:
        ready = _ready(client)

    assert ready["ready"] is False and "api_key setting is empty" in " ".join(ready["problems"])


def test_playwright_mcp_failing_to_start_stops_the_browser_it_was_given(site, browser_use, monkeypatch):
    monkeypatch.setattr(playwright_mcp, "command", lambda endpoint, port: [sys.executable, "-c", "raise SystemExit(3)"])
    with TestClient(site.create_app()) as client:
        refused = _call(client, _mcp(client), "browser_navigate", url="https://pypi.org/")

    assert refused["isError"] and "exited at start with 3" in refused["content"][0]["text"]
    assert browser_use.stopped == [browser_use.created[0][0]] and not site.playwright.running


def test_a_reset_while_the_browser_is_still_being_created_stops_it_once_created(site, browser_use):
    browser_use.delay_s = 1.0
    with TestClient(site.create_app()) as client:
        mcp = _mcp(client)
        calling = threading.Thread(target=_call, args=(client, mcp, "browser_navigate"), kwargs={"url": "https://x/"})
        calling.start()
        while not browser_use.asked:
            time.sleep(0.01)
        client.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset", "params": {}})
        stopped_by_reset = list(browser_use.stopped)
        calling.join(10)

    assert stopped_by_reset == [browser_use.created[0][0]] and len(browser_use.created) == 1
    assert browser_use.stopped == stopped_by_reset


def test_teardown_closing_a_deployed_bu_browser_stops_its_sites_browser_for_good(site, browser_use, monkeypatch):
    """agent-env's teardown closes the env a run's record names (``teardown._close``), as every run's teardown does
    before it terminates sandboxes, here with the real site served on loopback behind the record's card. Only Docker
    is stood in for: the record's sandbox, which the env reattaches and terminates last. Browser Use's first two stop
    requests hang past their wait, as a slow API's would: the third still goes through before the sandbox goes."""
    terminated: list[tuple[str, list[str]]] = []

    class Sandbox:
        sandbox_id, mode = "sandbox-1", None

        async def terminate(self):
            terminated.append((self.sandbox_id, list(browser_use.stopped)))

    monkeypatch.setattr(site_cloud, "STOP_REQUEST_S", 0.5)
    monkeypatch.setattr(site_cloud, "STOP_RETRY_S", 0.1)

    async def reattach(self, env, deployed):
        return Sandbox()

    declared = tomllib.loads((BUNDLE / "envs" / "bu-browser" / "env.toml").read_text())
    [site_name] = declared["website_envs"]

    def bu_browser(cls, id, version=None):
        assert (id, version) == (declared["name"], 1)
        website = WebsiteEnv(id=site_name, version=1, server_docker_image_artifact=MagicMock(),
                             environment_name=site_name, domains=["bu-browser.localhost"])
        return bu_env.BuBrowserEnv(id=declared["name"], version=1, mcp_server_envs=[], website_envs=[website],
                                   browser_envs=[])

    monkeypatch.setattr(Env, "get", classmethod(bu_browser))
    monkeypatch.setattr(EnvironmentGatewayProvider, "_reattach", reattach)
    with _served(site.create_app()) as url, httpx.Client(base_url=url, timeout=30) as client:
        _call(client, _mcp(client), "browser_navigate", url="https://pypi.org/")
        card = client.get("/.well-known/agent-env.json").json()
        record = DeployedEnv.from_dict({
            "env_id": declared["name"], "env_version": 1, "env_provider_type": "gateway", "gateway_url": url,
            "sandbox_id": "sandbox-1", "sandbox_type": "local_docker",
            "environment_card_url": f"{url}/.well-known/agent-env.json",
            "environment_card": {"name": declared["name"], "protocolVersion": "1.0", "url": "/agentenv",
                                 "children_environments": [{**card, "environment_name": site_name}]}})
        [(browser, _)] = browser_use.created
        running = list(browser_use.stopped)

        browser_use.hanging_stops = 2
        asyncio.run(teardown._close(record, teardown.TERMINATE_TIMEOUT_SECONDS))
        # A call that comes after teardown's stop, before the container is removed, starts no browser that nothing
        # would stop; a reset_env's data/reset, by contrast, leaves the next call free to start one.
        late = _call(client, _mcp(client), "browser_navigate", url="https://pypi.org/")

        assert running == [] and browser_use.stopped == [browser] and not site.playwright.running
        assert late["isError"] and "the env has closed" in late["content"][0]["text"]
        assert len(browser_use.created) == 1 and browser_use.hung == 2 and terminated == [("sandbox-1", [browser])]


class FakeBrowserUse:
    """Browser Use's API as cloud.py calls it: create a browser, its CDP discovery, stop it; each call recorded."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.asked = 0
        self.created: list[tuple[str, dict]] = []
        self.stopped: list[str] = []
        self.keys: set[str] = set()
        self.stopped_at: list[tuple[str, float]] = []
        self.ended: set[str] = set()
        self.refuse = False
        self.delay_s = 0.0
        #: Stop requests still to hang (answered after the client gave up), and how many did.
        self.hanging_stops = 0
        self.hung = 0
        #: What CDP discovery answers, once set: that status, or with "hang" nothing in time.
        self.probe: int | str | None = None

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
            if request.path_params["id"] in self.ended:
                return JSONResponse({"detail": "not found"}, 404)
            if self.probe == "hang":
                await asyncio.sleep(5)
            elif self.probe:
                return JSONResponse({"detail": "unavailable"}, self.probe)
            return JSONResponse({"webSocketDebuggerUrl": f"ws://cdp.test/devtools/{request.path_params['id']}"})

        async def stop(request):
            if self.hanging_stops:
                self.hanging_stops -= 1
                self.hung += 1
                await asyncio.sleep(site_cloud.STOP_REQUEST_S * 2)
                return JSONResponse({"detail": "too slow"}, 504)
            self.stopped.append(request.path_params["id"])
            self.stopped_at.append((request.path_params["id"], time.monotonic()))
            return JSONResponse({"id": request.path_params["id"]})

        return Starlette(routes=[Route("/browsers", create, methods=["POST"]),
                                 Route("/cdp/{id}/json/version", version),
                                 Route("/browsers/{id}", stop, methods=["PATCH"])])


@pytest.fixture
def browser_use():
    fake = FakeBrowserUse("")
    with _served(fake.app()) as fake.url:
        yield fake


@contextlib.contextmanager
def _served(app):
    """``app`` served on a loopback port, lifespan included, until the block ends; yields its URL."""
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(30)


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


def _closing(monkeypatch, answer: BaseException | Awaitable | None = None) -> tuple[list[tuple], object]:
    """What closing does, recorded, and a deployment whose extension calls answer ``answer``."""
    calls: list[tuple] = []

    class Deployed:
        def __init__(self) -> None:
            self.timeouts: list[int] = []

        async def invoke(self, uri, method, *, environment_name, timeout=30):  # agent-env's default timeout
            calls.append(("invoke", uri, method, environment_name))
            self.timeouts.append(timeout)
            if isinstance(answer, BaseException):
                raise answer
            if answer is not None:
                await answer

    async def closing(self):
        calls.append(("close",))

    monkeypatch.setattr(MultiEnv, "close", closing)
    return calls, Deployed()


def _env() -> bu_env.BuBrowserEnv:
    site = WebsiteEnv(id="bu-browser-site", version=1, server_docker_image_artifact=MagicMock(),
                      environment_name="bu-browser-site", domains=["bu-browser.localhost"])
    return bu_env.BuBrowserEnv(id="bu-browser", version=1, mcp_server_envs=[], website_envs=[site], browser_envs=[])


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


def _mcp(client) -> dict:
    first = client.post("/mcp", headers=MCP_HEADERS, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}})
    headers = {**MCP_HEADERS, "mcp-session-id": first.headers["mcp-session-id"]}
    client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    return headers


def _rpc(client, headers: dict, method: str, params: dict | None = None) -> dict:
    response = client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": method,
                                                          "params": params or {}})
    text = response.text
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        text = [line[len("data:"):] for line in text.splitlines() if line.startswith("data:")][-1]
    return json.loads(text)["result"]


def _call(client, headers: dict, tool: str, **arguments) -> dict:
    return _rpc(client, headers, "tools/call", {"name": tool, "arguments": arguments})


def _text(result: dict) -> str:
    assert not result.get("isError"), result
    return result["content"][0]["text"]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
