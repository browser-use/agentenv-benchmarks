import asyncio
import base64
import dataclasses
import importlib.util
import io
import json
import re
from pathlib import Path

import httpx
import pytest

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
    mini = re.findall(r'"(bu2-\d+)"', (BUNDLE / "evals" / "mini-20.toml").read_text())
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


def test_output_files_open_only_the_judges_share_and_split_its_cap():
    """The judge reads the first 50 files by name, 600 kB of text between them; a file past them is never opened, and a
    big one is read in chunks, whatever its size, into the same head and tail."""
    big = "é" + "x" * 2_000_000 + "€"  # multi-byte characters at both ends and across every chunk boundary
    opened = []

    def opener(name, raw):
        return lambda: opened.append(name) or io.BytesIO(raw)

    named = {"a.csv": big.encode(), "b.bin": b"\xff\xfe", "c.txt": b"ok \xe2\x82",  # c ends inside a character
             **{f"d{n:02}.txt": b"d" for n in range(60)}}
    files = evidence.output_files([(name, opener(name, raw)) for name, raw in reversed(named.items())])

    limit = 600_000 // 50
    assert [f["name"] for f in files] == sorted(named)[:50] == sorted(opened)
    assert files[0] == {"name": "a.csv", "size": len(big.encode()), "clipped": True,
                        "text": big[:limit // 3] + "\n...[middle omitted]...\n" + big[-(limit - limit // 3):]}
    assert files[1]["text"] is None and files[2]["text"] is None and files[2]["size"] == 5
    assert files[3] == {"name": "d00.txt", "size": 1, "text": "d", "clipped": False}


BROWSER_ID = "0f6c1a52-9d0e-4b8a-a1f7-3c2d5e6f7a8b"
WS = f"wss://{BROWSER_ID}.cdp.browser-use.com/devtools/browser/11111111-2222-3333-4444-555555555555"


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

    async def capture_often():
        for n in range(50):
            await shots.capture(f"t{n}")

    asyncio.run(capture_often())
    assert shots.status["errors"] and BROWSER_ID not in shots.status["errors"][0]
    assert shots.status["captures"] == 50 and len(shots.status["errors"]) == 20  # the last 20, however long it runs


def test_the_final_frame_is_saved_after_the_capture_reached_its_limit(tmp_path):
    """The judge reads the frames in order: the last must be the page the agent left, however long the task ran."""
    shots = cloud.Screenshots(WS, tmp_path, limit=3)
    frames = iter(range(1000))

    async def send(method, params, session_id=None):
        return {"data": base64.b64encode(f"frame {next(frames)}".encode()).decode()}

    async def page():
        return "page-1"

    shots._ws, shots._sessions, shots._send, shots._page = object(), {"page-1": "s"}, send, page

    async def run():
        for n in range(5):
            await shots.capture(f"t{n}")
        return await shots.stop()

    status = asyncio.run(run())
    assert shots.saved == ["0001-t0.jpg", "0002-t1.jpg", "0003-final.jpg"] and status["saved"] == 3
    assert (tmp_path / "0003-final.jpg").read_bytes() == b"frame 2"


def test_the_capture_ends_at_the_browsers_time_limit(tmp_path):
    shots = cloud.Screenshots(WS, tmp_path, interval=0.01, seconds=0.05)
    taken = []

    async def capture(label):
        taken.append(label)

    shots.capture = capture

    async def run():
        shots.start()
        await asyncio.sleep(0.5)
        return shots._loop.done()

    assert asyncio.run(run()) and 0 < len(taken) < 10


def codex_tools_agent(monkeypatch):
    """The Codex arms' agent module, imported over AgentEnv's own Codex agent sources (the image's base). Those are
    AgentEnv's private layout, not its API: a version that moves or renames them skips these tests, saying so."""
    import sys

    import agent_env

    codex = Path(agent_env.__file__).parent / "agents" / "agents" / "codex"
    if not (codex / "a2a_server.py").is_file():
        pytest.skip(f"this AgentEnv has no Codex agent sources at {codex}, which the Codex arms' image builds on")
    monkeypatch.syspath_prepend(str(codex))
    monkeypatch.syspath_prepend(str(BUNDLE / "artifacts" / "codex-browser-tools"))
    monkeypatch.delitem(sys.modules, "codex_tools_agent", raising=False)
    try:
        return importlib.import_module("codex_tools_agent")
    except ImportError as error:
        ours = {path.stem for path in codex.glob("*.py")}
        if error.name in ours or str(error.name).startswith("agentenv_protocol"):
            pytest.skip(f"this AgentEnv's Codex agent sources lack what the Codex arms' agent imports: {error}")
        raise


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


def test_codex_turn_result_and_bridge_logs_never_carry_the_browser(monkeypatch, caplog):
    """The framework sends the result's warnings in the turn's data and its error as the failure, and Codex's bridge logs
    both before run_codex returns them; a tool's error text can carry the address."""
    import logging

    agent = codex_tools_agent(monkeypatch)
    import codex_bridge

    async def fake_codex(**kwargs):
        logging.getLogger("codex_bridge").info("codex CLI emitted %d non-fatal notice(s): %s", 1, f"cannot reach {WS}")
        return codex_bridge.CodexResult(response=f"FINAL ANSWER: {WS}", tool_call_count=1, session_id="s",
                                        error=f"turn failed at {WS}", error_detail=WS,
                                        warnings=[f'{{"message": "connect {BROWSER_ID}.cdp.browser-use.com refused"}}'],
                                        trajectory=[{"type": "error", "message": WS}])

    monkeypatch.setattr(agent, "_run_codex", fake_codex)
    monkeypatch.setattr(agent, "_cdp_url", WS)
    with caplog.at_level(logging.INFO, logger="codex_bridge"):
        result = asyncio.run(agent.run_codex(prompt="task"))

    assert BROWSER_ID not in json.dumps(dataclasses.asdict(result)) and caplog.records
    assert BROWSER_ID not in caplog.text and result.response == "FINAL ANSWER: <cdp>" and result.tool_call_count == 1


def test_repository_carries_no_decrypted_task_text():
    """Task text, rubrics and the canary live only in the encrypted benchmark file; no other file may carry them."""
    data = tasks._benchmark()
    needles = {data["canary"]: "canary"} if data.get("canary") else {}
    for task in data["tasks"]:
        needles[task["canary"]] = f"{task['id']} canary"
        for field in ("task", "rubric", "summary", "title"):
            text = " ".join(str(task.get(field) or "").split())
            if len(text) >= 20:  # every 40 characters, and the tail, so any 80 copied characters hold a whole needle
                for needle in {text[i:i + 40] for i in range(0, len(text) - 39, 40)} | {text[-40:]}:
                    needles[needle] = f"{task['id']} {field}"
    long = {needle: where for needle, where in needles.items() if len(needle) == 40}
    short = {needle: where for needle, where in needles.items() if len(needle) != 40}
    skip = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "node_modules"}
    # The judge's files are the public benchmark's own, checked byte for byte against it as it runs; its prompt shares
    # the rubrics' template text.
    public = {tasks.JUDGE_DIR / name for name in json.loads((tasks.JUDGE_DIR / "SOURCE.json").read_text())["files"]}
    leaks = []
    # Every file this benchmark's change adds or touches.
    ours = [ROOT / "src" / "agentenv_benchmarks" / "bu_bench", BUNDLE, TOOLS, ROOT / "tests",
            ROOT / "src" / "agentenv_benchmarks" / "bu_browser", BUNDLE.parent / "bu-browser", ROOT / "README.md",
            ROOT / "NOTICE"]
    for path in (p for top in ours for p in ([top] if top.is_file() else top.rglob("*"))):
        if path.is_dir() or skip & set(path.relative_to(ROOT).parts) or path in public:
            continue
        text = " ".join(path.read_bytes().decode("utf-8", "replace").split())
        windows = {text[i:i + 40] for i in range(len(text) - 39)}
        leaks += [f"{path.relative_to(ROOT)}: {long[needle]}" for needle in windows & long.keys()]
        leaks += [f"{path.relative_to(ROOT)}: {where}" for needle, where in short.items() if needle in text]
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
        committed = {p.name: p.read_text() for p in (BUNDLE / kind).iterdir()}
        assert written == committed, kind


class FakeBrowserUseApi:
    """cloud.request as Browser Use answers it: every call recorded, the first ``failing_stops`` stops a 503."""

    def __init__(self, failing_stops: int = 0) -> None:
        self.calls: list[tuple[str, str]] = []
        self.failing_stops = failing_stops

    async def __call__(self, method, path, body=None, timeout=60):
        self.calls.append((method, path))
        if method == "PATCH" and self.failing_stops:
            self.failing_stops -= 1
            raise RuntimeError("Browser Use API PATCH /browsers returned 503")
        return {}

    @property
    def stops(self) -> int:
        return sum(method == "PATCH" for method, _ in self.calls)


def verifying(monkeypatch, tmp_path, judgment, failing_stops=0, evidence_dir=None, handle="h1", artifacts=None):
    """A live browser as bu_cloud_browser leaves it, the agent's answered turn, its delivered files (``artifacts``,
    name: bytes, or None for one whose download fails) in a faked object store, Browser Use's API faked, the judge
    answering ``judgment``, and the verifier step with the context it runs in."""
    from types import SimpleNamespace

    import agent_env.config
    from agent_env.task_step.context import PromptResponse

    objects = {"mem://trajectory": b"[]", **{f"mem://out/{name}": raw for name, raw in (artifacts or {}).items()}}

    class Store:
        def open(self, url):
            if objects[url] is None:
                raise OSError("object store unavailable")
            return io.BytesIO(objects[url])

        def get(self, url):
            return self.open(url).read()

    monkeypatch.setattr(agent_env.config, "get_config",
                        lambda: SimpleNamespace(get_object_store_at=lambda url: Store()))
    api = FakeBrowserUseApi(failing_stops)
    monkeypatch.setattr(cloud, "request", api)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda seconds: real_sleep(0))
    if evidence_dir is None:
        monkeypatch.delenv(steps.EVIDENCE_DIR_ENV, raising=False)
    else:
        monkeypatch.setenv(steps.EVIDENCE_DIR_ENV, str(evidence_dir))
    judged = []

    def judge(self, workspace):
        judged.append(workspace)
        return judgment

    monkeypatch.setattr(steps.BuFindingsVerifierTaskStep, "_judge", judge)
    browser = cloud.CloudBrowser(id=BROWSER_ID, ws_url=WS, cdp_url=f"https://{BROWSER_ID}.cdp.browser-use.com")
    shots = cloud.Screenshots("ws://127.0.0.1:9/devtools/browser/x", tmp_path / "shots")  # nothing answers there
    monkeypatch.setattr(steps, "_LIVE", {handle: steps._Live(browser, shots, tmp_path / "shots", 0.0)})
    answered = PromptResponse(prompt_id="solve", response="FINAL ANSWER: done",
                              agent_trajectory_object_url="mem://trajectory")
    context = SimpleNamespace(metadata={steps.STATE_KEY: {"handle": handle, "browser": browser.label},
                                        "run_group_id": "group-1",
                                        "artifacts": {name: f"mem://out/{name}" for name in artifacts or {}}},
                              prompt_responses=[answered], deployed_agents=[], instance_id=None)
    return steps.BuFindingsVerifierTaskStep("grade", 1, "bu2-140", "solve"), context, api, judged


SCORED = {"status": "scored", "score": 0.75, "rubric": {"a": True, "b": False}, "reason": "ok"}


def test_a_graded_task_stops_its_browser_once_and_drops_its_temporary_evidence(monkeypatch, tmp_path):
    step, context, api, judged = verifying(monkeypatch, tmp_path, SCORED)

    asyncio.run(step.execute(context))

    verification = context.metadata["verifications"]["findings"]
    assert verification["score"] == 0.75 and api.stops == 1 and steps._LIVE == {}
    # The workspace held the decrypted task and its rubric.
    assert not judged[0].exists() and verification["evidence_dir"] is None


def test_a_judge_error_stops_the_browser_once_records_no_score_and_keeps_the_evidence(monkeypatch, tmp_path):
    step, context, api, judged = verifying(monkeypatch, tmp_path, {"status": "error", "reason": "judge 500"})

    with pytest.raises(RuntimeError, match="did not score"):
        asyncio.run(step.execute(context))

    assert "score" not in context.metadata["verifications"]["findings"] and api.stops == 1 and steps._LIVE == {}
    assert (judged[0] / "result.json").exists()


def test_an_evidence_folder_that_cant_be_written_still_stops_the_browser(monkeypatch, tmp_path):
    (tmp_path / "not-a-folder").write_text("")
    step, context, api, judged = verifying(monkeypatch, tmp_path, SCORED, evidence_dir=tmp_path / "not-a-folder")

    with pytest.raises(OSError):
        asyncio.run(step.execute(context))

    assert api.stops == 1 and steps._LIVE == {} and judged == [] and "verifications" not in context.metadata


@pytest.mark.parametrize("failing_stops, stops, left", [(1, 2, False), (3, 3, True)],
                         ids=["one 503, then stopped", "every stop fails"])
def test_a_failed_stop_is_retried_and_a_browser_never_stopped_is_left_to_the_exit_hook(
        monkeypatch, tmp_path, failing_stops, stops, left):
    step, context, api, _ = verifying(monkeypatch, tmp_path, SCORED, failing_stops=failing_stops)
    patched = []
    monkeypatch.setattr(steps.httpx, "patch", lambda url, **kwargs: patched.append((url, kwargs["json"])))
    monkeypatch.setattr(cloud, "api_key", lambda: "k")

    asyncio.run(step.execute(context))
    steps._stop_leftovers()

    assert api.stops == stops and bool(steps._LIVE) is left
    assert patched == ([(f"{cloud.API}/browsers/{BROWSER_ID}", {"action": "stop"})] if left else [])


def test_a_task_whose_browser_is_gone_is_not_judged(monkeypatch, tmp_path):
    """Its screenshots went with it: a judgment would score the agent on no evidence."""
    step, context, api, judged = verifying(monkeypatch, tmp_path, SCORED)
    steps._LIVE.clear()

    with pytest.raises(RuntimeError, match="no live browser"):
        asyncio.run(step.execute(context))

    assert judged == [] and api.calls == [] and "verifications" not in context.metadata


def test_a_turn_that_left_no_response_is_not_judged(monkeypatch, tmp_path):
    """A turn that timed out, or whose agent died or never got the prompt, leaves no response: judging an empty reply
    would score the agent on what the run lost."""
    step, context, api, judged = verifying(monkeypatch, tmp_path, SCORED)
    context.prompt_responses = []

    with pytest.raises(RuntimeError, match="left no response"):
        asyncio.run(step.execute(context))

    assert judged == [] and api.stops == 1 and steps._LIVE == {} and "verifications" not in context.metadata


def test_a_delivered_file_that_cant_be_read_is_not_judged_as_missing(monkeypatch, tmp_path):
    step, context, api, judged = verifying(monkeypatch, tmp_path, SCORED, artifacts={"a.csv": b"1,2", "b.csv": None})

    with pytest.raises(RuntimeError, match="reading a delivered file failed"):
        asyncio.run(step.execute(context))

    assert judged == [] and api.stops == 1 and "verifications" not in context.metadata


@pytest.mark.parametrize("failed, judged_anyway", [("collect_artifacts", False), ("prompt_agent", True)],
                         ids=["collect failed", "the turn failed"])
def test_a_failed_collect_is_not_judged_as_no_files_and_a_failed_turn_is_judged_on_its_response(
        monkeypatch, tmp_path, failed, judged_anyway):
    """Both steps run with fail_task_on_error=false. A collect that failed (its sandbox died, an upload failed) kept only
    the files it uploaded first; a turn that failed left the response it is judged on, as the published runs were."""
    step, context, api, judged = verifying(monkeypatch, tmp_path, SCORED)
    grade = json.loads((BUNDLE / "tasks" / "bu2-140.json").read_text())[-1]
    step = steps.BuFindingsVerifierTaskStep.from_dict({**grade, "version": 1})
    failed_id = {"collect_artifacts": "collect", "prompt_agent": "solve"}[failed]
    context.metadata["failed_steps"] = [{"step_id": failed_id, "step_type": failed, "error": "sandbox gone",
                                         "error_type": "RuntimeError", "is_fatal": False}]

    if judged_anyway:
        asyncio.run(step.execute(context))
        assert context.metadata["verifications"]["findings"]["score"] == 0.75
    else:
        with pytest.raises(RuntimeError, match="step collect, which this verifier depends on, failed") as raised:
            asyncio.run(step.execute(context))
        assert judged == [] and "verifications" not in context.metadata
        assert Path(re.search(r"kept in (\S+)\)", str(raised.value)).group(1)).is_dir()
    assert api.stops == 1


def test_a_turn_whose_trajectory_cant_be_read_is_not_judged_as_zero_steps(monkeypatch, tmp_path):
    """Neither the stored copy nor the agent (gone here) gives the trajectory: the judge would see no steps at all."""
    step, context, api, judged = verifying(monkeypatch, tmp_path, SCORED)
    context.prompt_responses[0].agent_trajectory_object_url = "mem://lost"

    with pytest.raises(RuntimeError, match="neither the stored trajectory nor the agent's could be read"):
        asyncio.run(step.execute(context))

    assert judged == [] and api.stops == 1 and "verifications" not in context.metadata


def test_the_verifiers_object_store_reads_leave_other_tasks_running(monkeypatch, tmp_path):
    """The runner runs every task's steps on one event loop: a slow store read must not stall the others."""
    import time
    from types import SimpleNamespace

    import agent_env.config

    step, context, _, _ = verifying(monkeypatch, tmp_path, SCORED, artifacts={"a.csv": b"1,2"})

    reads = []

    class SlowStore:
        def open(self, url):
            started = time.monotonic()
            time.sleep(0.3)
            reads.append((started, time.monotonic()))
            return io.BytesIO(b"[]" if url == "mem://trajectory" else b"1,2")

        def get(self, url):
            return self.open(url).read()

    monkeypatch.setattr(agent_env.config, "get_config",
                        lambda: SimpleNamespace(get_object_store_at=lambda url: SlowStore()))
    ticks = []

    async def other_task():
        while True:
            ticks.append(time.monotonic())
            await asyncio.sleep(0)

    async def run():
        other = asyncio.ensure_future(other_task())
        await step.execute(context)
        other.cancel()

    asyncio.run(run())
    assert len(reads) == 2 and all(any(start < tick < end for tick in ticks) for start, end in reads)


def test_two_runs_of_a_task_in_one_run_group_keep_their_own_evidence(monkeypatch, tmp_path):
    """An explorer batch runs a task several times under one run group id."""
    for n, handle in enumerate(["a" * 32, "b" * 32]):
        step, context, _, _ = verifying(monkeypatch, tmp_path / str(n), SCORED, evidence_dir=tmp_path / "evidence",
                                        handle=handle)
        asyncio.run(step.execute(context))
    assert len(list((tmp_path / "evidence").iterdir())) == 2
    assert all((folder / "result.json").exists() for folder in (tmp_path / "evidence").iterdir())


@pytest.mark.parametrize("answers, runs, status", [(["error", "scored"], 2, "scored"),
                                                   (["error", "error", "error", "scored"], 3, "error")],
                         ids=["one transient error", "errs every time"])
def test_a_judge_error_is_asked_again_before_the_task_is_given_up(monkeypatch, tmp_path, answers, runs, status):
    import subprocess

    ran = []

    def judge_run(args, *, env, **kwargs):
        ran.append(args)
        Path(env["EVAL_JUDGMENT_PATH"]).write_text(json.dumps({"status": answers[len(ran) - 1], "score": 0.5}))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(steps.subprocess, "run", judge_run)
    monkeypatch.setattr(steps, "JUDGE_RETRY_S", 0)
    monkeypatch.setenv("BU_BENCH_JUDGE_API_KEY", "k")
    step = steps.BuFindingsVerifierTaskStep("grade", 1, "bu2-140", "solve")

    assert step._judge(tmp_path)["status"] == status and len(ran) == runs
@pytest.mark.parametrize("answer, failing_stops, left", [(401, 0, False), (500, 0, False), (500, 3, True)],
                         ids=["handoff 401", "handoff 500", "handoff 500, every stop fails"])
def test_a_failed_browser_handoff_stops_the_browser_or_leaves_it_to_the_exit_hook(monkeypatch, answer, failing_stops,
                                                                                  left):
    from types import SimpleNamespace

    from agent_env.a2a_agent.a2a_agent import A2AAgent

    real = httpx.AsyncClient
    monkeypatch.setattr(steps.httpx, "AsyncClient", lambda **kw: real(
        transport=httpx.MockTransport(lambda request: httpx.Response(answer, request=request)), **kw))
    browser = cloud.CloudBrowser(id=BROWSER_ID, ws_url=WS, cdp_url=f"https://{BROWSER_ID}.cdp.browser-use.com")

    async def start(**kwargs):
        return browser

    monkeypatch.setattr(steps.cloud.CloudBrowser, "start", start)
    api = FakeBrowserUseApi(failing_stops)
    monkeypatch.setattr(cloud, "request", api)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda seconds: real_sleep(0))
    monkeypatch.setattr(steps, "_LIVE", {})
    patched = []
    monkeypatch.setattr(steps.httpx, "patch", lambda url, **kwargs: patched.append((url, kwargs["json"])))
    monkeypatch.setattr(cloud, "api_key", lambda: "k")
    card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_CDP_CONFIG, "params": {"endpoint": "/ext/cdp"}}]}}
    step = steps.BuCloudBrowserTaskStep("b", 1, "bu2-140")
    agent = SimpleNamespace(agent_name=step.agent_name, a2a_url="http://agent", a2a_card=card, control_headers={})

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(step.execute(SimpleNamespace(metadata={}, deployed_agents=[agent])))
    steps._stop_leftovers()

    assert api.stops == (cloud.STOP_ATTEMPTS if left else 1) and bool(steps._LIVE) is left
    assert patched == ([(f"{cloud.API}/browsers/{BROWSER_ID}", {"action": "stop"})] if left else [])


def test_httpx_request_logs_name_no_browser_and_no_signed_link(monkeypatch, caplog):
    """httpx logs each request's URL at INFO, which agent-env up logs at: a browser's id or CDP host there reaches it."""
    import logging

    recording = "https://recordings.example/r.mp4?X-Amz-Signature=s3cr3t"

    def reply(request):
        if request.method == "POST":
            return httpx.Response(200, json={"id": BROWSER_ID, "cdpUrl": f"https://{BROWSER_ID}.cdp.browser-use.com"})
        if request.url.path == "/json/version":
            return httpx.Response(200, json={"webSocketDebuggerUrl": WS})
        return httpx.Response(200, json={"id": BROWSER_ID, "recordingUrl": recording})

    real = httpx.AsyncClient
    monkeypatch.setattr(cloud.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(reply), **kw))
    monkeypatch.setenv(cloud.KEY_NAME, "k")

    async def browser_lifetime():
        browser = await cloud.CloudBrowser.start()
        await browser.stop()
        session = await browser.session()
        async with httpx.AsyncClient() as client:
            await client.get(cloud.recording_urls(session)[0])

    with caplog.at_level(logging.INFO, logger="httpx"):
        asyncio.run(browser_lifetime())

    assert len([record for record in caplog.records if record.name == "httpx"]) == 5
    assert BROWSER_ID not in caplog.text and "cdp.browser-use.com" not in caplog.text and "s3cr3t" not in caplog.text
