"""Playwright MCP on a Browser Use Cloud browser: its tools, as the manifest the image's build wrote lists them, and one
browser, one Playwright MCP process attached to it over CDP and one MCP session on that process, started together and
stopped together.

Adapted from agentenv-framework 0.9.1573's page driver (src/agent_env/drivers/envs/page/playwright_mcp.py, Apache-2.0):
its process start and stop, its session held open by a task of its own, and its environment allowlist, cut down to
one session and given the cloud browser to own. Playwright MCP runs as the unprivileged user ``pw``, so nothing an
agent's tools reach can read the site's settings, which hold the Browser Use API key.

``python playwright_mcp.py --write-manifest`` is the image build's step: it checks the installed Playwright MCP is
VERSION and writes TOOLS from its own ``tools/list``, so the tools served can't drift from the version installed.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import cloud
import httpx
from agentenv_protocol import ForwardedTool
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

#: The Playwright MCP the image installs, the same pin as BU Bench's codex-browser-tools image.
VERSION = "0.0.85"
HERE = Path(__file__).resolve().parent
CLI = HERE / "node_modules" / "@playwright" / "mcp" / "cli.js"
TOOLS = HERE / "playwright-mcp-tools.json"
#: Runs the agent's JavaScript in the page: hidden until a task's tool access names it, as the page driver hides it.
HIDDEN_TOOLS = frozenset({"browser_evaluate"})
#: Runs the agent's code inside this container, beside the site that holds the key: never served.
DROPPED_TOOLS = frozenset({"browser_run_code_unsafe"})
#: Who Playwright MCP runs as, and all it keeps of the site's environment.
USER = "pw"
KEPT_ENVIRONMENT = ("PATH", "LANG", "LC_ALL", "TZ")
OUTPUT = Path("/home") / USER / "playwright-output"
LOG = Path(tempfile.gettempdir()) / "bu-browser-playwright-mcp.log"
START_S = 30.0
CALL_TIMEOUT_S = 300.0
STOP_S = 10.0
POLL_S = 0.1

logger = logging.getLogger(__name__)


@functools.cache
def manifest() -> list[dict]:
    return json.loads(TOOLS.read_text(encoding="utf-8"))["tools"]


def command(endpoint: str, port: int) -> list[str]:
    """Playwright MCP on loopback ``port``, attached to the browser at ``endpoint`` and never one of its own, as ``pw``
    with no groups, ended by the kernel if the site dies first. Its other flags are BU Bench's Playwright MCP arm's."""
    return ["setpriv", f"--reuid={USER}", f"--regid={USER}", "--clear-groups", "--pdeathsig", "TERM", "--",
            shutil.which("node") or "node", str(CLI), "--cdp-endpoint", endpoint, "--host", "127.0.0.1",
            "--port", str(port), "--output-dir", str(OUTPUT)]


def environment() -> dict[str, str]:
    return {**{name: os.environ[name] for name in KEPT_ENVIRONMENT if name in os.environ},
            "HOME": str(OUTPUT.parent), "USER": USER}


class PlaywrightMcp:
    """The browser ``start_browser`` starts (a ``cloud.CloudBrowser``), Playwright MCP on it and one session there.
    ``start`` brings all three up once, shared by every caller, and again after ``stop`` or once one has ended;
    ``stop`` ends them, a start under way included, and starts nothing, so a reset leaves no browser running."""

    def __init__(self, start_browser: Callable[[], Awaitable[Any]]) -> None:
        self._start_browser = start_browser
        self._lock = asyncio.Lock()
        self._starting: asyncio.Task | None = None
        self.browser: Any = None
        self._process: asyncio.subprocess.Process | None = None
        self._session: _Session | None = None
        self._orphans: list[asyncio.Task] = []

    def tools(self) -> list[ForwardedTool]:
        return [ForwardedTool.listed(t, functools.partial(self.call, t["name"]),
                                     hidden_by_default=t["name"] in HIDDEN_TOOLS)
                for t in manifest() if t["name"] not in DROPPED_TOOLS]

    @property
    def running(self) -> bool:
        return self.browser is not None or self._process is not None

    async def start(self) -> _Session:
        async with self._lock:
            if self._starting is None or self._starting.done() and not self._up():
                self._starting = asyncio.create_task(self._start())
            starting = self._starting
        return await asyncio.shield(starting)

    async def stop(self) -> None:
        async with self._lock:
            starting, self._starting = self._starting, None
            if starting is not None:
                starting.cancel()
                await asyncio.gather(starting, return_exceptions=True)
            await self._close()
            orphans, self._orphans = self._orphans, []
            await asyncio.gather(*orphans, return_exceptions=True)

    async def call(self, tool: str, /, **arguments: Any) -> CallToolResult:
        """``tool`` called on Playwright MCP, its text scrubbed of the browser's CDP address, or refused saying why."""
        try:
            held = await self.start()
        except asyncio.CancelledError:
            if (current := asyncio.current_task()) is not None and current.cancelling():
                raise
            return _refused(f"the browser was reset before {tool} ran")
        except Exception as e:  # noqa: BLE001 - the agent is told why, as the page driver tells it
            return _refused(f"the browser didn't start: {type(e).__name__}: {cloud.scrub(str(e), self._secrets())}")
        called = asyncio.ensure_future(held.session.call_tool(
            tool, arguments, read_timeout_seconds=timedelta(seconds=CALL_TIMEOUT_S)))
        try:
            await asyncio.wait({called, held.owner}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            called.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await called
            raise
        if not called.done():
            called.cancel()
            return _refused(f"the browser was reset or Playwright MCP ended during {tool}, which may have acted")
        try:
            result = called.result()
        except Exception as e:  # noqa: BLE001 - the call may have acted: refused, not retried
            return _refused(cloud.scrub(f"Playwright MCP didn't answer {tool}: {type(e).__name__}: {e}", held.secrets))
        return _scrubbed(result, held.secrets)

    def _up(self) -> bool:
        starting = self._starting
        return (starting is not None and starting.done() and not starting.cancelled() and starting.exception() is None
                and self.browser is not None and not self.browser.stopped
                and self._process is not None and self._process.returncode is None
                and self._session is not None and not self._session.owner.done())

    def _secrets(self) -> list[str]:
        return list(self.browser.secrets) if self.browser is not None else []

    async def _start(self) -> _Session:
        await self._close()  # what an ended start left: a stopped browser, an exited process
        browser_start = asyncio.ensure_future(self._start_browser())
        try:
            self.browser = await asyncio.shield(browser_start)
        except asyncio.CancelledError:
            # Cancelled mid-request, Browser Use may create the browser anyway: wait for it, then stop it.
            self._orphans.append(asyncio.ensure_future(_stop_once_started(browser_start)))
            raise
        try:
            port = _free_port()
            OUTPUT.mkdir(parents=True, exist_ok=True)
            log = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)  # it can name the CDP address
            try:
                self._process = await asyncio.create_subprocess_exec(
                    *command(self.browser.ws_url, port), cwd=OUTPUT, env=environment(),
                    stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=log)
            finally:
                os.close(log)
            await _listening(port, self._process)
            self._session = await _Session.open(f"http://127.0.0.1:{port}/mcp", self._secrets())
        except BaseException:
            await self._close()
            raise
        logger.info("cloud browser %s is up with Playwright MCP %s", self.browser.label, VERSION)
        return self._session

    async def _close(self) -> None:
        session, process, browser = self._session, self._process, self.browser
        self._session, self._process, self.browser = None, None, None
        try:
            if session is not None:
                session.close.set()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.gather(session.owner, return_exceptions=True), STOP_S)
            if process is not None and process.returncode is None:
                process.terminate()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(process.wait(), STOP_S)
                if process.returncode is None:
                    process.kill()
                    await process.wait()
        finally:
            if browser is not None:
                await browser.stop()
                logger.info("cloud browser %s stopped", browser.label)


async def _stop_once_started(browser_start: asyncio.Future) -> None:
    try:
        browser = await browser_start
    except Exception:  # noqa: BLE001 - a start that failed stopped what it had made (cloud.CloudBrowser.start)
        return
    await browser.stop()
    logger.info("cloud browser %s, started as it was stopped, stopped", browser.label)


class _Session:
    """One MCP session on Playwright MCP, held open by a task of its own until ``close`` is set, since mcp's client
    contexts are anyio task groups, which only the task that entered one can leave."""

    def __init__(self, url: str, secrets: list[str]) -> None:
        self.url, self.secrets = url, secrets
        self.close = asyncio.Event()
        self.session: ClientSession | None = None
        self.owner: asyncio.Task | None = None

    @classmethod
    async def open(cls, url: str, secrets: list[str]) -> _Session:
        held = cls(url, secrets)
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        held.owner = asyncio.create_task(held._own(ready))
        try:
            held.session = await asyncio.wait_for(ready, START_S)
        except BaseException:
            held.close.set()
            held.owner.cancel()
            raise
        return held

    async def _own(self, ready: asyncio.Future) -> None:
        try:
            timeout = httpx.Timeout(CALL_TIMEOUT_S, connect=START_S)
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as http, \
                    streamable_http_client(self.url, http_client=http) as (read, write, _), \
                    ClientSession(read, write) as session:
                await session.initialize()
                if ready.done():
                    return
                ready.set_result(session)
                await self.close.wait()
        except BaseException as e:
            if not ready.done():
                ready.set_exception(e if isinstance(e, Exception) else RuntimeError(repr(e)))
            elif not isinstance(e, asyncio.CancelledError):
                logger.warning("the session on Playwright MCP ended: %s", type(e).__name__)
            if not isinstance(e, Exception):
                raise


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def _listening(port: int, process: asyncio.subprocess.Process) -> None:
    deadline = asyncio.get_running_loop().time() + START_S
    while asyncio.get_running_loop().time() < deadline:
        if process.returncode is not None:
            raise RuntimeError(f"Playwright MCP exited at start with {process.returncode}; its log is {LOG}")
        with contextlib.suppress(OSError):
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            return
        await asyncio.sleep(POLL_S)
    raise RuntimeError(f"Playwright MCP didn't listen within {START_S:.0f} s")


def _scrubbed(result: CallToolResult, secrets: list[str]) -> CallToolResult:
    content = [part.model_copy(update={"text": cloud.scrub(part.text, secrets)}) if isinstance(part, TextContent)
               else part for part in result.content]
    structured = cloud.scrub_json(result.structuredContent, secrets) if result.structuredContent else None
    return result.model_copy(update={"content": content, "structuredContent": structured})


def _refused(message: str) -> CallToolResult:
    return CallToolResult(isError=True, content=[TextContent(type="text", text=message)])


def write_manifest() -> None:
    """Check the installed Playwright MCP is VERSION, and write TOOLS from its ``tools/list``, run attached to no
    browser, as agentenv-framework's scripts/upgrade_playwright_mcp.py lists it for the page driver."""
    node = shutil.which("node") or "node"
    said = subprocess.run([node, str(CLI), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    if said.split()[-1] != VERSION:
        raise SystemExit(f"Playwright MCP says {said!r}; this site serves {VERSION}'s tools")
    messages = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                            "clientInfo": {"name": "bu-browser-build", "version": "0"}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    # stdin stays open until the answer is read: Playwright MCP exits at its end, before a long answer is all written.
    with subprocess.Popen([node, str(CLI), "--cdp-endpoint", "http://127.0.0.1:9/"], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True) as server:
        timer = threading.Timer(120, server.kill)
        timer.start()
        try:
            server.stdin.write("".join(json.dumps(message) + "\n" for message in messages))
            server.stdin.flush()
            tools = next(answer["result"]["tools"] for answer in map(json.loads, server.stdout)
                         if answer.get("id") == 2)
        finally:
            timer.cancel()
            server.stdin.close()
    if not any(t["name"] == "browser_navigate" for t in tools):
        raise SystemExit(f"Playwright MCP {VERSION} lists no browser_navigate: {[t['name'] for t in tools]}")
    TOOLS.write_text(json.dumps({"version": VERSION, "tools": tools}, indent=1) + "\n", encoding="utf-8")
    print(f"Playwright MCP {VERSION}: {len(tools)} tools, serving "
          f"{len([t for t in tools if t['name'] not in DROPPED_TOOLS])}")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write-manifest"]:
        raise SystemExit("usage: python playwright_mcp.py --write-manifest")
    write_manifest()
