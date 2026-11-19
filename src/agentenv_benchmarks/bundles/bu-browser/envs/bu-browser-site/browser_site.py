"""A Browser Use Cloud browser as a website env that is an MCP server: Playwright MCP's tools, served on the browser.

The site starts one cloud browser as its server starts, and Playwright MCP on it, and is ready once both are up, so a
bad key or an account at its limit fails the deploy rather than the agent's first call. ``data/reset`` stops the
browser and starts nothing: the next tool call starts a fresh one. The server's shutdown, SIGTERM included, stops it
too, and the browser's own ``timeout_minutes`` ends it if nothing else does.

``api_key`` is a ``SecretStr`` setting, so its env.toml value is a ``secret:`` reference agentenv resolves only as it
writes the site's container; it stays in this process, and Playwright MCP, as another user, never sees it.
"""
from __future__ import annotations

import asyncio
import contextlib

import cloud
import playwright_mcp
from agentenv_protocol import add_data, get_data, reset_data
from agentenv_protocol.website import WebsiteEnvironment, ready_check, website
from pydantic import BaseModel, SecretStr
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route

API_URL = "https://api.browser-use.com/api/v3"


async def _page(request) -> HTMLResponse:
    return HTMLResponse("<!doctype html><title>bu-browser</title><p>A Browser Use Cloud browser, driven through this "
                        "site's MCP tools at /mcp.</p>")


@website(domains=["bu-browser.localhost"], clock="real", builtin_routes=False, ready_timeout_s=180)
class BuBrowser(WebsiteEnvironment):
    class Settings(BaseModel):
        api_key: SecretStr = SecretStr("")
        api_url: str = API_URL
        proxy_country_code: str = "us"
        #: Browser Use ends the browser after this long whatever happens here: at least the deploy plus a task's turns.
        timeout_minutes: int = 30

    #: The one page, at /: the browser is Browser Use's, reached only through the tools.
    app = Starlette(routes=[Route("/", _page)])

    def __init__(self) -> None:
        super().__init__()
        # cloud.py is BU Bench's, unchanged: it reads its API and key from these, here the site's settings.
        cloud.API = self.settings.api_url.rstrip("/")
        cloud.api_key = self.settings.api_key.get_secret_value
        self.playwright = playwright_mcp.PlaywrightMcp(self._start_browser)
        self._boot: asyncio.Task | None = None

    async def _start_browser(self) -> cloud.CloudBrowser:
        return await cloud.CloudBrowser.start(proxy_country_code=self.settings.proxy_country_code or None,
                                              timeout_minutes=self.settings.timeout_minutes, enable_recording=False)

    def forwarded_tools(self) -> list:
        return self.playwright.tools()

    def create_app(self):
        """The site's app, whose lifespan also starts the browser and stops it at shutdown, which uvicorn runs on
        SIGTERM before it exits."""
        app = super().create_app()
        served = app.router.lifespan_context

        @contextlib.asynccontextmanager
        async def lifespan(scope_app):
            async with served(scope_app):
                self._boot = asyncio.ensure_future(self.playwright.start())
                try:
                    yield
                finally:
                    await self.playwright.stop()

        app.router.lifespan_context = lifespan
        return app

    @ready_check
    async def browser_started(self) -> bool:
        """The browser the server started at boot is up with Playwright MCP on it, or was stopped since."""
        if not self.settings.api_key.get_secret_value():
            raise RuntimeError("the api_key setting is empty; set it to secret:BROWSER_USE_API_KEY")
        if self._boot is None or not self._boot.done():
            return False
        if not self._boot.cancelled() and (error := self._boot.exception()) is not None:
            raise RuntimeError(f"the browser didn't start: {type(error).__name__}: {error}")
        return True

    @reset_data
    async def stop_browser(self) -> None:
        """Stop the browser and Playwright MCP, starting nothing; the next tool call starts a fresh browser."""
        await self.playwright.stop()

    @get_data
    def export(self) -> list:
        """Nothing: what a task leaves is in its replies, and the browser is gone with its reset."""
        return []

    @add_data
    def load(self, parts: list) -> None:
        if parts:
            raise ValueError("bu-browser-site takes no data")


if __name__ == "__main__":
    BuBrowser().serve()
