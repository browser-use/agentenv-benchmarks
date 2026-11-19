"""A stand-in for Playwright MCP in tests/test_bu_browser.py: an MCP server on loopback ``--port`` whose tools say
which CDP endpoint it was given and which environment variables it has, as the real one would be given them."""
import json
import os
import sys

from mcp.server.fastmcp import FastMCP

endpoint = sys.argv[sys.argv.index("--cdp-endpoint") + 1]
server = FastMCP("fake-playwright-mcp", host="127.0.0.1", port=int(sys.argv[sys.argv.index("--port") + 1]))


@server.tool()
def browser_navigate(url: str) -> str:
    return f"navigated to {url} over {endpoint}"


@server.tool()
def browser_environment() -> str:
    return json.dumps(sorted(os.environ))


server.run(transport="streamable-http")
