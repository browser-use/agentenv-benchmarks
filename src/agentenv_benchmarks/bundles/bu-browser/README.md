A Browser Use Cloud browser as an env a trainer's model drives through Playwright MCP's tools, for RL on AgentEnv.

# bu-browser

Not BU Bench. BU Bench's 200 tasks (`bu-bench`) are an evaluation and must stay out of training. This bundle is the
browser env for RL tasks of your own, plus one example task.

## What it is

`bu-browser` (env type `bu_browser`, from `agentenv_benchmarks.bu_browser`) is a multi env with one website and no
browser of the run's own:

- `bu-browser-site` (`envs/bu-browser-site/browser_site.py`) starts one Browser Use Cloud browser when its server
  starts, attaches Playwright MCP 0.0.85 to it over CDP, and serves Playwright MCP's tools as its own MCP tools. The
  gateway serves them to the agent like any MCP env's tools.
- `browser_evaluate` is hidden until a task's tool access enables it. `browser_run_code_unsafe` is never served,
  because it would run the agent's code in the site's container.
- Text in tool results is scrubbed of the browser's CDP address, which is itself a credential.

In an OpenEnv episode (`agent_env.openenv.build_app`), the task's `deploy_agent` becomes a proxy, and the model gets
the tools of the envs in its `env_ids` plus `end_turn`. A task that lists `bu-browser` there gets this browser.

## Key

The site reads its key from `[settings] api_key = "secret:BROWSER_USE_API_KEY"`. AgentEnv resolves it from the
configured secret store (`[stores.secret]`: local, AWS or GCP) only when it writes the site's container. Only the site
process has it. Playwright MCP runs as another user with an allowlisted environment.

```bash
export BROWSER_USE_API_KEY=...   # the local secret store reads the environment; a cloud store holds it by this name
agent-env run bu-browser --eval example
```

## Cost and cleanup

One episode uses one browser. The browser is stopped when:

- the run tears down (`bu_browser` closes by sending `data/reset` to the site), which covers the final `end_turn`, an
  OpenEnv reset, an abandoned or timed-out episode, a closed session and server shutdown;
- a `reset_env` step runs;
- the site's server gets SIGTERM.

If none of these happens, for example because the server process is killed outright, Browser Use ends the browser
after `timeout_minutes` (30 by default). Keep that at or above the deploy time plus the task's turn timeouts.
`build_app` serves up to 8 episodes at once by default (`max_concurrent_envs`), so that is up to 8 browsers. Check
your Browser Use account's concurrency limit before you scale.

The site's container must reach `api.browser-use.com` and `*.cdp.browser-use.com`.

## Example task

`pypi-browser-use-wheel-sha256` asks for the SHA256 of `browser_use-1.0.3-py3-none-any.whl`, as shown on PyPI. That
file is immutable and was uploaded on 2026-11-12, so the answer comes from the page, not from memory. A regex
verifier scores the reply 0 or 1, and that score is the episode's reward. It runs on the live pypi.org, so it is a
smoke test, not a training set.
