"""The two task steps BU Bench adds to AgentEnv.

``bu_cloud_browser`` starts a Browser Use Cloud browser, hands its CDP address to the agent through the agent's
``cdp-config/v1`` (as the built-in ``browser_connect_cdp`` step does for AgentEnv's own Chromium), starts the
screenshot capture the judge reads, and puts the task's instruction into the seed value ``bu_task`` the prompt names.

``bu_findings_verifier`` stops the capture and the browser (always, even when the agent failed), keeps the
recording, builds the judge evidence from the agent's reply, its trajectory and the delivered files (scrubbed of the
browser's CDP address, which is its credential), and runs the frozen BU Bench V2.1 findings judge unchanged, as a
subprocess. The score is the judge's continuous one, 0 to 1. A
judge error fails the step and records no score: it is never a zero.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import httpx
from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep

from agentenv_benchmarks.bu_bench import cloud, evidence, tasks

logger = logging.getLogger(__name__)

STATE_KEY = "bu_cloud_browser"
SEED_KEY = "bu_task"
JUDGE_RUN = tasks.JUDGE_DIR / "run.py"
JUDGE_TIMEOUT_S = 1800
#: Where each task's judge inputs and outputs are kept after the run; a temporary folder otherwise.
EVIDENCE_DIR_ENV = "BU_BENCH_EVIDENCE_DIR"
JUDGE_KEY_NAMES = ("BU_BENCH_JUDGE_API_KEY", "OPENAI_API_KEY")
#: Not deliverables, as the browser-use eval adapter skipped them: browser-use's own REPL cells and logs, installs.
SKIPPED_PARTS = {".browser-use", "node_modules", ".git", "screenshots"}


@dataclass
class _Live:
    browser: cloud.CloudBrowser
    screenshots: cloud.Screenshots
    shots_dir: Path
    started: float


#: Browsers this process started and has not stopped, by handle. Both steps run in the runner's process.
_LIVE: dict[str, _Live] = {}


@atexit.register
def _stop_leftovers() -> None:
    """A run that ends before its verifier (Ctrl-C, a crash) still stops its browsers."""
    for live in list(_LIVE.values()):
        try:
            httpx.patch(f"{cloud.API}/browsers/{live.browser.id}", json={"action": "stop"}, timeout=20,
                        headers={"X-Browser-Use-API-Key": cloud.api_key()})
        except Exception:  # noqa: BLE001, S110 - the browser's own timeout ends it
            pass


def _control_headers(agent) -> dict[str, str]:
    """The agent's control token header: agents on agentenv-framework-protocol >= 0.1.582 answer 401 on extension
    routes without it. AgentEnv 0.9.1536 has no token, so nothing is sent there."""
    return dict(getattr(agent, "control_headers", None) or {})


def _agent(context: TaskStepContext, agent_name: str):
    agent = next((a for a in context.deployed_agents if a.agent_name == agent_name), None)
    if agent is None:
        raise RuntimeError(f"agent {agent_name!r} isn't deployed in this run")
    return agent


class BuCloudBrowserTaskStep(TaskStep):
    type: ClassVar[str] = "bu_cloud_browser"

    def __init__(self, id: str, version: int | None, task_id: str, agent_name: str | None = None,
                 name: str = "bu-cloud", proxy_country_code: str | None = "us", timeout_minutes: int = 70,
                 enable_recording: bool = True, screenshot_interval_seconds: float = 2.0,
                 depends_on: list | None = None, fail_task_on_error: bool = True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.task_id = task_id
        self.agent_name = agent_name or TaskStep.DEFAULT_AGENT_NAME
        self.name = name
        self.proxy_country_code = proxy_country_code
        self.timeout_minutes = timeout_minutes
        self.enable_recording = enable_recording
        self.screenshot_interval_seconds = screenshot_interval_seconds

    def config_problems(self) -> list[str]:
        problems = []
        if not isinstance(self.timeout_minutes, int) or not 1 <= self.timeout_minutes <= 240:
            problems.append(f"step {self.id!r}: timeout_minutes must be 1..240")
        return problems

    def preflight(self) -> list[str]:
        try:
            tasks.instruction(self.task_id)
        except Exception as error:  # noqa: BLE001
            return [f"step {self.id!r}: {error}"]
        return []

    def to_dict(self) -> dict:
        return {**super().to_dict(), "task_id": self.task_id, "agent_name": self.agent_name, "name": self.name,
                "proxy_country_code": self.proxy_country_code, "timeout_minutes": self.timeout_minutes,
                "enable_recording": self.enable_recording,
                "screenshot_interval_seconds": self.screenshot_interval_seconds}

    @classmethod
    def from_dict(cls, data: dict) -> BuCloudBrowserTaskStep:
        return cls(**cls._base_from_dict(data), task_id=data["task_id"], agent_name=data.get("agent_name"),
                   name=data.get("name", "bu-cloud"), proxy_country_code=data.get("proxy_country_code", "us"),
                   timeout_minutes=data.get("timeout_minutes", 70),
                   enable_recording=data.get("enable_recording", True),
                   screenshot_interval_seconds=data.get("screenshot_interval_seconds", 2.0))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        context.metadata.setdefault("seed", {})[SEED_KEY] = tasks.instruction(self.task_id)
        agent = _agent(context, self.agent_name)
        cdp_config = A2AAgent.find_extension(agent.a2a_card or {}, A2AAgent.EXT_CDP_CONFIG)
        if cdp_config is None or not agent.a2a_url:
            raise RuntimeError(f"agent {agent.agent_name!r} can't take a browser: its card has no "
                               f"{A2AAgent.EXT_CDP_CONFIG}")
        browser = await cloud.CloudBrowser.start(proxy_country_code=self.proxy_country_code,
                                                 timeout_minutes=self.timeout_minutes,
                                                 enable_recording=self.enable_recording)
        handle = uuid.uuid4().hex
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                added = await client.post(agent.a2a_url + cdp_config["params"]["endpoint"],
                                          headers=_control_headers(agent), json={
                    "name": self.name, "http_url": browser.cdp_url, "ws_url": browser.ws_url,
                    "metadata": {"browser": "browser-use-cloud", "browser_label": browser.label}})
            added.raise_for_status()
            shots_dir = Path(tempfile.mkdtemp(prefix=f"bu-shots-{self.task_id}-"))
            screenshots = cloud.Screenshots(browser.ws_url, shots_dir, interval=self.screenshot_interval_seconds)
            screenshots.start()
            _LIVE[handle] = _Live(browser, screenshots, shots_dir, time.time())
        except BaseException:
            await browser.stop()
            raise
        context.metadata[STATE_KEY] = {"handle": handle, "browser": browser.label, "task_id": self.task_id,
                                       "started_at": time.time()}
        logger.info(f"agent {agent.agent_name!r} has Browser Use Cloud browser {browser.label} as {self.name!r}")
        return context


class BuFindingsVerifierTaskStep(TaskStep):
    type: ClassVar[str] = "bu_findings_verifier"

    def __init__(self, id: str, version: int | None, task_id: str, prompt_id: str,
                 agent_name: str | None = None, verifier_id: str = "findings", judge_model: str = "gpt-5.6-luna",
                 reasoning_effort: str = "xhigh", keep_recording: bool = True,
                 depends_on: list | None = None, fail_task_on_error: bool = True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.task_id = task_id
        self.prompt_id = prompt_id
        self.agent_name = agent_name or TaskStep.DEFAULT_AGENT_NAME
        self.verifier_id = verifier_id
        self.judge_model = judge_model
        self.reasoning_effort = reasoning_effort
        self.keep_recording = keep_recording

    def to_dict(self) -> dict:
        return {**super().to_dict(), "task_id": self.task_id, "prompt_id": self.prompt_id,
                "agent_name": self.agent_name, "verifier_id": self.verifier_id, "judge_model": self.judge_model,
                "reasoning_effort": self.reasoning_effort, "keep_recording": self.keep_recording}

    @classmethod
    def from_dict(cls, data: dict) -> BuFindingsVerifierTaskStep:
        return cls(**cls._base_from_dict(data), task_id=data["task_id"], prompt_id=data["prompt_id"],
                   agent_name=data.get("agent_name"), verifier_id=data.get("verifier_id", "findings"),
                   judge_model=data.get("judge_model", "gpt-5.6-luna"),
                   reasoning_effort=data.get("reasoning_effort", "xhigh"),
                   keep_recording=data.get("keep_recording", True))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        state = context.metadata.get(STATE_KEY) or {}
        live = _LIVE.pop(state.get("handle", ""), None)
        workspace = self._workspace(context)
        report: dict[str, Any] = {"task_id": self.task_id, "browser": state.get("browser")}
        secrets = live.browser.secrets if live is not None else []
        session: dict[str, Any] = {}
        shot_names: list[str] = []
        try:
            if live is not None:
                report["screenshots"] = await live.screenshots.stop()
                shot_names = list(live.screenshots.saved)
                (workspace / "judge_screenshots").mkdir(exist_ok=True)
                for name in shot_names:
                    shutil.copy2(live.shots_dir / name, workspace / "judge_screenshots" / name)
        finally:
            if live is not None:
                await live.browser.stop()
                report["browser_seconds"] = round(time.time() - live.started, 1)
                try:
                    await asyncio.sleep(2)  # costs and the recording link appear once the session has stopped
                    session = await live.browser.session()
                except Exception as error:  # noqa: BLE001 - cost lookup never fails the task
                    logger.warning("reading cloud browser session %s failed (%s)", live.browser.label,
                                   cloud.scrub(str(error), secrets))
                shutil.rmtree(live.shots_dir, ignore_errors=True)
        report["cloud"] = cloud.costs(session)
        if self.keep_recording:
            report["recording"] = await self._keep_recording(session, context)

        response = next((r for r in reversed(context.prompt_responses) if r.prompt_id == self.prompt_id), None)
        final_output = cloud.scrub((response.response if response else "") or "", secrets)
        report["agent_error"] = ({"type": response.error_type, "code": response.error_code,
                                  "message": cloud.scrub(response.error_message or "", secrets)[:2000]}
                                 if response and response.error_type else None)
        trajectory, report["trajectory_source"] = await self._trajectory(context, response)
        trajectory = cloud.scrub_json(trajectory, secrets)
        (workspace / "trajectory.json").write_text(json.dumps(trajectory, default=str))
        fmt, steps, usage = evidence.steps_and_usage(trajectory)
        files = cloud.scrub_json(self._delivered(context), secrets)
        report.update(trajectory_format=fmt, steps=len(steps), usage=usage, model_usd=evidence.model_usd(usage),
                      delivered_files=[f["name"] for f in files])
        (workspace / "task.json").write_text(json.dumps({"task_id": self.task_id,
                                                         "confirmed_task": tasks.instruction(self.task_id)}))
        (workspace / "result.json").write_text(json.dumps({
            "final_output": final_output,
            "metadata": {"steps": steps, "judge_screenshots": [f"judge_screenshots/{n}" for n in shot_names],
                         "output_files": files}}))
        judgment = await asyncio.to_thread(self._judge, workspace)
        report["judgment"] = {k: judgment.get(k) for k in ("status", "score", "reason", "failure_class", "model")}
        (workspace / "report.json").write_text(json.dumps(report, indent=1, default=str))

        verification: dict[str, Any] = {
            "verifier_type": self.type, "judge": "findings-v21 (browser-use/benchmark@1e0e2f1e)",
            "judge_model": self.judge_model, "reasoning_effort": self.reasoning_effort,
            "results": [{"id": item, "result": met, "score": 1.0 if met else 0.0}
                        for item, met in (judgment.get("rubric") or {}).items()],
            "reason": judgment.get("reason"), "failure_class": judgment.get("failure_class"),
            "report": report, "evidence_dir": str(workspace),
        }
        if judgment.get("status") == "scored":
            verification["score"] = float(judgment["score"])
        context.metadata.setdefault("verifications", {})[self.verifier_id] = verification
        if judgment.get("status") != "scored":
            raise RuntimeError(f"findings judge did not score {self.task_id}: {judgment.get('reason')}")
        return context

    def _workspace(self, context: TaskStepContext) -> Path:
        root = os.environ.get(EVIDENCE_DIR_ENV)
        run = str(context.metadata.get("run_group_id") or uuid.uuid4().hex)[:12]
        if root:
            path = Path(root) / f"{self.task_id}-{run}"
            path.mkdir(parents=True, exist_ok=True)
            return path
        return Path(tempfile.mkdtemp(prefix=f"bu-judge-{self.task_id}-"))

    async def _trajectory(self, context: TaskStepContext, response) -> tuple[Any, str]:
        """The turn's trajectory as prompt_agent stored it, else asked of the agent itself, inline: the stored copy
        goes through the runner's grant server, and a failed fetch there is only logged."""
        if response is None:
            return [], "none"
        url = response.agent_trajectory_object_url or next(
            (u for u in reversed(response.target_agent_per_turn_trajectory_object_urls or []) if u), None)
        if url:
            from agent_env.config import get_config

            try:
                return json.loads(get_config().get_object_store_at(url).get(url)), "stored"
            except Exception:
                logger.exception("reading the stored trajectory of %s failed", self.task_id)
        try:
            agent = _agent(context, self.agent_name)
            extension = A2AAgent.find_extension(agent.a2a_card or {}, A2AAgent.EXT_TRAJECTORY)
            _, path = A2AAgent.operation(extension, "get")
            async with httpx.AsyncClient(timeout=120) as client:
                fetched = await client.post(agent.a2a_url + path, headers=_control_headers(agent),
                                            json={"context_id": response.a2a_context_id})
            fetched.raise_for_status()
            return fetched.json().get("trajectory") or [], "agent"
        except Exception:
            logger.exception("fetching the trajectory of %s from the agent failed", self.task_id)
            return [], "none"

    def _delivered(self, context: TaskStepContext) -> list[dict[str, Any]]:
        collected = context.metadata.get("artifacts") or {}
        if not isinstance(collected, dict):
            return []
        from agent_env.config import get_config

        files = []
        for name, url in collected.items():
            if any(part in SKIPPED_PARTS for part in Path(str(name)).parts):
                continue
            try:
                files.append((str(name), get_config().get_object_store_at(url).get(url)))
            except Exception:
                logger.exception("reading delivered file %s failed", name)
        return evidence.output_files(files)

    async def _keep_recording(self, session: dict[str, Any], context: TaskStepContext) -> dict[str, Any]:
        """The cloud browser's own MP4, stored as a file artifact; its signed link expires in an hour."""
        urls = cloud.recording_urls(session)
        if not urls:
            return {"available": bool(session.get("recordingAvailable")), "artifact_id": None}
        try:
            async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
                video = (await client.get(urls[0])).raise_for_status().content
            from agent_env.artifact.artifacts.file import FileArtifact

            # Under the run's instance id, so a local run keeps it in the local store, as collect_artifacts does.
            artifact_id = f"{context.instance_id}-recording" if context.instance_id else \
                f"bu-bench-recording-{self.task_id}-{uuid.uuid4().hex[:8]}"
            artifact = await asyncio.to_thread(
                FileArtifact.put_bytes, artifact_id,
                description=f"Browser Use Cloud recording of {self.task_id}", filename="recording.mp4",
                content=video, content_type="video/mp4")
            context.metadata.setdefault("recordings", {})[self.id] = [
                {"name": "recording.mp4", "artifact_id": artifact.id, "version": artifact.version,
                 "bytes": len(video), "content_type": "video/mp4"}]
            return {"available": True, "artifact_id": artifact.id, "bytes": len(video)}
        except Exception as error:  # noqa: BLE001 - the recording never fails the task
            # Not logger.exception: httpx's error text names the signed recording link.
            logger.warning("keeping the recording of %s failed (%s)", self.task_id, type(error).__name__)
            return {"available": True, "artifact_id": None, "error": type(error).__name__}

    def _judge(self, workspace: Path) -> dict[str, Any]:
        key = next((os.environ[name] for name in JUDGE_KEY_NAMES if os.environ.get(name)), "")
        if not key:
            try:
                from agent_env.config import get_config

                key = get_config().get_secret_store().get("OPENAI_API_KEY") or ""
            except Exception:  # noqa: BLE001
                key = ""
        if not key:
            return {"status": "error", "reason": "no judge key: set BU_BENCH_JUDGE_API_KEY or OPENAI_API_KEY"}
        env = {name: value for name, value in os.environ.items()
               if name in {"PATH", "HOME", "LANG", "SSL_CERT_FILE", "OPENAI_BASE_URL", "TMPDIR"}}
        env.update(EVAL_WORKSPACE=str(workspace), EVAL_JUDGMENT_PATH=str(workspace / "judgment.json"),
                   EVAL_TASK_PATH=str(workspace / "task.json"), EVAL_RESULT_PATH=str(workspace / "result.json"),
                   EVAL_JUDGE_MODEL=self.judge_model,
                   EVAL_JUDGE_OPTIONS_JSON=json.dumps({"reasoning_effort": self.reasoning_effort}),
                   EVAL_JUDGE_API_KEY=key, PYTHONDONTWRITEBYTECODE="1")
        with (workspace / "judge.log").open("w") as log:
            try:
                subprocess.run([sys.executable, str(JUDGE_RUN)], env=env, stdout=log, stderr=subprocess.STDOUT,
                               timeout=JUDGE_TIMEOUT_S, check=False)
            except subprocess.TimeoutExpired:
                return {"status": "error", "reason": f"judge timed out after {JUDGE_TIMEOUT_S}s"}
        try:
            return json.loads((workspace / "judgment.json").read_text())
        except (OSError, ValueError) as error:
            return {"status": "error", "reason": f"judge wrote no judgment: {error}"}
