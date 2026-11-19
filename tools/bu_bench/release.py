#!/usr/bin/env python3
"""Build BU Bench V2.1's Hugging Face release from a summarize.py CSV: a dataset card and runs.jsonl, a row per run.

    python tools/bu_bench/release.py RESULTS_CSV OUT --pass full200 --note "full200=What this pass is."

A row holds the run's pass, arm, harness, model, task id, continuous score (0-100), the reference score on Browser Use's
own platform for the same task, minutes, steps and the cloud browser's cost. Never task text, rubrics, answers or trajectories: BU Bench's
tasks are sealed, and those hold them. The card's tables are the leaderboard. Upload with `hf upload-large-folder`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

HARNESS = {
    "browser-use-104rc1": "browser-use 1.0.4rc1 agent",
    "codex-browser-use-mcp": "Codex + Browser Use MCP",
    "codex-playwright-mcp": "Codex + Playwright MCP",
    "codex-chrome-devtools-mcp": "Codex + Chrome DevTools MCP",
    "codex-bu-cli": "Codex + Browser Use CLI",
    "codex-agent-browser": "Codex + agent-browser",
    "codex-opencode-browse": "Codex + OpenCode browse",
    "codex-stagehand": "Codex + Stagehand",
}
MODEL, EFFORT, JUDGE = "gpt-6.1-terra", "medium", "findings-v21 (browser-use/benchmark@1e0e2f1e) on gpt-5.6-luna xhigh"


def number(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


def bootstrap(values: list[float], rounds: int = 10_000, seed: int = 7) -> tuple[float, float]:
    """summarize.py's interval, so the card's CIs are the results table's."""
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(rounds))
    return means[int(rounds * 0.025)], means[int(rounds * 0.975) - 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--pass", dest="pass_name", required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    rows = []
    for row in csv.DictReader(args.results.open()):
        rows.append({
            "pass": args.pass_name, "arm": row["agent"], "harness": HARNESS.get(row["agent"], row["agent"]),
            "model": MODEL, "effort": EFFORT, "judge": JUDGE, "task_id": row["task_id"], "status": row["status"],
            "score": number(row["score"]), "reference_score": number(row["reference_score"]),
            "minutes": number(row["minutes"]), "steps": number(row["steps"]),
            "cloud_usd": number(row["cloud_usd"]),
            "recording": row["recording"] == "True",
        })
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    board = []
    for arm in sorted({r["arm"] for r in rows}):
        mine = [r for r in rows if r["arm"] == arm and r["score"] is not None]
        paired = [r["score"] - r["reference_score"] for r in mine if r["reference_score"] is not None]
        low, high = bootstrap([r["score"] for r in mine])
        d_low, d_high = bootstrap(paired) if paired else (None, None)
        board.append((statistics.fmean(r["score"] for r in mine), arm, len(mine), low, high,
                      statistics.fmean(r["reference_score"] for r in mine if r["reference_score"] is not None),
                      statistics.fmean(paired) if paired else None, d_low, d_high,
                      statistics.fmean(r["minutes"] for r in mine if r["minutes"] is not None)))
    board.sort(reverse=True)
    table = ["| # | Harness | Runs | Score [95% CI] | Reference | Paired diff [95% CI] | Min/task |",
             "|---:|---|---:|---|---:|---|---:|"]
    for rank, (score, arm, n, low, high, ref, diff, d_low, d_high, minutes) in enumerate(board, 1):
        table.append(f"| {rank} | {HARNESS.get(arm, arm)} | {n} | {score:.1f} [{low:.1f}, {high:.1f}] | {ref:.1f} | "
                     f"{diff:+.1f} [{d_low:+.1f}, {d_high:+.1f}] | {minutes:.1f} |")
    notes = dict(part.split("=", 1) for part in args.note.split(";") if "=" in part)
    card = f"""---
license: apache-2.0
pretty_name: BU Bench V2.1 on agent-env
tags:
- web-agents
- browser-agents
- bu-bench
- agentenv
- rl-environment
configs:
- config_name: runs
  data_files: runs.jsonl
---

# BU Bench V2.1 on agent-env

Runs of [BU Bench V2.1](https://github.com/browser-use/benchmark)'s 200 live-web tasks on
[agent-env](https://github.com/scaleapi/agentenv-framework), each on a fresh Browser Use Cloud browser, graded by the
benchmark's frozen findings judge, as the [bu-bench bundle](https://github.com/earakely-scale/agentenv-benchmarks)
runs them. 8 harnesses, the same model, tasks and judge: only the harness changes.

- **{args.pass_name}:** {notes.get(args.pass_name, "")} Model `{MODEL}` at effort **{EFFORT}**. Judge: {JUDGE}.

## Results

{chr(10).join(table)}

Score is the judge's continuous score (0-100, weighted rubric items), not a pass rate. Reference is the same task on
Browser Use's own eval platform with the same harness, adapter, model and judge; the paired diff is per task, with a
bootstrap 95% CI.

## Files

- `runs.jsonl`: a row per run: pass, arm, harness, model and effort, judge, task id, status, score, reference score,
  minutes, steps, cloud browser cost, and whether the browser's video was kept.

The tasks are sealed: no row holds task text, rubrics, the agent's answer or its trajectory. Please don't publish
decrypted tasks or train on them.
"""
    (args.out / "README.md").write_text(card)
    print(f"{len(rows)} runs, {len(board)} arms -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
