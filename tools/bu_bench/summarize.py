#!/usr/bin/env python3
"""Per-agent results of AgentEnv runs, paired with the same tasks on our eval platform.

Reads OUT/<agent>/evidence/*/report.json (written by the bu_findings_verifier step) and OUT/model-usage.jsonl (the
model endpoint's log) from each OUT, plus reference/bakeoff-full.json (every bake-off entry on all 200 tasks), and
writes results.json, results.md and pertask.csv into the first OUT. Later OUTs override earlier ones per task, so
reruns go last. Scores are the judge's continuous score x100; a task the judge did not score has no score (listed,
never counted as 0).

Costs: model $ = everything through the endpoint except the judge (agent + Stagehand's own calls) over all attempts,
divided by the tasks with a report; judge $ the same for gpt-5.6-luna calls; both at the bake-off's $1 / $0.10 cached /
$5 out per 1M. Cloud $ = Browser Use browser + proxy cost of the task's session. Minutes = browser lifetime.

Usage: summarize.py OUT [OUT ...]
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = json.loads((HERE / "reference" / "bakeoff-full.json").read_text())["arms"]
JUDGE_MODEL = "gpt-5.6-luna"
CLOUD_KEYS = ("browser_usd", "proxy_usd")


def ci(values: list[float], rounds: int = 10_000, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def load(out: Path) -> tuple[dict[str, dict[str, dict]], dict[str, dict]]:
    """agent -> task -> latest report; agent -> endpoint usage (only when OUT holds one agent)."""
    agents: dict[str, dict[str, dict]] = {}
    for report_path in sorted(out.glob("*/evidence/*/report.json"), key=lambda p: p.stat().st_mtime):
        report = json.loads(report_path.read_text())
        agents.setdefault(report_path.parents[2].name, {})[report["task_id"]] = report
    usage: dict[str, dict] = {}
    log = out / "model-usage.jsonl"
    if log.exists() and len(agents) == 1:
        totals = {"model_usd": 0.0, "judge_usd": 0.0, "requests": 0, "rate_limited": 0, "spend_capped": 0,
                  "errors": 0}
        for line in log.read_text().splitlines():
            entry = json.loads(line)
            totals["requests"] += 1
            totals["rate_limited"] += entry.get("status") == 429 and "retry" in entry
            totals["spend_capped"] += bool(entry.get("spend_cap"))
            totals["errors"] += entry.get("status", 200) >= 400 and "retry" not in entry
            key = "judge_usd" if str(entry.get("model") or "").startswith(JUDGE_MODEL) else "model_usd"
            totals[key] += entry.get("usd") or 0.0
        usage[next(iter(agents))] = totals
    return agents, usage


def main() -> int:
    outs = [Path(p) for p in sys.argv[1:]]
    agents: dict[str, dict[str, dict]] = {}
    usage: dict[str, dict] = {}
    attempts: dict[str, int] = {}
    for out in outs:
        found, spent = load(out)
        for agent, reports in found.items():
            agents.setdefault(agent, {}).update(reports)
            attempts[agent] = attempts.get(agent, 0) + len(reports)
        for agent, totals in spent.items():
            mine = usage.setdefault(agent, dict.fromkeys(totals, 0))
            for key, value in totals.items():
                mine[key] += value
    summary: dict = {"agents": {}}
    head = [("| agent | scored | score [95% CI] | ours, same tasks | paired diff [95% CI] | model $/task | "
             "judge $/task | cloud $/task | min/task (ours) |"), "|---|---|---|---|---|---|---|---|---|"]
    lines, csv_rows = [], []
    for agent, reports in sorted(agents.items()):
        ref = REFERENCE.get(agent, {})
        rows = []
        for task_id, report in sorted(reports.items()):
            judgment = report.get("judgment") or {}
            scored = judgment.get("status") == "scored"
            rows.append({"task_id": task_id, "status": judgment.get("status"),
                         "failure_class": judgment.get("failure_class") or "",
                         "score": 100 * judgment["score"] if scored else None,
                         "reference_score": (ref.get(task_id) or {}).get("score"),
                         "reference_minutes": (ref.get(task_id) or {}).get("minutes"),
                         "minutes": (report.get("browser_seconds") or 0) / 60, "model_usd": report.get("model_usd"),
                         "cloud_usd": sum((report.get("cloud") or {}).get(k) or 0 for k in CLOUD_KEYS),
                         "trajectory_source": report.get("trajectory_source"), "steps": report.get("steps"),
                         "agent_error": (report.get("agent_error") or {}).get("code") or "",
                         "agent_error_message": ((report.get("agent_error") or {}).get("message") or "")[:120],
                         "recording": bool((report.get("recording") or {}).get("artifact_id"))})
        scores = [r["score"] for r in rows if r["score"] is not None]
        paired = [(r["score"], r["reference_score"]) for r in rows
                  if r["score"] is not None and r["reference_score"] is not None]
        n = max(1, len(rows))
        spent = usage.get(agent, {})
        entry = {"tasks": len(rows), "scored": len(scores), "attempts": attempts.get(agent, 0),
                 "missing": sorted(set(ref) - set(reports)) if len(ref) == 200 else [],
                 "cloud_usd": sum(r["cloud_usd"] for r in rows) / n,
                 "minutes": sum(r["minutes"] for r in rows) / n, "endpoint": spent,
                 "model_usd": (spent["model_usd"] / n) if spent else sum(r["model_usd"] or 0 for r in rows) / n,
                 "judge_usd": (spent["judge_usd"] / n) if spent else None}
        if scores:
            entry.update(score=statistics.fmean(scores), score_ci=ci(scores) if len(scores) > 1 else (None, None))
        if paired:
            diffs = [a - b for a, b in paired]
            entry.update(paired=len(paired), reference=statistics.fmean(b for _, b in paired),
                         diff=statistics.fmean(diffs), diff_ci=ci(diffs) if len(diffs) > 1 else (None, None),
                         reference_minutes=statistics.fmean(r["reference_minutes"] for r in rows
                                                            if r["reference_minutes"] is not None))
        summary["agents"][agent] = {**entry, "rows": rows}
        csv_rows += [{"agent": agent, **{k: v for k, v in r.items() if k != "agent_error_message"}} for r in rows]

        def interval(value, bounds, sign=""):
            if value is None:
                return "n/a"
            text = f"{value:{sign}.1f}"
            if bounds and bounds[0] is not None:
                text += f" [{bounds[0]:{sign}.1f}, {bounds[1]:{sign}.1f}]"
            return text

        judge = f"{entry['judge_usd']:.2f}" if entry["judge_usd"] is not None else "n/a"
        ref_min = f" ({entry['reference_minutes']:.1f})" if "reference_minutes" in entry else ""
        score = interval(entry.get("score"), entry.get("score_ci"))
        lines.append(f"| {agent} | {len(scores)}/{len(rows)} | {score} | "
                     f"{interval(entry.get('reference'), None)} | "
                     f"{interval(entry.get('diff'), entry.get('diff_ci'), '+')} | {entry['model_usd']:.2f} | "
                     f"{judge} | {entry['cloud_usd']:.2f} | {entry['minutes']:.1f}{ref_min} |")
    problems = ["", "| agent | task | status | score | agent error |", "|---|---|---|---|---|"]
    for agent, entry in sorted(summary["agents"].items()):
        for r in entry["rows"]:
            if r["status"] != "scored" or r["agent_error"]:
                score = f"{r['score']:.0f}" if r["score"] is not None else "-"
                problems.append(f"| {agent} | {r['task_id']} | {r['status']} | {score} | {r['agent_error']} "
                                f"{r['agent_error_message']} |")
        if entry["missing"]:
            problems.append(f"| {agent} | {len(entry['missing'])} tasks with no report | | | |")
    target = outs[0]
    (target / "results.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    (target / "results.md").write_text("\n".join(head + lines + problems) + "\n")
    if csv_rows:
        with (target / "pertask.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    print("\n".join(head + lines + problems))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
