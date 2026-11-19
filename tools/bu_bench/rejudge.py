#!/usr/bin/env python3
"""Rejudge saved tasks whose judge errored: run the frozen findings judge again on the task's saved evidence.

Each EVIDENCE_DIR is one task's judge workspace as the bu_findings_verifier step left it (task.json, result.json,
judge_screenshots/, delivered files). The agent is not rerun; the judge gets the same inputs, model and effort. The
errored judgment is kept as judgment-error-N.json and report.json gets the new judgment. A judge error is never a 0.

Usage: rejudge.py EVIDENCE_DIR [EVIDENCE_DIR ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agentenv_benchmarks.bu_bench.steps import BuFindingsVerifierTaskStep


def main() -> int:
    failed = 0
    for workspace in map(Path, sys.argv[1:]):
        report = json.loads((workspace / "report.json").read_text())
        config = json.loads((workspace / "judge-config.json").read_text()) \
            if (workspace / "judge-config.json").exists() else {}
        step = BuFindingsVerifierTaskStep.from_dict({
            "id": "grade", "task_id": report["task_id"], "prompt_id": "solve",
            "judge_model": config.get("model", "gpt-5.6-luna"),
            "reasoning_effort": config.get("reasoning_effort", "xhigh")})
        previous = workspace / "judgment.json"
        if previous.exists():
            n = len(list(workspace.glob("judgment-error-*.json"))) + 1
            previous.rename(workspace / f"judgment-error-{n}.json")
        judgment = step._judge(workspace)
        report["judgment"] = {k: judgment.get(k) for k in ("status", "score", "reason", "failure_class", "model")}
        report["rejudged"] = True
        (workspace / "report.json").write_text(json.dumps(report, indent=1, default=str))
        failed += judgment.get("status") != "scored"
        print(f"{report['task_id']}: {judgment.get('status')} {judgment.get('score')}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
