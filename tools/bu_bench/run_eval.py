#!/usr/bin/env python3
"""`agent-env run` with a cap on concurrent tasks, keeping each task's judge evidence under OUT/AGENT/evidence.

agent-env 0.9.1536 runs at most 4 tasks at a time and has no flag to change that; this sets the cap and then runs
its own CLI, unchanged. With --tasks, it first writes the bundle eval NAME (which must start with "ci-") listing
exactly those task ids, so CI can run one shard of an eval or rerun exact tasks.

Usage: run_eval.py BUNDLE --eval NAME --agent NAME --model MODEL --out DIR [--concurrency N] [--tasks ID,ID,...]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--eval", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--tasks", help="comma-separated task ids; writes the eval file first")
    args = parser.parse_args()

    if args.tasks:
        if not args.eval.startswith("ci-"):
            parser.error("--tasks needs an eval name starting with ci-")
        ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        missing = [t for t in ids if not (Path(args.bundle) / "tasks" / f"{t}.json").exists()]
        if not ids or missing:
            parser.error(f"unknown task ids: {missing or 'none given'}")
        listed = ", ".join(f'"{t}"' for t in ids)
        eval_file = Path(args.bundle) / "evals" / f"{args.eval}.toml"
        eval_file.write_text(f"# written by run_eval.py\ntasks = [{listed}]\n")

    evidence = (args.out / args.agent / "evidence").resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    os.environ["BU_BENCH_EVIDENCE_DIR"] = str(evidence)

    import agent_env.bundle.run as bundle_run
    from agent_env.cli import cli

    bundle_run.MAX_CONCURRENT_RUNS = args.concurrency
    argv = ["run", args.bundle, "--eval", args.eval, "--agent", args.agent, "--model", args.model]
    try:
        cli.main(args=argv, prog_name="agent-env")
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else (0 if exit_.code is None else 1)
    except Exception as error:  # noqa: BLE001 - reported, and the summary still runs on what finished
        print(f"agent-env run raised: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
