# Public V2.1 findings judge

This adapter freezes browser-use/benchmark commit
1e0e2f1ed12d3dfbdfab6b7bff415ec090ff9f74: all 200 task contracts, validation,
prompt, schema, screenshot selection and scoring. SOURCE.json verifies their
bytes and the browser-use 0.11.5 LLM adapter sources before grading.

Use datasets/BU_Bench_V21_200.json. It contains only task IDs and instructions.
The judge decrypts the separate upstream snapshot after execution and checks
that the instruction matches. As in the public benchmark, this is separation
of inputs, not a filesystem sandbox against an unrestricted coding agent.

The judge is gpt-5.6-luna, xhigh, 32768 completion tokens, 300-second request
timeout and two SDK retries. It excludes reasoning from judge input. Original
raw evidence is retained. Screenshot labels say after-action, matching the
Browser Code harness. Delivered file text comes from the harness collector;
binary or clipped files remain explicitly marked, not presumed inspected.

The public schema requires one finding per rubric item, including the reason
for not-assessable findings. Public code computes weighted credit and applies
the reward-hacking/canary zero. Judge errors publish no score.

Artifacts: judge-config.json, judge_trace.json, and judge-recording with
judge_input.json and judge_response.json. These capture SDK request/response
objects, not original HTTP wire bytes. Retain them privately.

The legacy findings judge and 60-task dataset remain unchanged.
