"""Load a suite, run every task N times through the adapter, grade each trial."""

from __future__ import annotations

import importlib
import logging
import string
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .checks import TaskContext, run_check
from .judge import Judge
from .schema import RunResult, Suite, Task, TrialResult
from .tracing import Tracer

logger = logging.getLogger("agent_harness")

# An adapter runs the agent on one task input and returns what to grade:
#   {"state": <final agent state>, "env": <observed side effects>}
Adapter = Callable[[dict, Tracer], dict]


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    suite = Suite.model_validate(yaml.safe_load(path.read_text()))
    # Let adapters referenced as "adapter:run" live next to the suite file.
    sys.path.insert(0, str(path.parent.resolve()))
    return suite


def load_adapter(ref: str) -> Adapter:
    module, func = ref.split(":")
    return getattr(importlib.import_module(module), func)


def _git_sha(cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _render(spec: Any, variables: dict) -> Any:
    """Substitute ${var} from the task input into check specs, so shared checks can say
    e.g. `equals: ${company_name}`."""
    if isinstance(spec, str):
        return string.Template(spec).safe_substitute({k: v for k, v in variables.items() if isinstance(v, str)})
    if isinstance(spec, list):
        return [_render(x, variables) for x in spec]
    if isinstance(spec, dict):
        return {k: _render(v, variables) for k, v in spec.items()}
    return spec


def run_trial(suite: Suite, task: Task, trial: int, adapter: Adapter, judge_client: Any) -> TrialResult:
    tracer = Tracer(faults=task.faults)
    task_input = {**suite.defaults, **task.input}
    start = time.perf_counter()
    crashed = None
    try:
        output = adapter(task_input, tracer) or {}
    except Exception as exc:
        logger.exception("Task %s trial %d crashed", task.id, trial)
        crashed = f"{type(exc).__name__}: {exc}"
        output = {}
    latency = time.perf_counter() - start
    if crashed:
        output["__crashed__"] = crashed

    trace = tracer.trace()
    tool_cost = sum(suite.tool_costs.get(s.name, 0.0) for s in trace.tool_calls)
    cost = trace.cost_usd + tool_cost

    judge = Judge(suite.judge_model, client=judge_client) if judge_client is not None else None
    ctx = TaskContext(task=task, output=output, trace=trace, latency_s=latency, cost_usd=cost,
                      judge=judge, tool_costs=suite.tool_costs)
    shared = [] if "no_shared_checks" in task.tags else suite.shared_checks
    specs = _render([{"type": "no_crash"}] + task.checks + shared, task_input)
    # Deterministic checks first; skip paid judge calls when the run already crashed.
    results = [run_check(ctx, s) for s in specs if s.get("type") not in ("judge", "faithfulness")]
    for spec in (s for s in specs if s.get("type") in ("judge", "faithfulness")):
        if crashed:
            continue
        results.append(run_check(ctx, spec))

    input_tokens, output_tokens = trace.tokens
    success = all(r.passed for r in results if r.required)
    logger.info("%-28s trial %d  %s  %5.1fs  $%.4f  %d steps", task.id, trial,
                "PASS" if success else "FAIL", latency, cost, trace.steps)
    return TrialResult(
        task_id=task.id, trial=trial, success=success, checks=results, latency_s=latency, cost_usd=cost,
        judge_cost_usd=judge.cost_usd if judge else 0.0, steps=trace.steps,
        llm_calls=len(trace.llm_calls), tool_calls=len(trace.tool_calls),
        input_tokens=input_tokens, output_tokens=output_tokens, node_path=trace.node_path,
        output={k: v for k, v in output.items() if k in ("state", "env", "__crashed__")},
        trace=trace, crashed=crashed,
    )


def run_suite(suite: Suite, *, label: str, trials: int = 1, concurrency: int = 4, use_judge: bool = True,
              task_filter: list[str] | None = None, config: dict | None = None,
              repo_dir: Path | None = None) -> RunResult:
    adapter = load_adapter(suite.adapter)
    judge_client = None
    if use_judge:
        import anthropic
        judge_client = anthropic.Anthropic(max_retries=4)

    tasks = [t for t in suite.tasks if not task_filter or t.id in task_filter]
    jobs = [(t, k) for k in range(trials) for t in tasks]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda job: run_trial(suite, job[0], job[1], adapter, judge_client), jobs))

    return RunResult(
        suite=suite.name, label=label, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=_git_sha(repo_dir), trials_per_task=trials, config=config or {},
        trials=sorted(results, key=lambda r: (r.task_id, r.trial)),
    )
