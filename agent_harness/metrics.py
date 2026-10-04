"""Aggregate trial results into suite-level reliability metrics."""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

from .schema import RunResult, TrialResult


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval. Behaves sensibly at small n and at 0% / 100%."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _category_rate(trials: list[TrialResult], category: str) -> float | None:
    checks = [c for t in trials for c in t.checks if c.category == category]
    return sum(c.passed for c in checks) / len(checks) if checks else None


def by_task(run: RunResult) -> dict[str, list[TrialResult]]:
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    for t in run.trials:
        grouped[t.task_id].append(t)
    return dict(grouped)


def summarize(run: RunResult) -> dict[str, Any]:
    trials = run.trials
    tasks = by_task(run)
    n = len(trials)
    passes = sum(t.success for t in trials)
    latencies = [t.latency_s for t in trials]
    costs = [t.cost_usd for t in trials]

    per_task = {}
    for task_id, ts in sorted(tasks.items()):
        failing = Counter(c.name for t in ts for c in t.checks if c.required and not c.passed)
        per_task[task_id] = {
            "passes": sum(t.success for t in ts),
            "trials": len(ts),
            "pass_rate": sum(t.success for t in ts) / len(ts),
            "mean_cost_usd": statistics.mean(t.cost_usd for t in ts),
            "mean_latency_s": statistics.mean(t.latency_s for t in ts),
            "mean_steps": statistics.mean(t.steps for t in ts),
            "failing_checks": dict(failing.most_common()),
        }

    failure_counts = Counter(c.name for t in trials for c in t.checks if c.required and not c.passed)
    lo, hi = wilson_interval(passes, n)
    return {
        "tasks": len(tasks),
        "trials": n,
        "trials_per_task": run.trials_per_task,
        "success_rate": passes / n if n else 0.0,
        "success_ci95": [lo, hi],
        # pass^k: the task succeeded on *every* trial. The reliability number, not the demo number.
        "pass_all_k": sum(all(t.success for t in ts) for ts in tasks.values()) / len(tasks) if tasks else 0.0,
        "flaky_tasks": sorted(tid for tid, ts in tasks.items() if 0 < sum(t.success for t in ts) < len(ts)),
        "tool_use_accuracy": _category_rate(trials, "tool_use"),
        "state_accuracy": _category_rate(trials, "state"),
        "quality_pass_rate": _category_rate(trials, "quality"),
        "budget_pass_rate": _category_rate(trials, "efficiency"),
        "mean_cost_usd": statistics.mean(costs) if costs else 0.0,
        "total_cost_usd": sum(costs),
        "judge_cost_usd": sum(t.judge_cost_usd for t in trials),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "mean_steps": statistics.mean(t.steps for t in trials) if trials else 0.0,
        "mean_llm_calls": statistics.mean(t.llm_calls for t in trials) if trials else 0.0,
        "mean_tool_calls": statistics.mean(t.tool_calls for t in trials) if trials else 0.0,
        "top_failures": dict(failure_counts.most_common(10)),
        "per_task": per_task,
    }
