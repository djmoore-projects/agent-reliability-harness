"""Baseline vs candidate comparison and the CI regression gate."""

from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import summarize
from .schema import GateConfig, RunResult


@dataclass
class GateOutcome:
    passed: bool
    breaches: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    markdown: str = ""


def _task_passes(stats: dict) -> bool:
    """Majority vote across trials: a task 'passes' if it succeeded on at least half its trials."""
    return stats["pass_rate"] >= 0.5


def _pct_change(old: float, new: float) -> float:
    return 0.0 if old == 0 else (new - old) / old * 100


def compare(baseline: RunResult, candidate: RunResult, gate: GateConfig) -> GateOutcome:
    b, c = summarize(baseline), summarize(candidate)
    breaches: list[str] = []

    if gate.min_success_rate is not None and c["success_rate"] < gate.min_success_rate:
        breaches.append(f"success rate {c['success_rate']:.0%} is below the floor of {gate.min_success_rate:.0%}")
    drop = b["success_rate"] - c["success_rate"]
    if gate.max_success_drop is not None and drop > gate.max_success_drop:
        breaches.append(f"success rate dropped {drop * 100:.1f} pts (limit {gate.max_success_drop * 100:.1f})")
    cost_delta = _pct_change(b["mean_cost_usd"], c["mean_cost_usd"])
    if gate.max_cost_increase_pct is not None and cost_delta > gate.max_cost_increase_pct:
        breaches.append(f"mean cost/task up {cost_delta:.0f}% (limit {gate.max_cost_increase_pct:.0f}%)")
    lat_delta = _pct_change(b["latency_p95_s"], c["latency_p95_s"])
    if gate.max_latency_increase_pct is not None and lat_delta > gate.max_latency_increase_pct:
        breaches.append(f"p95 latency up {lat_delta:.0f}% (limit {gate.max_latency_increase_pct:.0f}%)")

    for label, stats in (("baseline", b), ("candidate", c)):
        if stats["excluded_infra_trials"]:
            breaches.append(f"{label} has {stats['excluded_infra_trials']} trial(s) lost to infrastructure errors; re-run")

    shared = sorted(set(b["per_task"]) & set(c["per_task"]))
    regressions = [t for t in shared if _task_passes(b["per_task"][t]) and not _task_passes(c["per_task"][t])]
    fixes = [t for t in shared if not _task_passes(b["per_task"][t]) and _task_passes(c["per_task"][t])]
    if gate.fail_on_task_regression and regressions:
        breaches.append(f"{len(regressions)} task(s) regressed: {', '.join(regressions)}")

    outcome = GateOutcome(passed=not breaches, breaches=breaches, regressions=regressions, fixes=fixes)
    outcome.markdown = _render(baseline, candidate, b, c, outcome)
    return outcome


def _row(label: str, old: float | None, new: float | None, fmt: str, better: str) -> str:
    if old is None or new is None:
        return f"| {label} | – | – | |"
    delta = new - old
    arrow = ""
    if abs(delta) > 1e-9:
        good = (delta > 0) == (better == "up")
        arrow = "🟢" if good else "🔴"
    sign = "+" if delta > 0 else ""
    if fmt == "pct":
        return f"| {label} | {old:.0%} | {new:.0%} | {arrow} {sign}{delta * 100:.1f} pts |"
    if fmt == "usd":
        return f"| {label} | ${old:.4f} | ${new:.4f} | {arrow} {sign}{_pct_change(old, new):.0f}% |"
    if fmt == "s":
        return f"| {label} | {old:.1f}s | {new:.1f}s | {arrow} {sign}{_pct_change(old, new):.0f}% |"
    return f"| {label} | {old:.1f} | {new:.1f} | {arrow} {sign}{delta:.1f} |"


def _render(baseline: RunResult, candidate: RunResult, b: dict, c: dict, outcome: GateOutcome) -> str:
    status = "✅ **Agent quality gate passed**" if outcome.passed else "❌ **Agent quality gate failed**"
    lines = [
        "## Agent reliability report",
        "",
        status,
        "",
        (f"`{baseline.label}` ({baseline.git_sha or 'n/a'}) → `{candidate.label}` ({candidate.git_sha or 'n/a'}) · "
         f"{c['tasks']} tasks × {candidate.trials_per_task} trial(s) · suite `{candidate.suite}`"),
        "",
        "| Metric | Baseline | Candidate | Δ |",
        "|---|---|---|---|",
        _row("Task success (pass@1)", b["success_rate"], c["success_rate"], "pct", "up"),
        _row(f"Pass all {candidate.trials_per_task} trials (pass^k)", b["pass_all_k"], c["pass_all_k"], "pct", "up"),
        _row("Tool-use checks", b["tool_use_accuracy"], c["tool_use_accuracy"], "pct", "up"),
        _row("Final-state checks", b["state_accuracy"], c["state_accuracy"], "pct", "up"),
        _row("Quality (judge) checks", b["quality_pass_rate"], c["quality_pass_rate"], "pct", "up"),
        _row("Mean cost / task", b["mean_cost_usd"], c["mean_cost_usd"], "usd", "down"),
        _row("p50 latency", b["latency_p50_s"], c["latency_p50_s"], "s", "down"),
        _row("p95 latency", b["latency_p95_s"], c["latency_p95_s"], "s", "down"),
        _row("Mean steps (LLM + tool calls)", b["mean_steps"], c["mean_steps"], "n", "down"),
        "",
        (f"Candidate success 95% CI: {c['success_ci95'][0]:.0%}–{c['success_ci95'][1]:.0%} "
         f"(n={c['trials']} trials)."),
        "",
    ]
    if outcome.breaches:
        lines += ["**Gate breaches**", ""] + [f"- {x}" for x in outcome.breaches] + [""]
    if outcome.fixes:
        lines += [f"**Fixed:** {', '.join(f'`{t}`' for t in outcome.fixes)}", ""]
    if outcome.regressions:
        lines += [f"**Regressed:** {', '.join(f'`{t}`' for t in outcome.regressions)}", ""]

    lines += ["<details><summary>Per-task results</summary>", "",
              "| Task | Baseline | Candidate | Candidate failing checks |", "|---|---|---|---|"]
    for task in sorted(set(b["per_task"]) | set(c["per_task"])):
        bt, ct = b["per_task"].get(task), c["per_task"].get(task)
        fmt = (lambda s: f"{s['passes']}/{s['trials']}" if s else "–")
        failing = ", ".join(f"{k} ×{v}" for k, v in (ct or {}).get("failing_checks", {}).items()) or ""
        lines.append(f"| `{task}` | {fmt(bt)} | {fmt(ct)} | {failing} |")
    lines += ["", "</details>", ""]
    return "\n".join(lines)
