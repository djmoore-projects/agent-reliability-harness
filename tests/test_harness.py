"""Offline tests: tracing, fault injection, checks, metrics, and the gate. No API keys needed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent_harness.checks import TaskContext, run_check
from agent_harness.compare import compare
from agent_harness.metrics import percentile, summarize, wilson_interval
from agent_harness.pricing import cost_usd
from agent_harness.runner import _render, load_suite, run_suite
from agent_harness.schema import GateConfig, Span, Task, Trace

TOY = Path(__file__).resolve().parent.parent / "examples" / "toy_support_agent"
sys.path.insert(0, str(TOY))


@pytest.fixture(scope="module")
def toy_suite():
    return load_suite(TOY / "suite.yaml")


@pytest.fixture(scope="module")
def good_run(toy_suite):
    return run_suite(toy_suite, label="good", trials=2, use_judge=False)


@pytest.fixture(scope="module")
def buggy_run(toy_suite):
    import os
    os.environ["TOY_AGENT_VARIANT"] = "buggy"
    try:
        return run_suite(toy_suite, label="buggy", trials=2, use_judge=False)
    finally:
        del os.environ["TOY_AGENT_VARIANT"]


# --- tracing + fault injection ---------------------------------------------------


def test_tracer_records_nodes_llm_and_tools(good_run):
    trial = next(t for t in good_run.trials if t.task_id == "refund-eligible")
    assert trial.node_path == ["agent", "tools", "agent", "tools", "agent"]
    assert [s.name for s in trial.trace.tool_calls] == ["lookup_order", "issue_refund"]
    assert all(s.node == "tools" for s in trial.trace.tool_calls)
    llm = trial.trace.llm_calls
    assert len(llm) == 3 and all(s.model == "claude-haiku-4-5" and s.input_tokens > 0 for s in llm)
    assert trial.cost_usd == pytest.approx(sum(s.cost_usd for s in llm))


def test_injected_fault_surfaces_inside_agent(good_run):
    trial = next(t for t in good_run.trials if t.task_id == "lookup-outage-recovers")
    lookups = trial.trace.of("tool", "lookup_order")
    assert len(lookups) == 2
    assert "503" in (lookups[0].error or "") and lookups[1].error is None
    assert len(trial.trace.of("fault")) == 1
    assert trial.success


def test_fault_times_minus_one_fires_every_call(good_run):
    trial = next(t for t in good_run.trials if t.task_id == "lookup-hard-down-escalates")
    assert all(s.error for s in trial.trace.of("tool", "lookup_order"))
    assert trial.output["env"]["ledger"][0]["action"] == "escalate"


# --- checks ----------------------------------------------------------------------


def _ctx(state=None, env=None, spans=None, task_input=None):
    task = Task(id="t", input=task_input or {}, checks=[])
    return TaskContext(task=task, output={"state": state or {}, "env": env or {}},
                       trace=Trace(spans=spans or []), latency_s=1.0, cost_usd=0.1)


def test_field_and_each_predicates():
    ctx = _ctx(state={"seq": [
        {"channel": "email", "body": "word " * 10, "subject": "Hi"},
        {"channel": "linkedin", "body": "x" * 301, "subject": None},
    ]})
    assert run_check(ctx, {"type": "field", "path": "state.seq.1.channel", "equals": "linkedin"}).passed
    assert run_check(ctx, {"type": "field", "path": "state.seq", "min_len": 2, "max_len": 2}).passed
    assert not run_check(ctx, {"type": "field", "path": "state.missing", "exists": True}).passed
    assert run_check(ctx, {"type": "each", "path": "state.seq", "where": {"channel": "email"},
                           "field": "body", "max_words": 120}).passed
    bad = run_check(ctx, {"type": "each", "path": "state.seq", "where": {"channel": "linkedin"},
                          "field": "body", "max_chars": 300})
    assert not bad.passed and "max_chars" in bad.detail


def test_tool_and_budget_checks():
    spans = [Span(id="1", kind="tool", name="search", start=0, end=1, input="Stripe funding"),
             Span(id="2", kind="tool", name="search", start=1, end=2, input="Stripe CEO"),
             Span(id="3", kind="llm", name="m", start=0, end=1)]
    ctx = _ctx(spans=spans)
    assert run_check(ctx, {"type": "tool_calls", "tool": "search", "min": 1, "max": 3}).passed
    assert not run_check(ctx, {"type": "tool_calls", "tool": "search", "max": 1}).passed
    assert run_check(ctx, {"type": "tools_used", "must_use": ["search"], "must_not_use": ["refund"]}).passed
    assert run_check(ctx, {"type": "tool_args", "tool": "search", "contains_any": ["stripe"]}).passed
    assert run_check(ctx, {"type": "budget", "max_steps": 3, "max_cost_usd": 0.2}).passed
    assert not run_check(ctx, {"type": "budget", "max_steps": 2}).passed


def test_unknown_or_broken_check_fails_closed():
    ctx = _ctx()
    assert run_check(ctx, {"type": "nope"}).error
    judged = run_check(ctx, {"type": "judge", "inputs": ["state"], "rubric": "x"})  # no judge configured
    assert not judged.passed and judged.error


def test_template_rendering():
    assert _render({"equals": "${company_name}", "n": 3, "l": ["${x}"]}, {"company_name": "Stripe", "x": "y"}) == \
        {"equals": "Stripe", "n": 3, "l": ["y"]}


# --- metrics + gate --------------------------------------------------------------


def test_stats_helpers():
    lo, hi = wilson_interval(9, 10)
    assert 0.55 < lo < 0.6 and 0.98 < hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert cost_usd("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert cost_usd("claude-opus-5", 1_000_000, 0) == pytest.approx(5.0)  # longest prefix wins
    assert cost_usd("unknown-model", 1000, 1000) == 0.0


def test_summary_of_good_run(good_run):
    s = summarize(good_run)
    assert s["success_rate"] == 1.0 and s["pass_all_k"] == 1.0
    assert s["tool_use_accuracy"] == 1.0 and not s["flaky_tasks"]


def test_gate_passes_on_identical_runs(good_run):
    assert compare(good_run, good_run, GateConfig()).passed


def test_gate_catches_planted_regression(toy_suite, good_run, buggy_run):
    outcome = compare(good_run, buggy_run, toy_suite.gate)
    assert not outcome.passed
    assert "refund-over-limit-escalates" in outcome.regressions
    assert "Agent quality gate failed" in outcome.markdown
