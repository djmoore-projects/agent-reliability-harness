"""Graders. Each check takes a TaskContext and returns a CheckResult.

Checks are declared in YAML by `type`, and register themselves with `@check`.
Deterministic checks (tool use, state, budgets) run first and cost nothing.
Judge-backed checks (`judge`, `faithfulness`) call Claude.

Paths address the run output: `state.*` (the agent's final state), `env.*`
(side effects the adapter observed, e.g. records written to a CRM), and
`input.*` (the task input). Dotted segments index into dicts and lists, so
`state.outreach_sequence.0.channel` works.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .judge import Judge
from .schema import CheckCategory, CheckResult, Task, Trace

_MISSING = object()


@dataclass
class TaskContext:
    task: Task
    output: dict[str, Any]
    trace: Trace
    latency_s: float
    cost_usd: float
    judge: Judge | None = None
    tool_costs: dict[str, float] = field(default_factory=dict)

    def resolve(self, path: str) -> Any:
        root: Any = {"state": self.output.get("state"), "env": self.output.get("env"), "input": self.task.input}
        for part in path.split("."):
            if isinstance(root, dict):
                root = root.get(part, _MISSING)
            elif isinstance(root, list) and part.lstrip("-").isdigit() and -len(root) <= int(part) < len(root):
                root = root[int(part)]
            else:
                return _MISSING
            if root is _MISSING:
                return _MISSING
        return root


CheckFn = Callable[[TaskContext, dict], tuple[bool, str, float | None]]
_REGISTRY: dict[str, tuple[CheckCategory, CheckFn]] = {}


def check(type_name: str, category: CheckCategory) -> Callable[[CheckFn], CheckFn]:
    def register(fn: CheckFn) -> CheckFn:
        _REGISTRY[type_name] = (category, fn)
        return fn
    return register


def run_check(ctx: TaskContext, spec: dict) -> CheckResult:
    type_name = spec.get("type", "")
    name = spec.get("name") or _default_name(spec)
    required = spec.get("required", True)
    if type_name not in _REGISTRY:
        return CheckResult(name=name, type=type_name, category="state", passed=False, required=required,
                           error=f"unknown check type '{type_name}'")
    category, fn = _REGISTRY[type_name]
    category = spec.get("category", category)
    try:
        passed, detail, score = fn(ctx, spec)
    except Exception as exc:  # noqa: BLE001 - a broken check is a failed check, not a crashed run
        return CheckResult(name=name, type=type_name, category=category, passed=False, required=required,
                           detail=f"check errored: {exc}", error=f"{type(exc).__name__}: {exc}")
    return CheckResult(name=name, type=type_name, category=category, passed=passed, required=required,
                       detail=detail, score=score)


def _default_name(spec: dict) -> str:
    bits = [spec.get("type", "?")]
    for key in ("tool", "path", "field"):
        if key in spec:
            bits.append(str(spec[key]))
    return ":".join(bits)


def _fmt(value: Any, limit: int = 120) -> str:
    text = "<missing>" if value is _MISSING else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


# --- value predicates shared by `field` and `each` ----------------------------


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


PREDICATES: dict[str, Callable[[Any, Any], bool]] = {
    "equals": lambda v, x: v == x,
    "not_equals": lambda v, x: v != x,
    "in": lambda v, x: v in x,
    "exists": lambda v, x: (v is not _MISSING) == bool(x),
    "is_null": lambda v, x: (v is None) == bool(x),
    "not_empty": lambda v, x: bool(v) == bool(x) and v is not _MISSING,
    "matches": lambda v, x: v is not _MISSING and re.search(x, _text(v), re.IGNORECASE) is not None,
    "not_matches": lambda v, x: v is _MISSING or re.search(x, _text(v), re.IGNORECASE) is None,
    "contains_any": lambda v, x: v is not _MISSING and any(k.lower() in _text(v).lower() for k in x),
    "contains_none": lambda v, x: v is _MISSING or not any(k.lower() in _text(v).lower() for k in x),
    "min_len": lambda v, x: v is not _MISSING and v is not None and len(v) >= x,
    "max_len": lambda v, x: v is not _MISSING and v is not None and len(v) <= x,
    "max_words": lambda v, x: isinstance(v, str) and len(v.split()) <= x,
    "max_chars": lambda v, x: isinstance(v, str) and len(v) <= x,
    "gte": lambda v, x: isinstance(v, (int, float)) and v >= x,
    "lte": lambda v, x: isinstance(v, (int, float)) and v <= x,
}


def _apply_predicates(value: Any, spec: dict) -> list[str]:
    """Return a list of failed predicate descriptions (empty = pass)."""
    failures = []
    for key, expected in spec.items():
        if key in PREDICATES and not PREDICATES[key](value, expected):
            failures.append(f"{key}={_fmt(expected, 60)}")
    return failures


# --- built-in checks -----------------------------------------------------------


@check("tool_calls", "tool_use")
def _tool_calls(ctx: TaskContext, spec: dict):
    """Right tool, right number of times. `min`/`max` bound the call count."""
    calls = ctx.trace.of("tool", spec["tool"])
    n = len(calls)
    lo, hi = spec.get("min", 0), spec.get("max")
    ok = n >= lo and (hi is None or n <= hi)
    bound = f"[{lo}, {hi if hi is not None else '∞'}]"
    return ok, f"{spec['tool']} called {n}x, expected {bound}", None


@check("tools_used", "tool_use")
def _tools_used(ctx: TaskContext, spec: dict):
    """Tool selection: the set of tools called must include `must_use` and exclude `must_not_use`."""
    used = {s.name for s in ctx.trace.tool_calls}
    missing = set(spec.get("must_use", [])) - used
    forbidden = set(spec.get("must_not_use", [])) & used
    ok = not missing and not forbidden
    detail = f"used={sorted(used)}" + (f" missing={sorted(missing)}" if missing else "") + (
        f" forbidden={sorted(forbidden)}" if forbidden else "")
    return ok, detail, None


@check("tool_args", "tool_use")
def _tool_args(ctx: TaskContext, spec: dict):
    """Every (or `any`) call to `tool` has input matching the predicates (e.g. matches: 'Stripe')."""
    calls = ctx.trace.of("tool", spec["tool"])
    if not calls:
        return False, f"{spec['tool']} never called", None
    results = [not _apply_predicates(c.input or "", spec) for c in calls]
    ok = any(results) if spec.get("any") else all(results)
    return ok, f"{sum(results)}/{len(calls)} calls matched", None


@check("node_path", "tool_use")
def _node_path(ctx: TaskContext, spec: dict):
    """Control flow: exact node sequence, or must include / exclude nodes."""
    path = ctx.trace.node_path
    ok = True
    if "equals" in spec:
        ok &= path == spec["equals"]
    for node in spec.get("includes", []):
        ok &= node in path
    for node in spec.get("excludes", []):
        ok &= node not in path
    if "max_visits" in spec:
        ok &= all(path.count(n) <= limit for n, limit in spec["max_visits"].items())
    return ok, " → ".join(path) or "<no nodes>", None


@check("field", "state")
def _field(ctx: TaskContext, spec: dict):
    """Assert on a value in final state / environment with any predicates in PREDICATES."""
    value = ctx.resolve(spec["path"])
    failures = _apply_predicates(value, spec)
    return not failures, (f"failed {', '.join(failures)}; " if failures else "") + f"value={_fmt(value)}", None


@check("each", "state")
def _each(ctx: TaskContext, spec: dict):
    """Apply predicates to `field` of every item in the list at `path`, optionally filtered by `where`."""
    items = ctx.resolve(spec["path"])
    if not isinstance(items, list) or not items:
        return False, f"{spec['path']} is not a non-empty list", None
    where = spec.get("where", {})
    selected = [i for i in items if all(isinstance(i, dict) and i.get(k) == v for k, v in where.items())]
    if not selected:
        return False, f"no items matched where={where}", None
    bad = []
    for idx, item in enumerate(selected):
        value = item.get(spec["field"], _MISSING) if spec.get("field") else item
        failures = _apply_predicates(value, spec)
        if failures:
            bad.append(f"#{idx} {', '.join(failures)} (got {_fmt(value, 40) if not isinstance(value, str) else str(len(value.split())) + 'w/' + str(len(value)) + 'c'})")
    return not bad, f"{len(selected) - len(bad)}/{len(selected)} ok" + (f"; {'; '.join(bad)}" if bad else ""), None


@check("budget", "efficiency")
def _budget(ctx: TaskContext, spec: dict):
    """Cost / latency / step ceilings for a single task."""
    trace = ctx.trace
    measured = {
        "max_cost_usd": ctx.cost_usd,
        "max_latency_s": ctx.latency_s,
        "max_steps": trace.steps,
        "max_llm_calls": len(trace.llm_calls),
        "max_tool_calls": len(trace.tool_calls),
    }
    over = [f"{k}: {measured[k]:.3g} > {spec[k]}" for k in measured if k in spec and measured[k] > spec[k]]
    used = ", ".join(f"{k.removeprefix('max_')}={measured[k]:.3g}" for k in measured if k in spec)
    return not over, ("; ".join(over) if over else used), None


@check("no_crash", "state")
def _no_crash(ctx: TaskContext, spec: dict):
    crashed = ctx.output.get("__crashed__")
    return crashed is None, crashed or "returned normally", None


@check("judge", "quality")
def _judge(ctx: TaskContext, spec: dict):
    """LLM-as-judge against a rubric. `inputs` is a list of paths rendered as the material."""
    if ctx.judge is None:
        raise RuntimeError("judge checks disabled (--no-judge)")
    material = "\n\n".join(f"<{p}>\n{_text(ctx.resolve(p))}\n</{p}>" for p in spec["inputs"])
    verdict = ctx.judge.rubric(spec["rubric"], material)
    return verdict.passed, verdict.reasoning, None


@check("faithfulness", "quality")
def _faithfulness(ctx: TaskContext, spec: dict):
    """Fraction of factual claims at `path` supported by what the agent's tools actually returned.

    The evidence is pulled from the trace (tool outputs), so this catches facts
    the agent invented rather than retrieved. Passes if score >= `threshold`.
    """
    if ctx.judge is None:
        raise RuntimeError("judge checks disabled (--no-judge)")
    tools = spec.get("evidence_tools")
    evidence = "\n\n".join(
        f"[{s.name} input={s.input}]\n{s.output}" for s in ctx.trace.tool_calls
        if s.output and not s.error and (tools is None or s.name in tools)
    ) or "(the agent retrieved no evidence)"
    answer = _text(ctx.resolve(spec["path"]))
    verdict = ctx.judge.faithfulness(answer, evidence)
    claims = verdict.claims
    if not claims:
        return True, "no factual claims", 1.0
    supported = sum(c.supported for c in claims)
    score = supported / len(claims)
    unsupported = [c.claim for c in claims if not c.supported][:4]
    detail = f"{supported}/{len(claims)} claims supported" + (f"; unsupported: {unsupported}" if unsupported else "")
    return score >= spec.get("threshold", 0.8), detail, score


@check("custom", "state")
def _custom(ctx: TaskContext, spec: dict):
    """`fn: module:function` taking (ctx, spec) and returning (passed, detail)."""
    module, func = spec["fn"].split(":")
    passed, detail = getattr(importlib.import_module(module), func)(ctx, spec)
    return passed, detail, None
