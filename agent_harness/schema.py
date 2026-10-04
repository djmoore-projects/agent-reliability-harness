"""Pydantic contracts: task suites, traces, check results, and run results.

Everything the harness produces is one of these models, so a run serializes
to a single JSON file that `compare` and `report` can read back later (in CI,
on another machine, months from now).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CheckCategory = Literal["tool_use", "state", "efficiency", "quality"]

# --- Suite definition (loaded from YAML) -------------------------------------


class FaultSpec(BaseModel):
    """A failure to inject into the agent's run via the callback layer."""

    target: Literal["tool", "llm"]
    name: str | None = None  # tool name (target: tool); ignored for llm
    node: str | None = None  # only fire inside this LangGraph node
    times: int = 1  # how many times to fire; -1 = every time
    error: str = "Injected fault"


class Task(BaseModel):
    id: str
    description: str = ""
    input: dict[str, Any]
    checks: list[dict[str, Any]] = Field(default_factory=list)
    faults: list[FaultSpec] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class GateConfig(BaseModel):
    """Thresholds `agent-harness compare` enforces. Any breach fails CI."""

    min_success_rate: float | None = None
    max_success_drop: float | None = 0.05  # absolute, e.g. 0.05 = 5 points
    max_cost_increase_pct: float | None = 25.0
    max_latency_increase_pct: float | None = 50.0
    fail_on_task_regression: bool = True  # a task that passed in baseline now fails


class Suite(BaseModel):
    name: str
    adapter: str  # "module.path:function"
    defaults: dict[str, Any] = Field(default_factory=dict)  # merged into every task input
    shared_checks: list[dict[str, Any]] = Field(default_factory=list)  # appended to every task
    tool_costs: dict[str, float] = Field(default_factory=dict)  # USD per tool call
    judge_model: str = "claude-opus-5-5"
    gate: GateConfig = Field(default_factory=GateConfig)
    tasks: list[Task]


# --- Trace -------------------------------------------------------------------


class Span(BaseModel):
    id: str
    parent_id: str | None = None
    kind: Literal["node", "llm", "tool", "fault"]
    name: str
    node: str | None = None  # enclosing LangGraph node, if any
    start: float  # seconds since task start
    end: float | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    input: str | None = None  # truncated preview
    output: str | None = None  # truncated preview
    error: str | None = None

    @property
    def duration(self) -> float:
        return (self.end or self.start) - self.start


class Trace(BaseModel):
    spans: list[Span] = Field(default_factory=list)

    def of(self, kind: str, name: str | None = None) -> list[Span]:
        return [s for s in self.spans if s.kind == kind and (name is None or s.name == name)]

    @property
    def node_path(self) -> list[str]:
        return [s.name for s in sorted(self.of("node"), key=lambda s: s.start)]

    @property
    def tool_calls(self) -> list[Span]:
        return self.of("tool")

    @property
    def llm_calls(self) -> list[Span]:
        return self.of("llm")

    @property
    def steps(self) -> int:
        return len(self.llm_calls) + len(self.tool_calls)

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def tokens(self) -> tuple[int, int]:
        return (sum(s.input_tokens for s in self.llm_calls), sum(s.output_tokens for s in self.llm_calls))


# --- Results -----------------------------------------------------------------


class CheckResult(BaseModel):
    name: str
    type: str
    category: CheckCategory
    passed: bool
    required: bool = True
    detail: str = ""
    score: float | None = None  # for graded checks (e.g. faithfulness ratio)
    error: str | None = None  # the check itself failed to run


class TrialResult(BaseModel):
    task_id: str
    trial: int
    success: bool
    checks: list[CheckResult]
    latency_s: float
    cost_usd: float
    judge_cost_usd: float = 0.0
    steps: int
    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    node_path: list[str]
    output: dict[str, Any] = Field(default_factory=dict)
    trace: Trace
    crashed: str | None = None  # adapter raised instead of returning


class RunResult(BaseModel):
    suite: str
    label: str
    created_at: str
    git_sha: str | None = None
    trials_per_task: int
    config: dict[str, Any] = Field(default_factory=dict)
    trials: list[TrialResult]
