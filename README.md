# agent-reliability-harness

[![CI](https://github.com/djmoore-projects/agent-reliability-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/djmoore-projects/agent-reliability-harness/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Evals, tracing and CI regression gates for tool-using agents.** Point it at a
task suite and it answers the question every agent project eventually gets
asked — *"how do we know it works?"* — with numbers: did the agent pick the
right tools, is the final state (and the real-world side effect) correct, what
did each task cost, how long did it take, how many steps did it need, and does
it still pass when an API falls over halfway through. Then it runs as a CI
check that fails the PR when any of that gets worse.

It's the agent-shaped sequel to [fintech-rag-eval](https://github.com/djmoore-projects/fintech-rag-eval):
that project grades RAG *answers*; this one grades agent *runs*. Same
philosophy — typed contracts, structured judge outputs, scores computed in
Python rather than reported by a model.

## First real target: b2b-account-pipeline

The harness was built against
[b2b-account-pipeline](https://github.com/djmoore-projects/b2b-account-pipeline),
a LangGraph agent that researches a company with Claude + Tavily, writes a
3-step outreach sequence and syncs a CRM record. Its README says it "works"
and shows a clean run. The 11-task suite
([`evals/suite.yaml`](https://github.com/djmoore-projects/b2b-account-pipeline/blob/agent-reliability-evals/evals/suite.yaml))
checks only the contract the pipeline states for itself, and the checks were
frozen before the baseline ran.

**Baseline: 17% task success** (3/18 clean trials, 95% CI 6–39%). **All 13 clean
happy-path trials failed:**

- `[Name]` / `[Your Name]` placeholders in outreach that is meant to be sent as-is: **13/13**
- More than the stated 3 searches: **4/13**. The loop capped *iterations*, but Claude makes parallel tool calls inside one iteration.
- Research claims not supported by what the searches returned (faithfulness < 0.8): **3/14**
- A fictional company was assigned a similarly named real company's executives, funding and news
- One injected LLM error in the outreach step re-ran all research: **8 searches, 108s, $0.41** against a typical $0.18

The fault-injection tasks passed: the agent retries a transient search outage
and fails cleanly when search is down for the whole run.

The baseline run is **partial**. The API key ran out of credit partway through,
and the harness flagged the 15 affected trials as infrastructure errors and
excluded them instead of counting them against the agent (see design notes).
The fixes are in
[the PR](https://github.com/djmoore-projects/b2b-account-pipeline/pulls): a search-count budget, resume from the failed node,
a code-enforced outreach contract with one repair pass, and an entity-grounding
instruction. **The after-fix run is pending.** This README will only show
after-fix numbers once that run has been done.


## What it measures

| Dimension | How | Check types |
|---|---|---|
| **Tool selection** | Which tools were called, how often, with what arguments, in what control-flow path | `tool_calls`, `tools_used`, `tool_args`, `node_path` |
| **Final state** | Assertions on the agent's output state *and* on the environment it changed (e.g. the CRM record it actually wrote) | `field`, `each`, `custom` |
| **Quality** | LLM-as-judge on a rubric, and claim-level faithfulness of the output against what the agent's tools actually returned | `judge`, `faithfulness` |
| **Cost / latency / steps** | Token usage × per-model pricing + per-call tool pricing, wall clock, LLM + tool call counts | `budget` |
| **Reliability** | Every task runs *k* times: pass@1, **pass^k** (passed every trial), flaky-task detection, Wilson 95% CIs | built in |
| **Robustness** | Inject tool or LLM failures at a chosen node and check the agent recovers (or fails cleanly) | `faults:` on a task |

A task succeeds only if every required check passes. The suite reports
success rate, per-category check pass rates, cost, p50/p95 latency and
steps, plus a per-task breakdown and the most common failing checks.

## How it works

```
 suite.yaml ──► runner ──► adapter(task_input, tracer) ──► your agent (unchanged)
                  │                     │                        │
                  │            tracer.callbacks ◄── LangChain callback events
                  │            (nodes, LLM calls, tool calls, tokens, errors)
                  │            fault injector ──► raises inside the agent
                  ▼
        checks: deterministic first (free) → judge checks (Claude)
                  ▼
        run.json + report.md + report.html (trace waterfall)
                  ▼
        compare(baseline.json, candidate.json) ──► PR comment, exit 1 on regression
```

- **Non-invasive tracing.** The tracer is a LangChain callback handler. Pass
  `tracer.callbacks` into the run config and every LangGraph node, chat-model
  call and tool call becomes a span with timing, tokens, cost and errors —
  including calls made *inside* node functions, because LangChain propagates
  callbacks through a context var. No decorators, no SDK wrapping, no changes
  to the agent. Non-LangChain agents can call `tracer.record_llm()` /
  `tracer.record_tool()` directly.
- **Fault injection through the same channel.** A second callback handler with
  `raise_error = True` throws at `on_tool_start` / `on_chat_model_start`, so an
  injected "Tavily 503" or "Anthropic 529" surfaces exactly where a real one
  would — exercising the agent's own retry logic, not a mock of it. Target by
  tool name and/or LangGraph node, fire once or every time.
- **Environment, not just transcript.** Adapters return `{"state", "env"}`.
  `env` is whatever side effects the adapter observed (records written,
  refunds issued), so checks can assert the agent *did* the thing, not that it
  *said* it did.
- **Faithfulness against the trace.** The `faithfulness` check pulls tool
  outputs from the trace as evidence, has Claude split the agent's output
  into atomic claims and verify each, and scores `supported / total` in
  Python — fintech-rag-eval's faithfulness metric, with the agent's own
  retrievals as the context.
- **Gate, not dashboard.** `agent-harness compare` diffs two runs and exits
  non-zero when success drops, cost or p95 latency rises past a threshold, or
  any task that passed in the baseline now fails. Writes the report to the
  GitHub step summary and (via the bundled action) a sticky PR comment.

## Quickstart (offline, no API keys)

```bash
git clone https://github.com/djmoore-projects/agent-reliability-harness.git
cd agent-reliability-harness
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# A deterministic LangGraph refund agent with a scripted model.
agent-harness run examples/toy_support_agent/suite.yaml --out runs/good.json --trials 3 --no-judge

# Plant a regression and watch the gate catch it.
TOY_AGENT_VARIANT=buggy agent-harness run examples/toy_support_agent/suite.yaml --out runs/buggy.json --trials 3 --no-judge
agent-harness compare runs/good.json runs/buggy.json --suite examples/toy_support_agent/suite.yaml   # exit 1
open runs/good.html   # trace waterfall per task
```

## Writing a suite

```yaml
name: my-agent
adapter: adapter:run          # module:function, resolved next to this file
tool_costs: { web_search: 0.008 }
judge_model: claude-opus-5-5
gate: { min_success_rate: 0.8, max_success_drop: 0.05, max_cost_increase_pct: 25 }

shared_checks:                # applied to every task; ${var} is filled from task input
  - { type: tool_calls, tool: web_search, min: 1, max: 3 }
  - { type: field, path: env.crm_records.0.company_name, equals: "${company_name}" }
  - { type: each, path: state.outreach_sequence, where: { channel: email }, field: body, max_words: 120 }
  - { type: faithfulness, path: state.research_data, evidence_tools: [web_search], threshold: 0.8 }
  - { type: budget, max_cost_usd: 0.75, max_latency_s: 150, max_steps: 10 }

tasks:
  - { id: stripe, input: { company_name: Stripe } }
  - id: search-outage-recovers
    input: { company_name: Datadog }
    faults: [{ target: tool, name: web_search, times: 1, error: "Tavily 503" }]
    checks: [{ type: field, path: state.crm_status, equals: synced }]
```

The adapter is a few lines:

```python
def run(task_input: dict, tracer) -> dict:
    app = build_graph()
    final = app.invoke(initial_state(task_input), config={"callbacks": tracer.callbacks})
    return {"state": final, "env": {"crm_records": read_crm()}}
```

Field predicates: `equals`, `not_equals`, `in`, `exists`, `is_null`,
`not_empty`, `matches`, `not_matches`, `contains_any`, `contains_none`,
`min_len`, `max_len`, `max_words`, `max_chars`, `gte`, `lte`. Paths are dotted
and index lists (`state.outreach_sequence.0.channel`). Register your own check
with `@agent_harness.check("name", "state")`, or reference a function with
`{type: custom, fn: module:function}`.

## CLI

```bash
agent-harness run SUITE --out run.json [--trials K] [--concurrency N] [--task ID ...] [--no-judge]
agent-harness compare BASELINE.json CANDIDATE.json [--suite SUITE] [--markdown-out gate.md]
agent-harness report run.json            # re-render .md / .html
```

## CI

The repo ships a composite action. In the agent's repo:

```yaml
- uses: djmoore-projects/agent-reliability-harness@main
  with:
    suite: evals/suite.yaml
    baseline: evals/results/baseline.json
    trials: "1"
```

It runs the suite, compares against the committed baseline, uploads the
JSON/Markdown/HTML reports as an artifact, upserts a PR comment, and fails
the job if the gate fails. This repo's own CI dogfoods it on the toy agent:
the current agent must pass against the committed baseline, and a planted
regression must be blocked.

## Design notes

- **pass^k over pass@1.** A demo needs an agent to work once; production
  needs it to work every time. Running each task *k* times and reporting the
  share that passed *all* trials surfaces flakiness that a single run hides.
- **Confidence intervals on everything.** Agent suites are small. A Wilson
  interval next to the success rate keeps a 3-task swing from being sold as
  a breakthrough.
- **Deterministic checks first, judges last.** Cheap structural checks run on
  every trial; judge calls are skipped when the run already crashed, and judge
  spend is tracked separately from agent spend.
- **Infrastructure errors aren't agent failures.** Auth, billing and quota
  errors from the model provider are detected in the trace. The trial is
  flagged, excluded from every metric, and the remaining trials are skipped.
  `run` exits 2, and `compare` refuses to pass a gate on an incomplete run.
  This came out of a real run that ran out of API credit halfway through.
- **Broken checks fail closed.** An unknown check type or a judge error is a
  failed check with the error attached, never a silent pass.
- **Judge on Claude with structured outputs.** `messages.parse` with Pydantic
  verdict schemas; server-side refusal fallback enabled.

## Tests

```bash
pytest -q    # offline: tracer, fault injection, checks, stats, gate
```

## License

[MIT](LICENSE)
