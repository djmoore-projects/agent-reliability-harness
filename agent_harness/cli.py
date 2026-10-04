"""`agent-harness` command line: run, compare (CI gate), report."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .compare import compare
from .report import to_html, to_markdown
from .runner import load_suite, run_suite
from .schema import GateConfig, RunResult


def _load_run(path: str) -> RunResult:
    return RunResult.model_validate_json(Path(path).read_text())


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass
    suite = load_suite(args.suite)
    config = dict(kv.split("=", 1) for kv in args.config or [])
    run = run_suite(suite, label=args.label, trials=args.trials, concurrency=args.concurrency,
                    use_judge=not args.no_judge, task_filter=args.task, config=config,
                    repo_dir=Path(args.repo_dir) if args.repo_dir else None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(run.model_dump_json(indent=1))
    md = to_markdown(run)
    out.with_suffix(".md").write_text(md)
    out.with_suffix(".html").write_text(to_html(run))
    print(md.split("## Per task")[0])
    print(f"Wrote {out}, {out.with_suffix('.md')}, {out.with_suffix('.html')}")
    infra = [t for t in run.trials if t.infra_error]
    if infra:
        print(f"ERROR: {len(infra)} trial(s) hit infrastructure errors (e.g. {infra[0].infra_error}); results are incomplete.")
        return 2
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    baseline, candidate = _load_run(args.baseline), _load_run(args.candidate)
    gate = load_suite(args.suite).gate if args.suite else GateConfig()
    overrides = {k: getattr(args, k) for k in GateConfig.model_fields if getattr(args, k, None) is not None}
    gate = gate.model_copy(update=overrides)
    outcome = compare(baseline, candidate, gate)
    print(outcome.markdown)
    if args.markdown_out:
        Path(args.markdown_out).write_text(outcome.markdown)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as f:
            f.write(outcome.markdown + "\n")
    return 0 if outcome.passed else 1


def _cmd_report(args: argparse.Namespace) -> int:
    run = _load_run(args.run)
    Path(args.run).with_suffix(".md").write_text(to_markdown(run))
    Path(args.run).with_suffix(".html").write_text(to_html(run))
    print(to_markdown(run))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-harness", description="Evals, tracing and regression gates for tool-using agents.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a task suite and write a run JSON (+ .md/.html reports)")
    run.add_argument("suite")
    run.add_argument("--out", required=True)
    run.add_argument("--label", default="run")
    run.add_argument("--trials", type=int, default=1, help="repeat each task k times (enables pass^k)")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--task", action="append", help="only run this task id (repeatable)")
    run.add_argument("--no-judge", action="store_true", help="skip LLM-judge checks (they then count as failed)")
    run.add_argument("--config", action="append", help="key=value recorded in the run metadata")
    run.add_argument("--repo-dir", help="git repo of the agent under test, for recording its SHA")
    run.set_defaults(fn=_cmd_run)

    cmp_ = sub.add_parser("compare", help="compare candidate vs baseline; exit 1 if the gate fails")
    cmp_.add_argument("baseline")
    cmp_.add_argument("candidate")
    cmp_.add_argument("--suite", help="read gate thresholds from this suite's `gate:` block")
    cmp_.add_argument("--markdown-out")
    cmp_.add_argument("--min-success-rate", dest="min_success_rate", type=float)
    cmp_.add_argument("--max-success-drop", dest="max_success_drop", type=float)
    cmp_.add_argument("--max-cost-increase-pct", dest="max_cost_increase_pct", type=float)
    cmp_.add_argument("--max-latency-increase-pct", dest="max_latency_increase_pct", type=float)
    cmp_.set_defaults(fn=_cmd_compare)

    rep = sub.add_parser("report", help="re-render .md/.html reports from a run JSON")
    rep.add_argument("run")
    rep.set_defaults(fn=_cmd_report)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Injected faults are expected; LangChain logs each one as a warning before re-raising it.
    logging.getLogger("langchain_core.callbacks.manager").setLevel(logging.ERROR)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
