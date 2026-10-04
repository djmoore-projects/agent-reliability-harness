"""Adapter: run the toy agent on one task and return {state, env} for grading."""

from __future__ import annotations

from agent import LEDGER, build_agent


def run(task_input: dict, tracer) -> dict:
    ledger: list = []
    token = LEDGER.set(ledger)
    try:
        app = build_agent()
        final = app.invoke({"messages": [("human", task_input["message"])]},
                           config={"callbacks": tracer.callbacks, "recursion_limit": 12})
    finally:
        LEDGER.reset(token)
    return {
        "state": {"reply": final["messages"][-1].content},
        "env": {"ledger": ledger},
    }
