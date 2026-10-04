"""A tiny, deterministic LangGraph refund agent, so the harness runs offline (no API keys).

The "model" is a scripted policy wrapped as a LangChain chat model: it emits
real tool calls and token usage, so tracing, fault injection, and grading
work exactly as they do on a live agent. Set TOY_AGENT_VARIANT=buggy to
plant a regression (skips the order lookup and refunds without checking
policy), which is what the CI gate demo catches.

Policy: refund if delivered within 30 days and amount <= $100; escalate
amounts over $100; politely decline anything older than 30 days.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

ORDERS = {
    "A100": {"status": "delivered", "amount": 42.50, "days_since_delivery": 3},
    "A200": {"status": "delivered", "amount": 349.00, "days_since_delivery": 5},
    "A300": {"status": "delivered", "amount": 25.00, "days_since_delivery": 64},
    "A400": {"status": "in_transit", "amount": 80.00, "days_since_delivery": None},
}

# Side effects land here so checks can grade the environment, not just the transcript.
LEDGER: contextvars.ContextVar[list] = contextvars.ContextVar("ledger")


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order's status, amount, and days since delivery."""
    if order_id not in ORDERS:
        return json.dumps({"error": f"order {order_id} not found"})
    return json.dumps({"order_id": order_id, **ORDERS[order_id]})


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Issue a refund to the customer's original payment method."""
    LEDGER.get().append({"action": "refund", "order_id": order_id, "amount": amount})
    return json.dumps({"ok": True, "refund_id": f"rf_{order_id}"})


@tool
def escalate_to_human(order_id: str, reason: str) -> str:
    """Hand the case to a human agent with a reason."""
    LEDGER.get().append({"action": "escalate", "order_id": order_id, "reason": reason})
    return json.dumps({"ok": True, "ticket": f"t_{order_id}"})


TOOLS = [lookup_order, issue_refund, escalate_to_human]


def _call(name: str, args: dict, n: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{n}"}])


class ScriptedPolicyModel(BaseChatModel):
    """Deterministic stand-in for an LLM that follows (or, when buggy, ignores) the refund policy."""

    variant: str = "good"

    @property
    def _llm_type(self) -> str:
        return "scripted-policy"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedPolicyModel:
        return self

    def _decide(self, messages: list[BaseMessage]) -> AIMessage:
        request = next(m.content for m in messages if m.type == "human")
        order_id = (re.search(r"\b([A-Z]\d{3})\b", request) or [None, "UNKNOWN"])[1]
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        n = len(tool_msgs)

        if self.variant == "buggy" and n == 0:
            return _call("issue_refund", {"order_id": order_id, "amount": ORDERS.get(order_id, {}).get("amount", 0)}, n)
        if n == 0:
            return _call("lookup_order", {"order_id": order_id}, n)

        last = tool_msgs[-1]
        if last.name == "lookup_order":
            if last.status == "error" or "Error" in str(last.content):
                lookups_failed = sum(1 for m in tool_msgs if m.name == "lookup_order" and m.status == "error")
                if lookups_failed < 2:
                    return _call("lookup_order", {"order_id": order_id}, n)  # retry once
                return _call("escalate_to_human", {"order_id": order_id, "reason": "order system unavailable"}, n)
            order = json.loads(last.content)
            if "error" in order:
                return AIMessage(content=f"I couldn't find order {order_id}. Could you double-check the number?")
            if order["status"] != "delivered":
                return AIMessage(content=f"Order {order_id} hasn't been delivered yet, so it isn't eligible for a refund.")
            if order["days_since_delivery"] > 30:
                return AIMessage(content=f"Order {order_id} is outside the 30-day refund window, so I can't refund it.")
            if order["amount"] > 100:
                return _call("escalate_to_human", {"order_id": order_id, "reason": "refund over $100 limit"}, n)
            return _call("issue_refund", {"order_id": order_id, "amount": order["amount"]}, n)
        if last.name == "issue_refund":
            return AIMessage(content=f"Done. Your refund for order {order_id} is on its way.")
        return AIMessage(content=f"I've passed order {order_id} to a specialist who will follow up shortly.")

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        message = self._decide(messages)
        prompt_chars = sum(len(str(m.content)) for m in messages) + 1200  # + a notional system prompt
        message.usage_metadata = {"input_tokens": prompt_chars // 4, "output_tokens": 40,
                                  "total_tokens": prompt_chars // 4 + 40}
        message.response_metadata = {"model_name": "claude-haiku-4-5"}
        return ChatResult(generations=[ChatGeneration(message=message)])


def build_agent(variant: str | None = None):
    model = ScriptedPolicyModel(variant=variant or os.environ.get("TOY_AGENT_VARIANT", "good")).bind_tools(TOOLS)

    def agent(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


__all__ = ["END", "LEDGER", "build_agent"]
