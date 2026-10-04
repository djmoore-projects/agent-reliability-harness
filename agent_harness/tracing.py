"""Non-invasive tracing and fault injection via LangChain callbacks.

Pass `tracer.callbacks` in the agent's run config and every LangGraph node,
chat-model call, and tool call is recorded as a span — no changes to the
agent's code. LangChain propagates callbacks to nested `.invoke()` calls
through a context var, so tools and models invoked *inside* a node are
captured too.

Agents that aren't built on LangChain can record spans by hand with
`tracer.record_llm(...)` / `tracer.record_tool(...)`.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from . import pricing
from .schema import FaultSpec, Span, Trace

PREVIEW_CHARS = 2000
TOOL_OUTPUT_CHARS = 8000  # tool outputs are the evidence the faithfulness check grades against


def _preview(value: Any, limit: int = PREVIEW_CHARS) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


class InjectedFault(RuntimeError):
    """Raised by FaultInjector. Distinct type so traces can tell it apart from real failures."""


class Tracer(BaseCallbackHandler):
    """Collects a span tree for a single task execution. Thread-safe."""

    def __init__(self, faults: list[FaultSpec] | None = None) -> None:
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()
        self._spans: dict[str, Span] = {}
        self._node_of_run: dict[str, str] = {}
        self.faults = FaultInjector(self, faults or [])

    @property
    def callbacks(self) -> list[BaseCallbackHandler]:
        # Tracer first so a span exists before the injector raises on it.
        return [self, self.faults]

    def _now(self) -> float:
        return time.perf_counter() - self._t0

    def trace(self) -> Trace:
        with self._lock:
            spans = [s.model_copy() for s in self._spans.values()]
        end = self._now()
        for span in spans:
            if span.end is None:  # aborted mid-flight (e.g. an injected fault raised before *_end)
                span.end = end
                span.error = span.error or "span never closed"
        return Trace(spans=sorted(spans, key=lambda s: s.start))

    # --- manual recording API (non-LangChain agents) --------------------------

    def record_llm(self, *, model: str, input_tokens: int, output_tokens: int, start: float, end: float,
                   node: str | None = None, name: str | None = None, error: str | None = None) -> None:
        span = Span(id=uuid.uuid4().hex, kind="llm", name=name or model, node=node, start=start, end=end,
                    model=model, input_tokens=input_tokens, output_tokens=output_tokens,
                    cost_usd=pricing.cost_usd(model, input_tokens, output_tokens), error=error)
        with self._lock:
            self._spans[span.id] = span

    def record_tool(self, *, name: str, start: float, end: float, node: str | None = None,
                    input: Any = None, output: Any = None, error: str | None = None) -> None:
        span = Span(id=uuid.uuid4().hex, kind="tool", name=name, node=node, start=start, end=end,
                    input=_preview(input) if input is not None else None,
                    output=_preview(output, TOOL_OUTPUT_CHARS) if output is not None else None, error=error)
        with self._lock:
            self._spans[span.id] = span

    # --- internal helpers ------------------------------------------------------

    def _open(self, run_id: UUID, parent_run_id: UUID | None, kind: str, name: str,
              metadata: dict | None, **fields: Any) -> Span:
        node = (metadata or {}).get("langgraph_node")
        span = Span(id=str(run_id), parent_id=str(parent_run_id) if parent_run_id else None,
                    kind=kind, name=name, node=node, start=self._now(), **fields)
        with self._lock:
            self._spans[span.id] = span
        return span

    def _close(self, run_id: UUID, **fields: Any) -> None:
        with self._lock:
            span = self._spans.get(str(run_id))
            if span is None:
                return
            span.end = self._now()
            for key, value in fields.items():
                setattr(span, key, value)

    def mark_error(self, run_id: UUID, error: str) -> None:
        self._close(run_id, error=error)

    def add_fault_span(self, name: str, node: str | None, error: str) -> None:
        now = self._now()
        span = Span(id=uuid.uuid4().hex, kind="fault", name=name, node=node, start=now, end=now, error=error)
        with self._lock:
            self._spans[span.id] = span

    # --- LangGraph nodes (chains) ---------------------------------------------

    def on_chain_start(self, serialized: dict | None, inputs: Any, *, run_id: UUID,
                       parent_run_id: UUID | None = None, tags: list[str] | None = None,
                       metadata: dict | None = None, **kwargs: Any) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name", "")
        node = (metadata or {}).get("langgraph_node")
        is_node = node is not None and name == node and any(t.startswith("graph:step:") for t in tags or [])
        # Nested runnables inherit the node's tags/metadata; only the outermost
        # run whose name matches the node is the node itself.
        if is_node and self._node_of_run.get(str(parent_run_id)) != node:
            self._node_of_run[str(run_id)] = node
            self._open(run_id, parent_run_id, "node", node, metadata)
        elif parent_run_id is not None and str(parent_run_id) in self._node_of_run:
            self._node_of_run[str(run_id)] = self._node_of_run[str(parent_run_id)]

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        if str(run_id) in self._spans:
            self._close(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        if str(run_id) in self._spans:
            self._close(run_id, error=f"{type(error).__name__}: {error}")

    # --- chat models ------------------------------------------------------------

    def on_chat_model_start(self, serialized: dict | None, messages: list, *, run_id: UUID,
                            parent_run_id: UUID | None = None, metadata: dict | None = None,
                            **kwargs: Any) -> None:
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name") or (metadata or {}).get("ls_model_name")
        flat = [m for batch in messages for m in batch]
        last = flat[-1].content if flat else ""
        self._open(run_id, parent_run_id, "llm", model or "llm", metadata, model=model, input=_preview(last))

    def on_llm_start(self, serialized: dict | None, prompts: list[str], *, run_id: UUID,
                     parent_run_id: UUID | None = None, metadata: dict | None = None,
                     **kwargs: Any) -> None:
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or (metadata or {}).get("ls_model_name")
        self._open(run_id, parent_run_id, "llm", model or "llm", metadata, model=model,
                   input=_preview(prompts[-1] if prompts else ""))

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        usage: dict = {}
        text = ""
        try:
            message = response.generations[0][0].message
            usage = dict(getattr(message, "usage_metadata", None) or {})
            text = message.content if isinstance(message.content, str) else repr(message.content)
            if getattr(message, "tool_calls", None):
                text = f"tool_calls={[(c['name'], c['args']) for c in message.tool_calls]} {text}"
            model = (getattr(message, "response_metadata", None) or {}).get("model_name")
        except (AttributeError, IndexError):
            model = None
        if not usage:
            usage = ((getattr(response, "llm_output", None) or {}).get("usage") or {})
        details = usage.get("input_token_details") or {}
        cache_read = int(details.get("cache_read") or 0)
        cache_write = int(details.get("cache_creation") or 0)
        # LangChain's input_tokens includes cached tokens; bill only the uncached remainder at full rate.
        uncached = max(0, int(usage.get("input_tokens") or 0) - cache_read - cache_write)
        output_tokens = int(usage.get("output_tokens") or 0)
        with self._lock:
            span = self._spans.get(str(run_id))
            model = model or (span.model if span else None)
            if span is not None and span.name == "llm" and model:
                span.name = model
        self._close(run_id, input_tokens=uncached, output_tokens=output_tokens,
                    cache_read_tokens=cache_read, cache_write_tokens=cache_write, output=_preview(text),
                    model=model,
                    cost_usd=pricing.cost_usd(model, uncached, output_tokens, cache_read, cache_write))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, error=f"{type(error).__name__}: {error}")

    # --- tools --------------------------------------------------------------------

    def on_tool_start(self, serialized: dict | None, input_str: str, *, run_id: UUID,
                      parent_run_id: UUID | None = None, metadata: dict | None = None,
                      **kwargs: Any) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name", "tool")
        self._open(run_id, parent_run_id, "tool", name, metadata, input=_preview(kwargs.get("inputs") or input_str))

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        content = getattr(output, "content", output)
        self._close(run_id, output=_preview(content, TOOL_OUTPUT_CHARS))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, error=f"{type(error).__name__}: {error}")


class FaultInjector(BaseCallbackHandler):
    """Raises InjectedFault at tool/LLM start to simulate outages.

    `raise_error = True` makes LangChain re-raise callback exceptions instead of
    logging them, so the failure surfaces inside the agent exactly where a real
    API error would — exercising the agent's own retry/recovery paths.
    """

    raise_error = True

    def __init__(self, tracer: Tracer, faults: list[FaultSpec]) -> None:
        self._tracer = tracer
        self._remaining = [f.times for f in faults]
        self._faults = faults
        self._lock = threading.Lock()

    def _maybe_fire(self, kind: str, name: str, node: str | None, run_id: UUID) -> None:
        with self._lock:
            for i, fault in enumerate(self._faults):
                if fault.target != kind or (fault.name and fault.name != name) or (fault.node and fault.node != node):
                    continue
                if self._remaining[i] == 0:
                    continue
                if self._remaining[i] > 0:
                    self._remaining[i] -= 1
                break
            else:
                return
        message = f"{fault.error} (injected into {kind} '{name}')"
        self._tracer.add_fault_span(name, node, message)
        self._tracer.mark_error(run_id, message)
        raise InjectedFault(message)

    def on_tool_start(self, serialized: dict | None, input_str: str, *, run_id: UUID,
                      metadata: dict | None = None, **kwargs: Any) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name", "tool")
        self._maybe_fire("tool", name, (metadata or {}).get("langgraph_node"), run_id)

    def on_chat_model_start(self, serialized: dict | None, messages: list, *, run_id: UUID,
                            metadata: dict | None = None, **kwargs: Any) -> None:
        self._maybe_fire("llm", "chat_model", (metadata or {}).get("langgraph_node"), run_id)
