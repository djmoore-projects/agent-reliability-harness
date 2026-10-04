"""LLM-as-judge on Claude with structured outputs.

Same principle as fintech-rag-eval: the judge returns per-unit verdicts as
typed JSON and Python computes the score. The model never reports a final
number on its own.
"""

from __future__ import annotations

import threading
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from . import pricing

T = TypeVar("T", bound=BaseModel)


class RubricVerdict(BaseModel):
    reasoning: str
    passed: bool


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    evidence: str  # quote or "none"


class FaithfulnessVerdict(BaseModel):
    claims: list[ClaimVerdict]


JUDGE_SYSTEM = (
    "You are a strict evaluator of AI agent outputs. Judge only against the rubric and "
    "evidence you are given. Do not reward effort or plausibility; if the rubric is not "
    "clearly met, it fails."
)


class Judge:
    def __init__(self, model: str = "claude-opus-5-5", client: anthropic.Anthropic | None = None) -> None:
        self.model = model
        self._client = client or anthropic.Anthropic(max_retries=4)
        self._lock = threading.Lock()
        self.cost_usd = 0.0

    def _parse(self, prompt: str, schema: type[T]) -> T:
        response = self._client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": "medium"},
            # Server-side fallback: if the judge model declines, the API reroutes by category.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        usage = response.usage
        with self._lock:
            self.cost_usd += pricing.cost_usd(
                response.model, usage.input_tokens, usage.output_tokens,
                usage.cache_read_input_tokens or 0, usage.cache_creation_input_tokens or 0,
            )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise RuntimeError(f"judge returned no verdict (stop_reason={response.stop_reason})")
        return response.parsed_output

    def rubric(self, rubric: str, material: str) -> RubricVerdict:
        prompt = (
            f"<rubric>\n{rubric}\n</rubric>\n\n<agent_output>\n{material}\n</agent_output>\n\n"
            "Explain your reasoning briefly, then decide whether the agent output passes the rubric."
        )
        return self._parse(prompt, RubricVerdict)

    def faithfulness(self, answer: str, evidence: str) -> FaithfulnessVerdict:
        prompt = (
            "Break the agent's answer into atomic factual claims (specific facts: names, titles, "
            "numbers, dates, events; skip opinions and generic statements). For each claim, decide "
            "whether the evidence explicitly supports it, and quote the supporting evidence or "
            "write 'none'.\n\n"
            f"<evidence>\n{evidence}\n</evidence>\n\n<answer>\n{answer}\n</answer>"
        )
        return self._parse(prompt, FaithfulnessVerdict)
