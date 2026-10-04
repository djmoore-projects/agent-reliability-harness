"""Per-model token pricing (USD per million tokens) for cost-per-task.

First-party Anthropic API list prices. Cache reads bill at 0.1x input and
5-minute cache writes at 1.25x input. Override or extend with
`register_price()` for other providers or negotiated rates.
"""

from __future__ import annotations

PRICES: dict[str, tuple[float, float]] = {
    # model id prefix: (input $/MTok, output $/MTok)
    "claude-fable-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def register_price(model_prefix: str, input_per_mtok: float, output_per_mtok: float) -> None:
    PRICES[model_prefix] = (input_per_mtok, output_per_mtok)


def lookup(model: str | None) -> tuple[float, float] | None:
    if not model:
        return None
    # Longest prefix wins, so "claude-opus-5-5" isn't priced as "claude-opus-5".
    for prefix in sorted(PRICES, key=len, reverse=True):
        if model.startswith(prefix):
            return PRICES[prefix]
    return None


def cost_usd(model: str | None, input_tokens: int, output_tokens: int,
             cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
    price = lookup(model)
    if price is None:
        return 0.0
    in_price, out_price = price
    return (
        input_tokens * in_price
        + output_tokens * out_price
        + cache_read_tokens * in_price * CACHE_READ_MULTIPLIER
        + cache_write_tokens * in_price * CACHE_WRITE_MULTIPLIER
    ) / 1_000_000
