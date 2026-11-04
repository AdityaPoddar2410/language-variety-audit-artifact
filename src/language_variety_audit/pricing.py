"""Token-price estimates used for experiment cost accounting."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPrice:
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float = 0.0


_PRICES: dict[str, TokenPrice] = {
    "gpt-4o-mini": TokenPrice(0.15, 0.60, 0.075),
    "gpt-4o-mini-2024-07-18": TokenPrice(0.15, 0.60, 0.075),
    "gpt-4.1-mini": TokenPrice(0.40, 1.60, 0.10),
    "gpt-4.1-mini-2025-04-14": TokenPrice(0.40, 1.60, 0.10),
    "gemini-2.5-flash-lite": TokenPrice(0.10, 0.40),
}


def token_price(model: str) -> TokenPrice | None:
    return _PRICES.get(model)


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float | None:
    price = token_price(model)
    if price is None:
        return None
    uncached = max(input_tokens - cached_input_tokens, 0)
    return (
        uncached * price.input_per_million
        + cached_input_tokens * price.cached_input_per_million
        + output_tokens * price.output_per_million
    ) / 1_000_000
