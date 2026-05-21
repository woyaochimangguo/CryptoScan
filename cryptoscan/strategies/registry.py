from __future__ import annotations

from cryptoscan.config import settings

from .base import Strategy
from .oi_funding_flip import OiFundingFlipStrategy


_STRATEGIES: dict[str, Strategy] = {
    "oi_funding_flip": OiFundingFlipStrategy(),
}


def list_strategies() -> list[Strategy]:
    return list(_STRATEGIES.values())


def get_strategy(strategy_id: str) -> Strategy:
    try:
        return _STRATEGIES[strategy_id]
    except KeyError as e:
        raise ValueError(f"unknown strategy: {strategy_id}") from e


def enabled_strategies() -> list[Strategy]:
    raw = settings.enabled_strategies or "oi_funding_flip"
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    return [get_strategy(strategy_id) for strategy_id in ids]
