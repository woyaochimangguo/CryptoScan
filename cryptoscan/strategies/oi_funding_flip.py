from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cryptoscan.harness.agent import Decision, Policy, rule_policy
from cryptoscan.harness.dual_policy import DualPolicy
from cryptoscan.harness.llm_policy import LLMPolicy
from cryptoscan.tools.derivs_scanner import scan_oi_funding_flip

from .base import Signal


@dataclass(frozen=True)
class OiFundingFlipStrategy:
    """Funding turns negative while OI keeps rising: short-squeeze candidate."""

    id: str = "oi_funding_flip"
    name: str = "OI + funding flip"
    version: str = "1.0.0"
    trigger: str = "oi_funding_flip"
    default_policy_id: str = "dual"

    def scan(self, **kwargs: Any) -> list[Signal]:
        raw = scan_oi_funding_flip(min_volume_usdt=kwargs.get("min_volume_usdt"))
        return [
            Signal(
                strategy_id=self.id,
                symbol=str(item["symbol"]),
                trigger=self.trigger,
                data=dict(item),
            )
            for item in raw
        ]

    def warm(self, signals: list[Signal]) -> None:
        if not signals:
            return
        from cryptoscan.tools.binance_market import market_caps, prefetch_square_hashtags, spot_symbols

        market_caps()
        spot_symbols()
        prefetch_square_hashtags([sig.symbol.replace("USDT", "") for sig in signals])

    def policy(self, mode: str = "dual") -> Policy:
        if mode == "rule":
            return rule_policy
        if mode == "llm":
            return LLMPolicy()
        return DualPolicy(llm_policy=LLMPolicy())

    def decide(self, snapshot: dict[str, Any], mode: str = "dual") -> Decision:
        return self.policy(mode)(snapshot)
