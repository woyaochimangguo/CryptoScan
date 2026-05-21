from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from cryptoscan.harness.agent import Decision, Policy


@dataclass(frozen=True)
class Signal:
    """Raw strategy candidate before policy/model decision."""

    strategy_id: str
    symbol: str
    trigger: str
    data: dict[str, Any] = field(default_factory=dict)


class Strategy(Protocol):
    id: str
    name: str
    version: str
    trigger: str
    default_policy_id: str

    def scan(self, **kwargs: Any) -> list[Signal]: ...

    def warm(self, signals: list[Signal]) -> None: ...

    def policy(self, mode: str = "dual") -> Policy: ...

    def decide(self, snapshot: dict[str, Any], mode: str = "dual") -> Decision: ...
