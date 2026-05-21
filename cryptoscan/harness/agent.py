"""Harness Agent: the central decision loop.

MVP design:
- Triggers come from tools (e.g. derivs scanner) and produce candidate signals.
- For each candidate, build a full snapshot, run a *policy* to decide
  long/short/skip/watch, and persist a complete Episode record.
- A `policy` is just a callable(snapshot) -> Decision. P1 ships a rule-based
  policy. P2 will add an LLM policy that calls tools via ReAct; the interface
  is identical so policies are swappable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from sqlmodel import select

from ..db import session_scope
from ..models import Episode
from .context import build_snapshot


@dataclass
class Decision:
    decision: str  # long | short | skip | watch
    confidence: float
    rationale: str
    entry_plan: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


class Policy(Protocol):
    def __call__(self, snapshot: dict[str, Any]) -> Decision: ...


def rule_policy(snapshot: dict[str, Any]) -> Decision:
    """Default rule-based policy for the OI+FR flip trigger.

    Goes long when: funding flipped negative + OI segments are monotonically
    rising + total OI change exceeds threshold + non-trivial 24h volume.
    """
    oi_chg = snapshot.get("oi_change_pct", 0.0)
    oi_rising = snapshot.get("oi_rising", False)
    curr_fr = snapshot.get("curr_funding", 0.0)
    vol = snapshot.get("volume_24h", 0.0)

    tags: list[str] = []
    if curr_fr < 0:
        tags.append("funding_negative")
    if oi_rising:
        tags.append("oi_rising")
    if snapshot.get("has_spot"):
        tags.append("has_spot")
    else:
        tags.append("perp_only")

    if curr_fr < 0 and oi_rising and oi_chg >= 8 and vol >= 1_000_000:
        confidence = min(0.9, 0.5 + oi_chg / 100)
        tags.append("squeeze_setup")
        return Decision(
            decision="long",
            confidence=confidence,
            rationale=(
                f"Funding flipped from {snapshot.get('prev_funding', 0):+.4%} to "
                f"{curr_fr:+.4%} while OI rose {oi_chg:.1f}% across 4 segments — "
                "shorts paying longs while new money still piling in: classic squeeze setup."
            ),
            entry_plan={
                "side": "long",
                "type": "market_or_limit",
                "size_pct": 0.02,  # 2% of bankroll suggested
                "stop_loss_pct": -3.0,
                "take_profit_pct": [5.0, 10.0],
                "timeframe": "1h-4h",
            },
            tags=tags,
        )

    return Decision(
        decision="watch" if curr_fr < 0 else "skip",
        confidence=0.2,
        rationale="Conditions not strong enough; logging for journal only.",
        tags=tags,
    )


def _is_duplicate(symbol: str, trigger: str, dedup_hours: int) -> bool:
    cutoff = datetime.utcnow() - timedelta(hours=dedup_hours)
    with session_scope() as s:
        q = select(Episode).where(
            Episode.symbol == symbol,
            Episode.trigger == trigger,
            Episode.created_at >= cutoff,
        )
        return s.exec(q).first() is not None


@dataclass
class HarnessAgent:
    policy: Policy = rule_policy
    dedup_hours: int = 24

    def _policy_id(self) -> str:
        name = self.policy.__class__.__name__ if not isinstance(self.policy, type(rule_policy)) else "rule_policy"
        if name == "function":
            return "rule"
        if name == "DualPolicy":
            return "dual"
        if name == "LLMPolicy":
            return "llm"
        return name

    def _model_profile(self) -> str:
        if self._policy_id() not in {"dual", "llm"}:
            return ""
        try:
            from ..llm_clients import resolve

            return resolve("decision").profile or resolve("decision").model
        except Exception:
            return ""

    def handle_signal(
        self,
        trigger: str,
        symbol: str,
        base_signal: dict[str, Any],
        venue: str = "binance_perp",
        strategy_id: str = "legacy",
        strategy_name: str = "",
        strategy_version: str = "",
        policy_id: str | None = None,
        model_profile: str | None = None,
        risk_profile: str = "paper_default",
    ) -> Episode | None:
        if _is_duplicate(symbol, trigger, self.dedup_hours):
            return None

        snapshot = build_snapshot(symbol, base_signal)
        decision = self.policy(snapshot)

        # If the policy exposes traces (tool calls, retrieved memory), persist them.
        tools_called = list(getattr(self.policy, "tools_called", None) or [])
        similar_episode_ids = list(getattr(self.policy, "similar_episode_ids", None) or [])

        episode = Episode(
            trigger=trigger,
            symbol=symbol,
            venue=venue,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            policy_id=policy_id or self._policy_id(),
            model_profile=model_profile if model_profile is not None else self._model_profile(),
            risk_profile=risk_profile,
            snapshot=snapshot,
            tools_called=tools_called,
            similar_episode_ids=similar_episode_ids,
            reasoning=decision.rationale,
            decision=decision.decision,
            confidence=decision.confidence,
            entry_plan=decision.entry_plan,
            rationale=decision.rationale,
            tags=decision.tags,
        )

        with session_scope() as s:
            s.add(episode)
            s.flush()
            s.refresh(episode)
            # Detach for use after session close
            ep_id = episode.id

        # Reload as a fresh detached object so callers can mutate without session
        with session_scope() as s:
            return s.get(Episode, ep_id)

    def handle_strategy_signal(
        self,
        strategy: Any,
        signal: Any,
        venue: str = "binance_perp",
        policy_id: str | None = None,
        risk_profile: str = "paper_default",
    ) -> Episode | None:
        return self.handle_signal(
            trigger=signal.trigger,
            symbol=signal.symbol,
            base_signal=signal.data,
            venue=venue,
            strategy_id=strategy.id,
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            policy_id=policy_id or getattr(strategy, "default_policy_id", None),
            risk_profile=risk_profile,
        )

    # --- Lifecycle helpers ---------------------------------------------------

    def mark_executed(self, episode_id: str, price: float, size: float, extra: dict | None = None) -> None:
        with session_scope() as s:
            ep = s.get(Episode, episode_id)
            if not ep:
                return
            ep.executed = True
            entry = {"price": price, "size": size, "ts": datetime.utcnow().isoformat()}
            if extra:
                entry.update(extra)
            ep.actual_entry = entry
            s.add(ep)

    def close_trade(
        self,
        episode_id: str,
        exit_price: float,
        reason: str = "manual",
    ) -> None:
        with session_scope() as s:
            ep = s.get(Episode, episode_id)
            if not ep or not ep.actual_entry:
                return
            entry = float(ep.actual_entry.get("price", 0))
            size = float(ep.actual_entry.get("size", 0))
            if entry <= 0:
                return
            side = ep.entry_plan.get("side", "long")
            mult = 1 if side == "long" else -1
            pnl_pct = (exit_price - entry) / entry * 100 * mult
            ep.actual_exit = {"price": exit_price, "reason": reason, "ts": datetime.utcnow().isoformat()}
            ep.closed_at = datetime.utcnow()
            ep.pnl_pct = pnl_pct
            ep.pnl_usd = (exit_price - entry) * size * mult
            ep.outcome_label = "win" if pnl_pct > 0.5 else "loss" if pnl_pct < -0.5 else "breakeven"
            s.add(ep)

    def annotate(self, episode_id: str, reflection: str = "", lessons: list[str] | None = None) -> None:
        with session_scope() as s:
            ep = s.get(Episode, episode_id)
            if not ep:
                return
            if reflection:
                ep.reflection = reflection
            if lessons:
                ep.lessons = list({*ep.lessons, *lessons})
            s.add(ep)
