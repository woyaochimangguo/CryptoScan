from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel


def _uid() -> str:
    return uuid.uuid4().hex[:12]


class Episode(SQLModel, table=True):
    """One full trade decision lifecycle.

    Stages: trigger -> snapshot -> reasoning -> decision -> execution -> outcome -> reflection.
    Fields are nullable so we can persist progressively as info arrives.
    """

    id: str = Field(default_factory=_uid, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    # Trigger
    trigger: str = Field(index=True)  # e.g. "oi_funding_flip"
    symbol: str = Field(index=True)
    venue: str = "binance_perp"

    # Snapshot (full market context at decision time)
    snapshot: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Agent reasoning
    tools_called: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    similar_episode_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    reasoning: str = ""

    # Decision
    decision: str = "skip"  # long | short | skip | watch
    confidence: float = 0.0
    entry_plan: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    rationale: str = ""

    # Execution (manual at MVP stage)
    executed: bool = False
    actual_entry: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Outcome
    closed_at: Optional[datetime] = None
    actual_exit: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    pnl_pct: Optional[float] = None
    pnl_usd: Optional[float] = None
    outcome_label: Optional[str] = None  # win | loss | breakeven | missed

    # Reflection
    reflection: str = ""
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    lessons: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    # Notification trace
    notified: bool = False
    notify_message_id: Optional[str] = None


class RuntimeState(SQLModel, table=True):
    """Small durable cache for process-local runtime state.

    The dashboard reads these rows instead of calling exchange/scanner services
    directly, so the web process remains available when external APIs stall.
    """

    key: str = Field(primary_key=True)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    value: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
