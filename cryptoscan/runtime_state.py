from __future__ import annotations

from datetime import datetime
from typing import Any

from .db import session_scope
from .models import RuntimeState

KEY_ACCOUNT = "exchange.account"
KEY_POSITIONS = "exchange.positions"
KEY_SCHEDULER = "scheduler.status"
KEY_CONTRACT_RANKINGS = "market.contract_rankings"


def set_state(key: str, value: dict[str, Any]) -> RuntimeState:
    with session_scope() as s:
        row = s.get(RuntimeState, key)
        if row is None:
            row = RuntimeState(key=key)
        row.updated_at = datetime.utcnow()
        row.value = value
        s.add(row)
        s.flush()
        s.refresh(row)
        return row


def get_state(key: str) -> dict[str, Any] | None:
    with session_scope() as s:
        row = s.get(RuntimeState, key)
        if row is None:
            return None
        return {
            "key": row.key,
            "updated_at": row.updated_at.isoformat(),
            "value": row.value or {},
        }


def mark_scheduler(status: str, **extra: Any) -> None:
    current = get_state(KEY_SCHEDULER)
    value = dict((current or {}).get("value") or {})
    value.update(extra)
    value["status"] = status
    set_state(KEY_SCHEDULER, value)
