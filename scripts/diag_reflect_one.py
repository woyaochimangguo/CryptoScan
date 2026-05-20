#!/usr/bin/env python3
"""Pick the most recent closed episode without a reflection and run the
auto-reflect LLM pipeline on it end-to-end. Prints the result."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta

from sqlmodel import select

from cryptoscan.db import session_scope
from cryptoscan.harness.memory import auto_reflect_episode
from cryptoscan.models import Episode


def main() -> int:
    cut = datetime.utcnow() - timedelta(days=30)
    with session_scope() as s:
        row = s.exec(
            select(Episode)
            .where(Episode.closed_at.is_not(None))
            .where(Episode.closed_at >= cut)
            .where((Episode.reflection == "") | (Episode.reflection.is_(None)))
            .order_by(Episode.closed_at.desc())
        ).first()
        if not row:
            print("no candidate (all closed episodes already reflected)")
            return 1
        ep_id = row.id
        sym = row.symbol
        pnl = row.pnl_pct
        outcome = row.outcome_label

    print(f"target: {ep_id[:8]}  {sym}  pnl={pnl:+.2f}%  outcome={outcome}")
    ok = auto_reflect_episode(ep_id)
    print(f"auto_reflect_episode -> {ok}")

    with session_scope() as s:
        r = s.get(Episode, ep_id)
        print("\n--- reflection ---")
        print(r.reflection or "(empty)")
        print("\n--- lessons ---")
        for l in r.lessons or []:
            print(f"  - {l}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
