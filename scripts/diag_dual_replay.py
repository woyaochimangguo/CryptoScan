#!/usr/bin/env python3
"""Replay DualPolicy on existing episode snapshots without writing to DB.
Verifies the new escalation logic: rule=watch + interesting -> LLM call.

Usage: scripts/diag_dual_replay.py [n=5]
Picks the N most recent episodes whose snapshot has |oi_change_pct|>=3, calls
DualPolicy on each snapshot fresh, prints rule/llm/final decisions side by side.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta

from sqlmodel import select

from cryptoscan.db import session_scope
from cryptoscan.models import Episode
from cryptoscan.harness.dual_policy import DualPolicy
from cryptoscan.harness.llm_policy import LLMPolicy
from cryptoscan.harness.agent import rule_policy


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cut = datetime.utcnow() - timedelta(hours=24)
    with session_scope() as s:
        rows = list(s.exec(
            select(Episode).where(Episode.created_at >= cut).order_by(Episode.created_at.desc())
        ).all())

    candidates = [
        r for r in rows
        if abs((r.snapshot or {}).get("oi_change_pct", 0)) >= 3.0
        and (r.snapshot or {}).get("curr_funding", 0) < 0
        and (r.snapshot or {}).get("volume_24h", 0) >= 1_000_000
    ][:n]

    if not candidates:
        print("no episodes meet escalation criteria (|oi|>=3 & fund<0 & vol>=1M)")
        # Fall back to top-3 by |oi_chg|
        candidates = sorted(rows, key=lambda r: -abs((r.snapshot or {}).get("oi_change_pct", 0)))[:3]
        print(f"falling back to top-{len(candidates)} by |oi_chg|")

    dp = DualPolicy(llm_policy=LLMPolicy())

    for r in candidates:
        sn = r.snapshot or {}
        print("=" * 78)
        print(f"{r.symbol}  oi_chg={sn.get('oi_change_pct',0):+.1f}%  "
              f"funding={sn.get('curr_funding',0):+.4%}  vol=${sn.get('volume_24h',0)/1e6:.1f}M  "
              f"rising={sn.get('oi_rising')}")

        rule = rule_policy(sn)
        print(f"  rule:    {rule.decision:6s}  conf={rule.confidence:.2f}")

        out = dp(sn)
        print(f"  dual:    {out.decision:6s}  conf={out.confidence:.2f}")
        for t in out.tags:
            if t.startswith("dual:"):
                print(f"    tag:   {t}")
        print(f"  rationale (first 200 chars): {out.rationale[:200]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
