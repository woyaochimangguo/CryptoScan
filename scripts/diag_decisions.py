#!/usr/bin/env python3
"""Show distribution of decisions / OI / funding for episodes created in the
last N minutes. Replaces a fragile `python -c` one-liner.

Usage: scripts/diag_decisions.py [minutes=5]
"""
from __future__ import annotations

import statistics
import sys
from collections import Counter
from datetime import datetime, timedelta

from sqlmodel import select

from cryptoscan.db import session_scope
from cryptoscan.models import Episode


def main() -> int:
    minutes = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)

    with session_scope() as s:
        rows = list(s.exec(select(Episode).where(Episode.created_at >= cutoff)).all())

    print(f"# episodes in last {minutes}min: {len(rows)}")
    if not rows:
        return 0

    # decisions
    print("decisions:", Counter(r.decision for r in rows).most_common())

    # tag prefixes (first colon-segment)
    tagc: Counter[str] = Counter()
    for r in rows:
        for t in (r.tags or []):
            tagc[t.split(":", 1)[0]] += 1
    print("tag prefixes:", tagc.most_common(15))

    agree = [r for r in rows if any((t or "").startswith("dual:agree") for t in (r.tags or []))]
    print(f"dual:agree count: {len(agree)}")

    # OI / funding distribution
    ois = [(r.snapshot or {}).get("oi_change_pct", 0.0) for r in rows]
    risings = sum(1 for r in rows if (r.snapshot or {}).get("oi_rising"))
    funds = [(r.snapshot or {}).get("curr_funding", 0.0) for r in rows]
    print(f"oi_chg: min={min(ois):+.1f}  max={max(ois):+.1f}  median={statistics.median(ois):+.1f}  rising={risings}/{len(rows)}")
    print(f"funding<0: {sum(1 for f in funds if f<0)}/{len(rows)}")

    print("\ntop-8 by oi_change_pct:")
    print(f"  {'symbol':<14} {'oi_chg':>8} {'rising':>7} {'funding':>10} {'vol_M':>8}  decision")
    top = sorted(rows, key=lambda r: -((r.snapshot or {}).get("oi_change_pct", 0)))[:8]
    for r in top:
        sn = r.snapshot or {}
        oi = sn.get("oi_change_pct", 0)
        rising = sn.get("oi_rising")
        fr = sn.get("curr_funding", 0)
        vol_m = sn.get("volume_24h", 0) / 1e6
        print(f"  {r.symbol:<14} {oi:+7.1f}% {str(rising):>7} {fr:+9.4%} {vol_m:>7.1f}M  {r.decision}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
