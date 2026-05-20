#!/usr/bin/env python3
"""One-shot system status snapshot. Safe to run anytime.

Shows:
  * scheduler process status
  * testnet account & open positions w/ live PnL%
  * episodes opened/closed in last N hours
  * recent auto_execute / position_watch events from logs
"""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from sqlmodel import select

from cryptoscan.db import session_scope
from cryptoscan.exchange import testnet
from cryptoscan.models import Episode


def section(t: str) -> None:
    print(f"\n=== {t} ===")


def main() -> int:
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 2

    section("scheduler")
    try:
        out = subprocess.check_output(["pgrep", "-fl", "cryptoscan run"], text=True).strip()
        print(out or "  (not running)")
    except subprocess.CalledProcessError:
        print("  (not running)")

    section("testnet account")
    try:
        acct = testnet.account()
        print(f"  total: ${acct['total_usdt']:.2f}    "
              f"free: ${acct['free_usdt']:.2f}    used: ${acct['used_usdt']:.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        acct = {}

    section("open positions")
    try:
        live = testnet.all_open_positions()
    except Exception as e:
        print(f"  ERROR: {e}")
        live = []
    if not live:
        print("  (none)")
    else:
        print(f"  {'symbol':<14} {'side':<6} {'qty':>10} {'entry':>10} {'mark':>10} {'PnL%':>7} {'uPnL':>8}")
        for p in live:
            entry = float(p["entry_price"])
            mark = float(p["mark_price"])
            sign = 1 if p["side"] == "long" else -1
            pct = (mark - entry) / entry * 100 * sign if entry else 0.0
            print(f"  {p['symbol']:<14} {p['side']:<6} {p['contracts']:>10.2f} "
                  f"{entry:>10.6f} {mark:>10.6f} {pct:>+6.2f}% ${p['unrealized_pnl']:>+7.4f}")

    section(f"episodes (last {hours}h)")
    cut = datetime.utcnow() - timedelta(hours=hours)
    with session_scope() as s:
        rows = list(s.exec(
            select(Episode).where(Episode.created_at >= cut).order_by(Episode.created_at.desc())
        ).all())
    opened = [r for r in rows if r.executed]
    closed = [r for r in rows if r.closed_at is not None]
    decisions = {}
    for r in rows:
        decisions[r.decision] = decisions.get(r.decision, 0) + 1
    print(f"  total={len(rows)}  opened={len(opened)}  closed={len(closed)}  decisions={decisions}")
    if opened:
        print("  --- opened ---")
        for r in opened[:6]:
            tag = next((t for t in (r.tags or []) if t.startswith("dual:")), "?")
            entry_p = (r.actual_entry or {}).get("price")
            exit_p = (r.actual_exit or {}).get("price")
            pnl = f"{r.pnl_pct:+.2f}%" if r.pnl_pct is not None else "open"
            print(f"  {r.id[:8]}  {r.symbol:<12} {r.decision:<5} entry=${entry_p}  "
                  f"exit=${exit_p}  pnl={pnl}  ({tag})")

    section("recent scheduler events (auto_execute / position_watch / errors)")
    log = Path("logs/scheduler.log")
    if not log.exists():
        print("  (no log file)")
    else:
        # Tail last ~500 lines and filter to interesting events
        try:
            out = subprocess.check_output(["tail", "-n", "800", str(log)], text=True)
        except subprocess.CalledProcessError:
            out = ""
        keep = re.compile(r"(auto_execute: opened|auto_execute: skip|position_watch.*->|position_watch.*closing|scan_dual: \d+ episodes|WARNING|ERROR|warning|error)")
        lines = [l for l in out.splitlines() if keep.search(l)]
        for l in lines[-15:]:
            print(f"  {l}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
