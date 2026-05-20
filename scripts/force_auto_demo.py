#!/usr/bin/env python3
"""Forced end-to-end auto-execute demo.

Builds a synthetic-but-strong long snapshot for a real liquid symbol, runs the
real DualPolicy (rule + LLM) on it, and writes an Episode tagged for
auto-execute. A running `cryptoscan run` scheduler will then pick it up within
~30s and open a $20 paper position on testnet.

Usage:
  scripts/force_auto_demo.py [SYMBOL=DOGEUSDT] [--side long|short]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from cryptoscan.db import init_db, session_scope
from cryptoscan.harness.agent import rule_policy
from cryptoscan.harness.context import build_snapshot
from cryptoscan.harness.dual_policy import DualPolicy
from cryptoscan.harness.llm_policy import LLMPolicy
from cryptoscan.models import Episode
from cryptoscan.tools import binance_market as bm


def synth_snapshot(symbol: str, side: str) -> dict:
    """Synthesize a snapshot strong enough that rule_policy fires.

    For long: funding flipped negative + monotonic OI rise of 15% + good volume.
    For short: funding flipped positive + OI flush -15% (rule won't fire short
    today, LLM may; that's the dual:llm_lead path).
    """
    tickers = bm.perp_tickers()
    t = tickers.get(symbol, {})
    price = float(t.get("lastPrice") or 0)
    vol = float(t.get("quoteVolume") or 0)
    if price <= 0:
        raise SystemExit(f"{symbol} not tradable on Binance perp")

    if side == "long":
        base = {
            "symbol": symbol,
            "price": price,
            "price_chg_24h": float(t.get("priceChangePercent") or 0),
            "volume_24h": max(vol, 5_000_000),
            "prev_funding": 0.0001,         # was slightly positive
            "curr_funding": -0.0005,        # flipped negative — squeeze fuel
            "oi_change_pct": 15.0,          # strong rise
            "oi_segments": [100.0, 105.0, 110.0, 115.0],
            "oi_rising": True,              # monotonic
        }
    else:
        base = {
            "symbol": symbol,
            "price": price,
            "price_chg_24h": float(t.get("priceChangePercent") or 0),
            "volume_24h": max(vol, 5_000_000),
            "prev_funding": -0.0001,
            "curr_funding": 0.0008,         # crowded longs paying up
            "oi_change_pct": -12.0,         # OI bleeding
            "oi_segments": [115.0, 110.0, 105.0, 100.0],
            "oi_rising": False,
        }
    return build_snapshot(symbol, base)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default="DOGEUSDT")
    ap.add_argument("--side", choices=["long", "short"], default="long")
    ap.add_argument("--bypass-llm", action="store_true",
                    help="Skip LLM check; force-write an episode with dual:llm_lead "
                         "tag so auto_execute will pick it up. For pipeline test only.")
    args = ap.parse_args()

    init_db()
    print(f"=== forced demo  symbol={args.symbol}  side={args.side} ===")

    snap = synth_snapshot(args.symbol, args.side)
    print(f"price=${snap['price']}  vol=${snap['volume_24h']/1e6:.1f}M  "
          f"oi_chg={snap['oi_change_pct']:+.1f}%  funding={snap['curr_funding']:+.4%}")

    # Step 1: rule alone
    rule_dec = rule_policy(snap)
    print(f"\n[rule]  decision={rule_dec.decision}  conf={rule_dec.confidence:.2f}")

    if args.bypass_llm:
        print("[bypass-llm] skipping LLM, fabricating dual:llm_lead tag")
        from cryptoscan.harness.agent import Decision
        out = Decision(
            decision=args.side,
            confidence=0.6,
            rationale=f"[FORCED-DEMO] pipeline test on {args.symbol} {args.side}; "
                      "bypassed LLM consensus to validate auto_execute → testnet → "
                      "position_watch chain.",
            entry_plan={
                "side": args.side,
                "type": "market_or_limit",
                "size_pct": 0.01,
                "stop_loss_pct": 1.5,
                "take_profit_pct": [2.5, 5.0],
                "timeframe": "demo",
            },
            tags=list(rule_dec.tags) + [f"dual:llm_lead:{args.side}", "forced_demo"],
        )
        tools_called = []
    else:
        # Step 2: dual policy (will escalate to LLM)
        print("[dual]  calling LLM (this can take 10-30s)...")
        t0 = time.time()
        dp = DualPolicy(llm_policy=LLMPolicy())
        out = dp(snap)
        tools_called = list(getattr(dp, "tools_called", None) or [])
        print(f"[dual]  decision={out.decision}  conf={out.confidence:.2f}  "
              f"({time.time()-t0:.1f}s)")
        print(f"        tags: {[t for t in out.tags if t.startswith('dual:')]}")
        print(f"        plan: {out.entry_plan}")

        if out.decision not in {"long", "short"}:
            print("\nDual policy refused to act on synthetic signal. "
                  "LLM judged the setup not actionable — no episode created.")
            print("Hint: re-run with --bypass-llm to test the auto_execute pipeline anyway.")
            return 0

    # Step 3: persist as fresh episode (different trigger so dedup doesn't block)
    ep = Episode(
        trigger="manual_demo",
        symbol=args.symbol,
        venue="binance_perp",
        snapshot=snap,
        tools_called=tools_called,
        reasoning=out.rationale,
        decision=out.decision,
        confidence=out.confidence,
        entry_plan=out.entry_plan,
        rationale=out.rationale,
        tags=list(out.tags),
        created_at=datetime.utcnow(),
    )
    with session_scope() as s:
        s.add(ep)
        s.flush()
        ep_id = ep.id
    print(f"\n[db] persisted episode {ep_id}")

    print("\nNow watch the scheduler:")
    print("  tail -f logs/scheduler.log | grep -E 'auto_execute|position_watch'")
    print(f"\nOr inspect via:")
    print(f"  .venv/bin/python scripts/show_episode.py {ep_id[:12]}")
    print(f"  http://127.0.0.1:8766/  (find row {args.symbol} trigger=manual_demo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
