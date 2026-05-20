#!/usr/bin/env python3
"""Prove that LLMPolicy now injects past episodes into the prompt.

Constructs a synthetic snapshot for a symbol that HAS historical closed
trades in DB (e.g. DOGEUSDT/REDUSDT), runs LLMPolicy once, then prints:
  * how many past episodes were retrieved
  * the MEMORY block that was actually fed into the prompt
  * the new decision + rationale (so you can see if it references prior losses)
"""
from __future__ import annotations

import sys

from cryptoscan.harness.context import build_snapshot
from cryptoscan.harness.llm_policy import LLMPolicy
from cryptoscan.harness.memory import retrieve_similar_episodes, summarize_for_prompt
from cryptoscan.tools import binance_market as bm


def main() -> int:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AXSUSDT"

    tickers = bm.perp_tickers()
    t = tickers.get(symbol, {})
    if not t:
        print(f"{symbol} not tradable")
        return 2

    # Construct snapshot similar to the past losing AXS short (negative OI, flipping funding, declining price)
    base = {
        "symbol": symbol,
        "price": float(t.get("lastPrice") or 0),
        "price_chg_24h": float(t.get("priceChangePercent") or 0),
        "volume_24h": float(t.get("quoteVolume") or 0),
        "prev_funding": 0.0001,
        "curr_funding": -0.0005,
        "oi_change_pct": -8.0,
        "oi_segments": [100, 98, 96, 92],
        "oi_rising": False,
    }
    snap = build_snapshot(symbol, base)

    past = retrieve_similar_episodes(symbol, snap, limit=5)
    print(f"=== retrieval for {symbol} ===")
    print(f"past closed episodes retrieved: {len(past)}")
    for p in past:
        print(f"  {p.id[:8]}  {p.symbol}  pnl={p.pnl_pct:+.2f}%  "
              f"tags={[t for t in (p.tags or []) if t.startswith('dual:')]}  "
              f"reflected={'yes' if (p.reflection or '').strip() else 'no'}")

    block = summarize_for_prompt(past)
    print("\n=== MEMORY block that will be sent to LLM ===")
    print(block)

    print("\n=== running LLMPolicy with memory enabled ===")
    p = LLMPolicy(use_memory=True, verbose=False)
    dec = p(snap)
    print(f"decision: {dec.decision}  conf={dec.confidence:.2f}")
    print(f"similar_episode_ids: {p.similar_episode_ids}")
    print(f"rationale (first 400 chars):\n{(dec.rationale or '')[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
