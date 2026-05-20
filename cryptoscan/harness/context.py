"""Snapshot builder: assemble all relevant context at decision time."""
from __future__ import annotations

from typing import Any

from ..tools import binance_market as bm


def build_snapshot(symbol: str, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Enrich a base signal dict with extra market context.

    `base` should contain at minimum: symbol, price, price_chg_24h, volume_24h,
    prev_funding, curr_funding, oi_change_pct, oi_segments, oi_rising.
    """
    snap: dict[str, Any] = dict(base or {})
    coin = symbol.replace("USDT", "")

    # Market caps + spot availability are looked up lazily and cached per call site.
    try:
        mcaps = bm.market_caps()
    except Exception:
        mcaps = {}
    try:
        spot = bm.spot_symbols()
    except Exception:
        spot = set()

    snap["coin"] = coin
    snap["market_cap_usd"] = mcaps.get(coin, 0.0)
    snap["has_spot"] = coin in spot

    # Binance Square attention
    try:
        posts, views = bm.square_hashtag(coin)
    except Exception:
        posts, views = 0, 0
    snap["square_posts"] = posts
    snap["square_views"] = views

    return snap
