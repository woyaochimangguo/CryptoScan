"""Build Binance USDT perpetual rankings enriched with spot market cap data."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from . import binance_market as bm


def build_contract_rankings(include_unknown: bool = True) -> dict[str, Any]:
    """Return all Binance USDT perpetual contracts enriched with market caps.

    Binance Futures does not expose circulating market cap directly. The market
    cap values here come from Binance's public marketing endpoint and are best
    treated as approximate spot/circulating market caps for the base asset.
    """
    symbols = bm.perp_symbols()
    tickers = bm.perp_tickers()
    funding = bm.perp_funding()
    mcaps = bm.market_caps()
    spot = bm.spot_symbols()

    ranked_caps = sorted(
        ((coin, float(cap)) for coin, cap in mcaps.items() if float(cap or 0) > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    cap_rank = {coin: idx for idx, (coin, _cap) in enumerate(ranked_caps, start=1)}

    rows: list[dict[str, Any]] = []
    unknown_count = 0
    for symbol in symbols:
        coin = symbol.removesuffix("USDT")
        ticker = tickers.get(symbol, {})
        market_cap = float(mcaps.get(coin, 0) or 0)
        if market_cap <= 0:
            unknown_count += 1
            if not include_unknown:
                continue
        rows.append(
            {
                "symbol": symbol,
                "coin": coin,
                "market_cap_usd": market_cap,
                "market_cap_rank": cap_rank.get(coin),
                "price": float(ticker.get("lastPrice", 0) or 0),
                "price_change_24h_pct": float(ticker.get("priceChangePercent", 0) or 0),
                "volume_24h_usdt": float(ticker.get("quoteVolume", 0) or 0),
                "funding_rate": float(funding.get(symbol, 0) or 0),
                "has_spot": coin in spot,
            }
        )

    rows.sort(key=lambda r: float(r["market_cap_usd"] or 0), reverse=True)
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "source": "binance_futures + binance_marketing_marketCap",
        "market_cap_note": "Approximate spot/circulating market cap from Binance marketing endpoint.",
        "total": len(rows),
        "unknown_market_cap": unknown_count,
        "rows": rows,
    }
