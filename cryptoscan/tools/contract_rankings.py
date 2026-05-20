"""Build Binance USDT perpetual rankings enriched with spot market cap data."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from . import binance_market as bm


_MULTIPLIER_PREFIXES = ("1000000", "100000", "10000", "1000")


def _underlying_asset(asset: str, known_assets: set[str]) -> str:
    for prefix in _MULTIPLIER_PREFIXES:
        if asset.startswith(prefix):
            stripped = asset[len(prefix):]
            if stripped in known_assets:
                return stripped
    return asset


def build_contract_rankings(include_unknown: bool = True) -> dict[str, Any]:
    """Return all Binance USDT perpetual contracts enriched with market caps.

    Binance Futures does not expose circulating market cap directly. The market
    cap values here come from Binance's public marketing endpoint and are best
    treated as approximate spot/circulating market caps for the base asset.
    """
    details = bm.perp_symbol_details()
    symbols = list(details)
    tickers = bm.perp_tickers()
    funding = bm.perp_funding()
    mcaps = bm.market_caps()
    spot_pairs = bm.spot_usdt_pairs()
    known_assets = set(mcaps) | {p.removesuffix("USDT") for p in spot_pairs}

    ranked_caps = sorted(
        ((coin, float(cap)) for coin, cap in mcaps.items() if float(cap or 0) > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    cap_rank = {coin: idx for idx, (coin, _cap) in enumerate(ranked_caps, start=1)}

    rows: list[dict[str, Any]] = []
    unknown_count = 0
    exact_spot_count = 0
    underlying_spot_count = 0
    perp_only_count = 0
    for symbol in symbols:
        info = details.get(symbol) or {}
        coin = str(info.get("baseAsset") or symbol.removesuffix("USDT"))
        market_cap_asset = _underlying_asset(coin, known_assets)
        ticker = tickers.get(symbol, {})
        market_cap = float(mcaps.get(coin) or mcaps.get(market_cap_asset) or 0)
        if market_cap <= 0:
            unknown_count += 1
            if not include_unknown:
                continue
        exact_spot_symbol = f"{coin}USDT"
        underlying_spot_symbol = f"{market_cap_asset}USDT"
        has_exact_spot = symbol == exact_spot_symbol and exact_spot_symbol in spot_pairs
        has_underlying_spot = (
            not has_exact_spot
            and market_cap_asset != coin
            and underlying_spot_symbol in spot_pairs
        )
        if has_exact_spot:
            listing_type = "spot_perp"
            listing_label = "spot+perp"
            exact_spot_count += 1
        elif has_underlying_spot:
            listing_type = "underlying_spot"
            listing_label = f"underlying spot ({market_cap_asset})"
            underlying_spot_count += 1
        else:
            listing_type = "perp_only"
            listing_label = "perp only"
            perp_only_count += 1
        rows.append(
            {
                "symbol": symbol,
                "coin": coin,
                "market_cap_asset": market_cap_asset,
                "market_cap_usd": market_cap,
                "market_cap_rank": cap_rank.get(coin) or cap_rank.get(market_cap_asset),
                "price": float(ticker.get("lastPrice", 0) or 0),
                "price_change_24h_pct": float(ticker.get("priceChangePercent", 0) or 0),
                "volume_24h_usdt": float(ticker.get("quoteVolume", 0) or 0),
                "funding_rate": float(funding.get(symbol, 0) or 0),
                "has_spot": has_exact_spot or has_underlying_spot,
                "has_exact_spot": has_exact_spot,
                "has_underlying_spot": has_underlying_spot,
                "spot_symbol": exact_spot_symbol if has_exact_spot else (
                    underlying_spot_symbol if has_underlying_spot else None
                ),
                "listing_type": listing_type,
                "listing_label": listing_label,
            }
        )

    rows.sort(key=lambda r: float(r["market_cap_usd"] or 0), reverse=True)
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "source": "binance_futures + binance_marketing_marketCap",
        "market_cap_note": "Approximate spot/circulating market cap from Binance marketing endpoint.",
        "total": len(rows),
        "unknown_market_cap": unknown_count,
        "exact_spot_count": exact_spot_count,
        "underlying_spot_count": underlying_spot_count,
        "perp_only_count": perp_only_count,
        "rows": rows,
    }
