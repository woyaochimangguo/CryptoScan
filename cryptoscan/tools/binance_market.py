"""Binance public market data helpers (no API key required)."""
from __future__ import annotations

import time
from threading import Lock
from typing import Any, Callable, TypeVar

import httpx

from .registry import tool

FAPI = "https://fapi.binance.com"
SAPI = "https://api.binance.com"
BAPI = "https://www.binance.com/bapi"

_HEADERS = {"User-Agent": "Mozilla/5.0 cryptoscan/0.1"}
_TIMEOUT = 10.0


def _get(url: str, params: dict | None = None, timeout: float = _TIMEOUT) -> Any:
    r = httpx.get(url, params=params, headers=_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Tiny TTL cache: avoids hitting bulk endpoints 30x per scan.
# ---------------------------------------------------------------------------
_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_LOCK = Lock()
T = TypeVar("T")


def _ttl_cache(key: str, ttl: float, producer: Callable[[], T]) -> T:
    now = time.time()
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    value = producer()
    with _CACHE_LOCK:
        _CACHE[key] = (now, value)
    return value


def cache_clear() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


@tool("binance.perp_symbols", "List active USDT-margined perpetual symbols on Binance Futures.")
def perp_symbols() -> list[str]:
    def _produce() -> list[str]:
        info = _get(f"{FAPI}/fapi/v1/exchangeInfo")
        return [
            s["symbol"]
            for s in info.get("symbols", [])
            if s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        ]
    return _ttl_cache("perp_symbols", 600, _produce)


def perp_symbol_details() -> dict[str, dict]:
    """Active USDT-margined perpetual exchangeInfo rows keyed by symbol."""
    def _produce() -> dict[str, dict]:
        info = _get(f"{FAPI}/fapi/v1/exchangeInfo")
        return {
            s["symbol"]: s
            for s in info.get("symbols", [])
            if s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        }
    return _ttl_cache("perp_symbol_details", 600, _produce)


@tool("binance.perp_tickers", "Get 24h ticker for all USDT perps as {symbol: ticker_dict}.")
def perp_tickers() -> dict[str, dict]:
    return _ttl_cache("perp_tickers", 30,
                      lambda: {t["symbol"]: t for t in _get(f"{FAPI}/fapi/v1/ticker/24hr")})


@tool("binance.perp_funding", "Get current funding rate (premiumIndex) for all symbols.")
def perp_funding() -> dict[str, float]:
    return _ttl_cache("perp_funding", 30,
                      lambda: {x["symbol"]: float(x["lastFundingRate"]) for x in _get(f"{FAPI}/fapi/v1/premiumIndex")})


@tool("binance.oi_history", "Hourly open interest history for a perp symbol.")
def oi_history(symbol: str, period: str = "1h", limit: int = 48) -> list[dict]:
    return _get(
        f"{FAPI}/futures/data/openInterestHist",
        {"symbol": symbol, "period": period, "limit": limit},
    )


@tool("binance.long_short_ratio", "Top trader long/short account ratio.")
def long_short_ratio(symbol: str, period: str = "1h", limit: int = 24) -> list[dict]:
    return _get(
        f"{FAPI}/futures/data/topLongShortAccountRatio",
        {"symbol": symbol, "period": period, "limit": limit},
    )


@tool("binance.spot_symbols", "Set of base assets that have a spot USDT pair listed.")
def spot_symbols() -> set[str]:
    def _produce() -> set[str]:
        info = _get(f"{SAPI}/api/v3/exchangeInfo")
        return {
            s["baseAsset"]
            for s in info.get("symbols", [])
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        }
    return _ttl_cache("spot_symbols", 600, _produce)


def spot_usdt_pairs() -> set[str]:
    """Exact Binance Spot symbols quoted in USDT, e.g. BTCUSDT."""
    def _produce() -> set[str]:
        info = _get(f"{SAPI}/api/v3/exchangeInfo")
        return {
            s["symbol"]
            for s in info.get("symbols", [])
            if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
        }
    return _ttl_cache("spot_usdt_pairs", 600, _produce)


@tool("binance.market_caps", "Approximate circulating market caps from Binance marketing endpoint.")
def market_caps() -> dict[str, float]:
    def _produce() -> dict[str, float]:
        try:
            data = _get(f"{BAPI}/composite/v1/public/marketing/symbol/list")
        except Exception:
            return {}
        out: dict[str, float] = {}
        for item in data.get("data", []) or []:
            name = item.get("name")
            mc = item.get("marketCap")
            if name and mc:
                try:
                    out[name] = float(mc)
                except (TypeError, ValueError):
                    continue
        return out
    return _ttl_cache("market_caps", 300, _produce)


@tool("binance.square_hashtag", "Binance Square hashtag stats: (post_count, view_count) for a coin.")
def square_hashtag(coin: str) -> tuple[int, int]:
    def _produce() -> tuple[int, int]:
        try:
            data = _get(
                f"{BAPI}/composite/v4/friendly/pgc/content/queryByHashtag",
                {"hashtag": f"#{coin.lower()}", "pageIndex": 1, "pageSize": 1, "orderBy": "HOT"},
            )
            ht = (data.get("data") or {}).get("hashtag") or {}
            return int(ht.get("contentCount", 0) or 0), int(ht.get("viewCount", 0) or 0)
        except Exception:
            return 0, 0
    return _ttl_cache(f"square_hashtag:{coin.lower()}", 300, _produce)


def prefetch_square_hashtags(coins: list[str], workers: int = 8) -> None:
    """Warm the square_hashtag cache for many coins concurrently.
    Cuts per-scan latency from N*~3s sequential to ~3-5s total."""
    from concurrent.futures import ThreadPoolExecutor
    coins = list(dict.fromkeys(coins))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(square_hashtag, coins))
