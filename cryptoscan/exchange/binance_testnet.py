"""Binance USDⓈ-M Futures Testnet wrapper.

We tried ccxt for the convenience layer but ccxt's `loadMarkets`/`fetch_balance`
auto-pull from mainnet sapi endpoints which our testnet keys can't authenticate.
After repeated workarounds, the most reliable path is direct HTTP + HMAC against
testnet.binancefuture.com — proven to respond in <3s for every endpoint we use.

We keep ccxt only for symbol/precision helpers (no network calls)."""
from __future__ import annotations

import hashlib
import hmac
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any

import ccxt
import httpx

from ..config import settings

BASE = "https://testnet.binancefuture.com"
_TIMEOUT = 10.0


_client: ccxt.binance | None = None


def get_client() -> ccxt.binance:
    """Return a singleton ccxt client pointing at Binance Futures testnet."""
    global _client
    if _client is not None:
        return _client
    if not settings.binance_testnet_key or not settings.binance_testnet_secret:
        raise RuntimeError(
            "BINANCE_TESTNET_KEY / BINANCE_TESTNET_SECRET not set. "
            "Get them at https://testnet.binancefuture.com -> API Key tab."
        )
    c = ccxt.binance({
        "apiKey": settings.binance_testnet_key,
        "secret": settings.binance_testnet_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
            # Limit market discovery to USDT-M linear perpetuals so loadMarkets
            # never touches spot/inverse/options (which on testnet point at
            # mainnet sapi endpoints we don't have permission for).
            "fetchMarkets": ["linear"],
        },
    })
    # Disable sapi-based currency fetch; testnet keys aren't valid on mainnet sapi.
    c.has["fetchCurrencies"] = False
    # ccxt's set_sandbox_mode no longer works for binance futures; rewrite URLs manually
    # to point at testnet.binancefuture.com (REST) for both public and private endpoints.
    base = "https://testnet.binancefuture.com"
    c.urls["api"] = {
        **(c.urls.get("api") or {}),
        "fapiPublic": f"{base}/fapi/v1",
        "fapiPublicV2": f"{base}/fapi/v2",
        "fapiPublicV3": f"{base}/fapi/v3",
        "fapiPrivate": f"{base}/fapi/v1",
        "fapiPrivateV2": f"{base}/fapi/v2",
        "fapiPrivateV3": f"{base}/fapi/v3",
        "fapiData": f"{base}/futures/data",
    }
    _client = c
    return c


# ---------------------------------------------------------------------------
# Direct HTTP/HMAC layer (bypass ccxt's loadMarkets/sapi pitfalls)
# ---------------------------------------------------------------------------

def _sign(params: dict[str, Any]) -> str:
    qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    sig = hmac.new(settings.binance_testnet_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return qs + "&signature=" + sig


def _signed_request(method: str, path: str, params: dict[str, Any] | None = None) -> Any:
    p = dict(params or {})
    p["timestamp"] = int(time.time() * 1000)
    p.setdefault("recvWindow", 5000)
    qs = _sign(p)
    url = f"{BASE}{path}?{qs}"
    headers = {"X-MBX-APIKEY": settings.binance_testnet_key}
    r = httpx.request(method, url, headers=headers, timeout=_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"testnet {method} {path} -> {r.status_code}: {r.text[:300]}")
    return r.json()


def _public_get(path: str, params: dict[str, Any] | None = None) -> Any:
    r = httpx.get(f"{BASE}{path}", params=params, timeout=_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"testnet GET {path} -> {r.status_code}: {r.text[:300]}")
    return r.json()


# ---------------------------------------------------------------------------
# Symbol/precision metadata (loaded once, cached)
# ---------------------------------------------------------------------------

_EXCHANGE_INFO: dict[str, dict[str, Any]] | None = None


def _exchange_info() -> dict[str, dict[str, Any]]:
    """Map of {symbol: {pricePrecision, quantityPrecision, lotSize, ...}} from
    testnet exchangeInfo. Cached for the process lifetime."""
    global _EXCHANGE_INFO
    if _EXCHANGE_INFO is not None:
        return _EXCHANGE_INFO
    info = _public_get("/fapi/v1/exchangeInfo")
    out: dict[str, dict[str, Any]] = {}
    for s in info.get("symbols", []):
        if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT":
            continue
        filters = {f["filterType"]: f for f in s.get("filters", [])}
        lot = filters.get("LOT_SIZE", {})
        prc = filters.get("PRICE_FILTER", {})
        out[s["symbol"]] = {
            "qty_precision": int(s.get("quantityPrecision", 0)),
            "price_precision": int(s.get("pricePrecision", 0)),
            "step_size": float(lot.get("stepSize", 0) or 0),
            "min_qty": float(lot.get("minQty", 0) or 0),
            "tick_size": float(prc.get("tickSize", 0) or 0),
        }
    _EXCHANGE_INFO = out
    return out


def _round_qty(symbol: str, qty: float) -> float:
    meta = _exchange_info().get(symbol)
    if not meta:
        return float(f"{qty:.6f}")
    step = meta["step_size"] or (10 ** -meta["qty_precision"])
    if step <= 0:
        return float(f"{qty:.{meta['qty_precision']}f}")
    rounded = (qty // step) * step  # truncate down to step grid
    return float(f"{rounded:.{meta['qty_precision']}f}")


# ---------------------------------------------------------------------------
# Public helpers (sync)
# ---------------------------------------------------------------------------

def account() -> dict[str, Any]:
    rows = _signed_request("GET", "/fapi/v3/balance")
    for row in rows:
        if row.get("asset") == "USDT":
            return {
                "total_usdt": float(row.get("balance") or 0),
                "free_usdt": float(row.get("availableBalance") or 0),
                "used_usdt": float(row.get("balance") or 0) - float(row.get("availableBalance") or 0),
            }
    return {"total_usdt": 0.0, "free_usdt": 0.0, "used_usdt": 0.0}


def set_leverage(symbol: str, leverage: int) -> None:
    """Set isolated leverage. Idempotent. Errors are silently dropped (set on
    open positions etc.) so order placement isn't blocked."""
    try:
        _signed_request("POST", "/fapi/v1/leverage",
                        {"symbol": symbol, "leverage": int(leverage)})
    except Exception:
        pass


@dataclass
class OrderResult:
    id: str
    symbol: str
    side: str       # 'long' | 'short'
    qty: float
    avg_price: float
    notional_usdt: float
    raw: dict[str, Any]


def _mark_price(symbol: str) -> float:
    data = _public_get("/fapi/v1/premiumIndex", {"symbol": symbol})
    return float(data.get("markPrice") or 0)


def open_position(symbol: str, side: str, size_usdt: float, leverage: int | None = None) -> OrderResult:
    if side not in {"long", "short"}:
        raise ValueError(f"side must be long/short, got {side!r}")
    lev = leverage or settings.paper_leverage
    set_leverage(symbol, lev)

    px = _mark_price(symbol)
    if px <= 0:
        raise RuntimeError(f"could not fetch mark price for {symbol}")
    qty = _round_qty(symbol, size_usdt / px)
    if qty <= 0:
        raise RuntimeError(f"computed qty <= 0 (size_usdt={size_usdt}, mark={px})")

    api_side = "BUY" if side == "long" else "SELL"
    resp = _signed_request("POST", "/fapi/v1/order", {
        "symbol": symbol,
        "side": api_side,
        "type": "MARKET",
        "quantity": qty,
        "newOrderRespType": "RESULT",
    })
    avg = float(resp.get("avgPrice") or 0) or float(resp.get("price") or 0) or px
    return OrderResult(
        id=str(resp.get("orderId")),
        symbol=symbol,
        side=side,
        qty=qty,
        avg_price=avg,
        notional_usdt=qty * avg,
        raw=resp,
    )


def close_position(symbol: str, side: str) -> OrderResult:
    """Reduce-only market close. `side` = original entry side."""
    pos = current_position(symbol)
    if not pos or abs(pos["contracts"]) < 1e-12:
        raise RuntimeError(f"no open position on {symbol}")
    qty = abs(float(pos["contracts"]))

    api_side = "SELL" if side == "long" else "BUY"
    resp = _signed_request("POST", "/fapi/v1/order", {
        "symbol": symbol,
        "side": api_side,
        "type": "MARKET",
        "quantity": qty,
        "reduceOnly": "true",
        "newOrderRespType": "RESULT",
    })
    avg = float(resp.get("avgPrice") or 0) or float(resp.get("price") or 0) or 0
    return OrderResult(
        id=str(resp.get("orderId")),
        symbol=symbol,
        side=side,
        qty=qty,
        avg_price=avg,
        notional_usdt=qty * avg,
        raw=resp,
    )


def _raw_positions() -> list[dict[str, Any]]:
    return _signed_request("GET", "/fapi/v3/positionRisk")


def current_position(symbol: str) -> dict[str, Any] | None:
    for p in _raw_positions():
        if p.get("symbol") != symbol:
            continue
        amt = float(p.get("positionAmt") or 0)
        if abs(amt) <= 0:
            continue
        return {
            "symbol": symbol,
            "side": "long" if amt > 0 else "short",
            "contracts": amt,
            "entry_price": float(p.get("entryPrice") or 0),
            "mark_price": float(p.get("markPrice") or 0),
            "unrealized_pnl": float(p.get("unRealizedProfit") or 0),
            "leverage": float(p.get("leverage") or 0),
            "raw": p,
        }
    return None


def all_open_positions() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in _raw_positions():
        amt = float(p.get("positionAmt") or 0)
        if abs(amt) <= 0:
            continue
        out.append({
            "symbol": p.get("symbol"),
            "side": "long" if amt > 0 else "short",
            "contracts": amt,
            "entry_price": float(p.get("entryPrice") or 0),
            "mark_price": float(p.get("markPrice") or 0),
            "unrealized_pnl": float(p.get("unRealizedProfit") or 0),
        })
    return out
