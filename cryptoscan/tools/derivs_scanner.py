"""OI surge + funding-rate flip scanner.

Ported & refactored from the connectfarm1 single-file script. Detects symbols
where funding rate just turned negative AND open interest is rising, which is
a classic short-squeeze precursor.

State (last funding snapshot) is kept in a small JSON file under the data dir.
Heavy OI-history calls are only made for the few candidates that just flipped.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from ..config import settings
from . import binance_market as bm
from .registry import tool

FR_SNAPSHOT_FILE = settings.data_dir / "fr_snapshot.json"


@dataclass
class DerivSignal:
    symbol: str
    price: float
    price_chg_24h: float
    volume_24h: float
    prev_funding: float
    curr_funding: float
    oi_change_pct: float
    oi_segments: list[float]
    oi_rising: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_snapshot(path: Path) -> dict[str, float]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_snapshot(path: Path, snap: dict[str, float]) -> None:
    path.write_text(json.dumps(snap))


def _segment_oi(oi_hist: list[dict]) -> tuple[float, list[float], bool]:
    if len(oi_hist) < 12:
        return 0.0, [], False
    values = [float(x["sumOpenInterestValue"]) for x in oi_hist]
    seg_len = len(values) // 4
    if seg_len < 3:
        return 0.0, [], False
    segs = [
        sum(values[:seg_len]) / seg_len,
        sum(values[seg_len : seg_len * 2]) / seg_len,
        sum(values[seg_len * 2 : seg_len * 3]) / seg_len,
        sum(values[seg_len * 3 :]) / max(1, len(values[seg_len * 3 :])),
    ]
    chg = (segs[3] - segs[0]) / segs[0] * 100 if segs[0] > 0 else 0.0
    rising = chg > 0 and segs[3] > segs[2] and segs[2] > segs[1]
    return chg, segs, rising


@tool(
    "derivs.scan_oi_funding_flip",
    "Scan all USDT perps and return symbols whose funding rate just flipped "
    "from >=0 to <0 with open interest still rising (short-squeeze precursor).",
)
def scan_oi_funding_flip(min_volume_usdt: float | None = None) -> list[dict[str, Any]]:
    min_vol = settings.min_volume_usdt if min_volume_usdt is None else min_volume_usdt

    symbols = bm.perp_symbols()
    tickers = bm.perp_tickers()
    funding_now = bm.perp_funding()

    active = [
        s for s in symbols
        if float(tickers.get(s, {}).get("quoteVolume", 0) or 0) >= min_vol
        and s in funding_now
    ]

    prev = _load_snapshot(FR_SNAPSHOT_FILE)
    _save_snapshot(FR_SNAPSHOT_FILE, funding_now)

    if not prev:
        return []  # first run: just persist baseline

    flipped = [s for s in active if prev.get(s, 0.0) >= 0 and funding_now[s] < 0]

    def _fetch_one(sym: str) -> DerivSignal | None:
        try:
            hist = bm.oi_history(sym, period="1h", limit=48)
        except Exception:
            return None
        oi_chg, segs, rising = _segment_oi(hist)
        t = tickers.get(sym, {})
        return DerivSignal(
            symbol=sym,
            price=float(t.get("lastPrice", 0) or 0),
            price_chg_24h=float(t.get("priceChangePercent", 0) or 0),
            volume_24h=float(t.get("quoteVolume", 0) or 0),
            prev_funding=float(prev.get(sym, 0.0)),
            curr_funding=float(funding_now[sym]),
            oi_change_pct=oi_chg,
            oi_segments=segs,
            oi_rising=rising,
        )

    signals: list[DerivSignal] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fut in as_completed([ex.submit(_fetch_one, s) for s in flipped]):
            sig = fut.result()
            if sig is not None:
                signals.append(sig)

    return [s.to_dict() for s in signals]
