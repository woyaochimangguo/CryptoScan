"""FastAPI dashboard.

Endpoints:
  GET  /                    -> HTML dashboard (positions + episodes)
  GET  /api/account         -> testnet balance JSON
  GET  /api/positions       -> open testnet positions JSON
  GET  /api/episodes        -> recent episodes (default last 24h)
  GET  /api/episode/{id}    -> single episode detail
  POST /api/exec/{id}       -> open paper position for episode
  POST /api/close/{id}      -> close paper position

Run: cryptoscan web
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from sqlmodel import select

from .config import settings
from .db import init_db, session_scope
from .harness import HarnessAgent
from .models import Episode
from .runtime_state import (
    KEY_ACCOUNT,
    KEY_CONTRACT_RANKINGS,
    KEY_POSITIONS,
    KEY_SCHEDULER,
    get_state,
)

TEMPLATE_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="cryptoscan", version="0.1")


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@app.get("/api/account")
def api_account(live: bool = False) -> dict[str, Any]:
    init_db()
    if not live:
        cached = get_state(KEY_ACCOUNT)
        if cached:
            value = dict(cached["value"])
            value["_cached"] = True
            value["_cached_at"] = cached["updated_at"]
            return value
        if settings.binance_testnet_key and settings.binance_testnet_secret:
            return {"error": "account cache not ready; scheduler has not written a snapshot yet"}
        return {"error": "no testnet credentials configured"}

    if not (settings.binance_testnet_key and settings.binance_testnet_secret):
        return {"error": "no testnet credentials configured"}
    from .exchange import testnet
    try:
        out = testnet.account()
        out["_cached"] = False
        return out
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/positions")
def api_positions(live: bool = False) -> list[dict[str, Any]]:
    init_db()
    if not live:
        cached = get_state(KEY_POSITIONS)
        if cached:
            value = cached["value"]
            if value.get("error"):
                return [{"error": value["error"], "_cached": True, "_cached_at": cached["updated_at"]}]
            rows = list(value.get("rows") or [])
            for row in rows:
                row["_cached"] = True
                row["_cached_at"] = cached["updated_at"]
            return rows
        if settings.binance_testnet_key and settings.binance_testnet_secret:
            return [{"error": "positions cache not ready; scheduler has not written a snapshot yet"}]
        return []

    if not (settings.binance_testnet_key and settings.binance_testnet_secret):
        return []
    from .exchange import testnet
    try:
        rows = testnet.all_open_positions()
        for row in rows:
            row["_cached"] = False
        return rows
    except Exception as e:
        return [{"error": str(e)}]


@app.get("/api/runtime")
def api_runtime() -> dict[str, Any]:
    init_db()

    def meta(key: str) -> dict[str, Any] | None:
        cached = get_state(key)
        if not cached:
            return None
        value = cached["value"]
        return {
            "updated_at": cached["updated_at"],
            "status": value.get("status"),
            "error": value.get("error"),
            "count": len(value.get("rows") or []) if isinstance(value.get("rows"), list) else None,
        }

    return {
        "scheduler": get_state(KEY_SCHEDULER),
        "account": meta(KEY_ACCOUNT),
        "positions": meta(KEY_POSITIONS),
        "contracts": meta(KEY_CONTRACT_RANKINGS),
    }


@app.get("/api/contracts")
def api_contracts(live: bool = False, limit: int = 0) -> dict[str, Any]:
    """Return Binance USDT perpetuals enriched with approximate spot market caps."""
    init_db()
    if live:
        try:
            from .tools.contract_rankings import build_contract_rankings

            payload = build_contract_rankings(include_unknown=True)
            payload["_cached"] = False
        except Exception as e:
            return {"rows": [], "error": str(e), "_cached": False}
    else:
        cached = get_state(KEY_CONTRACT_RANKINGS)
        if cached:
            payload = dict(cached["value"])
            payload["_cached"] = True
            payload["_cached_at"] = cached["updated_at"]
        else:
            return {
                "rows": [],
                "error": "contract ranking cache not ready; scheduler has not refreshed it yet",
                "_cached": True,
            }

    rows = list(payload.get("rows") or [])
    if limit and limit > 0:
        payload["rows"] = rows[:limit]
    return payload


@app.get("/api/episodes")
def api_episodes(hours: int = 48, limit: int = 200) -> list[dict[str, Any]]:
    init_db()
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    with session_scope() as s:
        rows = list(
            s.exec(select(Episode)
                   .where(Episode.created_at >= cutoff)
                   .order_by(Episode.created_at.desc())
                   .limit(limit)).all()
        )
    out: list[dict[str, Any]] = []
    for ep in rows:
        snap = ep.snapshot or {}
        out.append({
            "id": ep.id,
            "created_at": ep.created_at.isoformat(),
            "symbol": ep.symbol,
            "trigger": ep.trigger,
            "decision": ep.decision,
            "confidence": round(ep.confidence, 3),
            "executed": ep.executed,
            "closed_at": ep.closed_at.isoformat() if ep.closed_at else None,
            "pnl_pct": ep.pnl_pct,
            "outcome": ep.outcome_label,
            "tags": ep.tags or [],
            "price": snap.get("price"),
            "price_chg_24h": snap.get("price_chg_24h"),
            "curr_funding": snap.get("curr_funding"),
            "oi_change_pct": snap.get("oi_change_pct"),
            "market_cap_usd": snap.get("market_cap_usd"),
        })
    return out


@app.get("/api/episode/{episode_id}")
def api_episode(episode_id: str) -> dict[str, Any]:
    init_db()
    with session_scope() as s:
        ep = s.get(Episode, episode_id)
    if not ep:
        raise HTTPException(404, "not found")
    return ep.model_dump(mode="json")


@app.post("/api/exec/{episode_id}")
def api_exec(episode_id: str, size_usdt: float = 100.0, leverage: int = 5) -> dict[str, Any]:
    if not (settings.binance_testnet_key and settings.binance_testnet_secret):
        raise HTTPException(400, "no testnet credentials configured")
    from .exchange import testnet
    init_db()
    with session_scope() as s:
        ep = s.get(Episode, episode_id)
    if not ep:
        raise HTTPException(404, "episode not found")
    if ep.decision not in {"long", "short"}:
        raise HTTPException(400, f"episode decision is {ep.decision!r}")
    try:
        order = testnet.open_position(ep.symbol, ep.decision, size_usdt=size_usdt, leverage=leverage)
    except Exception as e:
        raise HTTPException(500, f"paper exec failed: {e}")
    HarnessAgent().mark_executed(episode_id, price=order.avg_price, size=order.qty,
                                 extra={"paper": True, "order_id": order.id, "leverage": leverage,
                                        "notional_usdt": order.notional_usdt})
    return {"ok": True, "order_id": order.id, "qty": order.qty, "avg_price": order.avg_price,
            "notional_usdt": order.notional_usdt}


@app.get("/api/stats/patterns")
def api_stats_patterns(days: int = 30) -> dict[str, Any]:
    """Aggregate closed episodes by several dimensions to see what works.

    Returns win-rate / avg PnL / count bucketed by:
      * dual tag  (dual:agree:long, dual:llm_lead:short, ...)
      * symbol
      * trigger
      * exit reason (auto_sl, auto_tp1, manual, ...)
    Plus overall totals.
    """
    init_db()
    cutoff = datetime.utcnow() - timedelta(days=days)
    with session_scope() as s:
        rows = list(s.exec(
            select(Episode)
            .where(Episode.closed_at.is_not(None))
            .where(Episode.pnl_pct.is_not(None))
            .where(Episode.closed_at >= cutoff)
        ).all())

    def bucket(key_fn) -> list[dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            for k in key_fn(r):
                if not k:
                    continue
                d = out.setdefault(k, {"key": k, "n": 0, "wins": 0, "losses": 0,
                                       "breakevens": 0, "sum_pnl": 0.0, "best": None, "worst": None})
                d["n"] += 1
                d["sum_pnl"] += r.pnl_pct
                if r.outcome_label == "win":
                    d["wins"] += 1
                elif r.outcome_label == "loss":
                    d["losses"] += 1
                else:
                    d["breakevens"] += 1
                if d["best"] is None or r.pnl_pct > d["best"]:
                    d["best"] = r.pnl_pct
                if d["worst"] is None or r.pnl_pct < d["worst"]:
                    d["worst"] = r.pnl_pct
        result = []
        for d in out.values():
            d["avg_pnl"] = round(d["sum_pnl"] / d["n"], 3) if d["n"] else 0
            d["win_rate"] = round(d["wins"] / d["n"], 3) if d["n"] else 0
            d["best"] = round(d["best"], 3) if d["best"] is not None else None
            d["worst"] = round(d["worst"], 3) if d["worst"] is not None else None
            del d["sum_pnl"]
            result.append(d)
        result.sort(key=lambda x: (-x["n"], x["avg_pnl"]))
        return result

    by_tag = bucket(lambda r: [t for t in (r.tags or [])
                               if t.startswith("dual:agree") or t.startswith("dual:llm_lead")])
    by_symbol = bucket(lambda r: [r.symbol])
    by_trigger = bucket(lambda r: [r.trigger])
    by_exit = bucket(lambda r: [(r.actual_exit or {}).get("reason") or "unknown"])

    overall = {
        "n": len(rows),
        "wins": sum(1 for r in rows if r.outcome_label == "win"),
        "losses": sum(1 for r in rows if r.outcome_label == "loss"),
        "breakevens": sum(1 for r in rows if r.outcome_label == "breakeven"),
        "avg_pnl": round(sum(r.pnl_pct for r in rows) / len(rows), 3) if rows else 0,
        "total_pnl": round(sum(r.pnl_pct for r in rows), 3) if rows else 0,
    }
    overall["win_rate"] = round(overall["wins"] / overall["n"], 3) if overall["n"] else 0

    return {
        "since_days": days,
        "overall": overall,
        "by_tag": by_tag,
        "by_symbol": by_symbol,
        "by_trigger": by_trigger,
        "by_exit_reason": by_exit,
    }


@app.post("/api/annotate/{episode_id}")
def api_annotate(episode_id: str, reflection: str = "", lessons: str = "") -> dict[str, Any]:
    """Attach post-trade reflection + lessons (comma-separated)."""
    init_db()
    lessons_list = [x.strip() for x in lessons.split(",") if x.strip()] if lessons else []
    HarnessAgent().annotate(episode_id, reflection=reflection, lessons=lessons_list)
    return {"ok": True}


@app.post("/api/close/{episode_id}")
def api_close(episode_id: str, reason: str = "manual_web") -> dict[str, Any]:
    if not (settings.binance_testnet_key and settings.binance_testnet_secret):
        raise HTTPException(400, "no testnet credentials configured")
    from .exchange import testnet
    init_db()
    with session_scope() as s:
        ep = s.get(Episode, episode_id)
    if not ep:
        raise HTTPException(404, "episode not found")
    if not ep.executed:
        raise HTTPException(400, "not executed")
    try:
        order = testnet.close_position(ep.symbol, ep.decision)
    except Exception as e:
        raise HTTPException(500, f"close failed: {e}")
    HarnessAgent().close_trade(episode_id, exit_price=order.avg_price, reason=reason)
    with session_scope() as s:
        ep2 = s.get(Episode, episode_id)
    return {"ok": True, "exit_price": order.avg_price, "pnl_pct": ep2.pnl_pct, "outcome": ep2.outcome_label}


# ---------------------------------------------------------------------------
# UI (single-page)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (TEMPLATE_DIR / "dashboard.html").read_text(encoding="utf-8")


@app.get("/contracts", response_class=HTMLResponse)
def contracts_page() -> str:
    return (TEMPLATE_DIR / "contracts.html").read_text(encoding="utf-8")


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn
    init_db()
    uvicorn.run("cryptoscan.web:app", host=host, port=port, reload=False, log_level="info")
