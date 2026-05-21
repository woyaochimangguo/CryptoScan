"""APScheduler-based supervisor:
  - Every N minutes: scan_oi_funding_flip + DualPolicy + push consensus to TG.
  - Every K seconds: reconcile open testnet positions, close on SL/TP hit, mark
    PnL on the originating episode.

Run with `cryptoscan run` (foreground). Ctrl-C to stop.
"""
from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from sqlmodel import select

from .config import settings
from .db import init_db, session_scope
from .harness import HarnessAgent
from .models import Episode
from .notify import send_episode
from .runtime_state import KEY_ACCOUNT, KEY_CONTRACT_RANKINGS, KEY_POSITIONS, mark_scheduler, set_state

log = logging.getLogger("cryptoscan.scheduler")


def _cache_state(key: str, value: dict) -> None:
    try:
        set_state(key, value)
    except Exception as e:
        log.warning("state cache write failed for %s: %s", key, e)


def _mark_scheduler(status: str, **extra) -> None:
    try:
        mark_scheduler(status, **extra)
    except Exception as e:
        log.warning("scheduler status write failed: %s", e)


def refresh_exchange_snapshot() -> None:
    """Refresh cached account/position data for the dashboard.

    Web reads these rows and does not depend on Binance being reachable.
    Trading actions still use live exchange calls.
    """
    if not (settings.binance_testnet_key and settings.binance_testnet_secret):
        _cache_state(KEY_ACCOUNT, {"error": "no testnet credentials configured"})
        _cache_state(KEY_POSITIONS, {"rows": []})
        return

    from .exchange import testnet

    try:
        acct = testnet.account()
        _cache_state(KEY_ACCOUNT, acct)
    except Exception as e:
        _cache_state(KEY_ACCOUNT, {"error": str(e)})
        log.warning("exchange_snapshot: account fetch failed: %s", e)

    try:
        rows = testnet.all_open_positions()
        _cache_state(KEY_POSITIONS, {"rows": rows})
    except Exception as e:
        _cache_state(KEY_POSITIONS, {"rows": [], "error": str(e)})
        log.warning("exchange_snapshot: positions fetch failed: %s", e)


def refresh_contract_rankings() -> None:
    """Refresh cached Binance USDT perpetuals enriched with spot market caps."""
    try:
        from .tools.contract_rankings import build_contract_rankings

        payload = build_contract_rankings(include_unknown=True)
        _cache_state(KEY_CONTRACT_RANKINGS, payload)
        log.info(
            "contract_rankings: refreshed %d rows (%d unknown market caps)",
            payload.get("total", 0),
            payload.get("unknown_market_cap", 0),
        )
    except Exception as e:
        _cache_state(KEY_CONTRACT_RANKINGS, {"rows": [], "error": str(e)})
        log.warning("contract_rankings: refresh failed: %s", e)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def job_scan_dual(consensus_only_notify: bool = True) -> None:
    """Scan enabled strategies + policy + optional TG notify."""
    from .strategies import enabled_strategies

    strategies = enabled_strategies()
    log.info("scan_dual: start strategies=%s", ",".join(s.id for s in strategies))
    _mark_scheduler("scanning", job="scan_dual", strategies=[s.id for s in strategies])
    created: list[Episode] = []
    candidate_count = 0

    for strategy in strategies:
        try:
            signals = strategy.scan()
        except Exception as e:
            log.error("strategy scan failed %s: %s", strategy.id, e)
            continue
        candidate_count += len(signals)
        if not signals:
            log.info("strategy %s: no candidates", strategy.id)
            continue

        try:
            strategy.warm(signals)
        except Exception as e:
            log.warning("strategy warm failed %s (continuing): %s", strategy.id, e)

        agent = HarnessAgent(policy=strategy.policy("dual"))
        for sig in signals:
            try:
                ep = agent.handle_strategy_signal(strategy, sig, policy_id="dual")
            except Exception as e:
                log.warning("handle_signal failed for %s/%s: %s", strategy.id, sig.symbol, e)
                continue
            if ep:
                created.append(ep)

    if candidate_count == 0:
        log.info("scan_dual: no candidates")
        _mark_scheduler("idle", job="scan_dual", candidates=0, episodes=0, pushed=0)
        return

    pushed = 0
    for ep in created:
        actionable = any(
            (t or "").startswith("dual:agree") or (t or "").startswith("dual:llm_lead")
            for t in (ep.tags or [])
        )
        push = ep.decision in {"long", "short"}
        if consensus_only_notify:
            push = push and actionable
        if push:
            try:
                send_episode(ep)
                with session_scope() as s:
                    db_ep = s.get(Episode, ep.id)
                    if db_ep:
                        db_ep.notified = True
                        s.add(db_ep)
                pushed += 1
            except Exception as e:
                log.warning("notify failed for %s: %s", ep.id, e)
    log.info("scan_dual: %d episodes / %d pushed (from %d candidates)", len(created), pushed, candidate_count)
    _mark_scheduler("idle", job="scan_dual", candidates=candidate_count, episodes=len(created), pushed=pushed)


def job_position_watch() -> None:
    """For each executed-but-open episode with a SL/TP plan, check live price
    on testnet and close if hit."""
    from .exchange import testnet
    if not (settings.binance_testnet_key and settings.binance_testnet_secret):
        return  # paper trading not configured

    try:
        live = {p["symbol"]: p for p in testnet.all_open_positions()}
        _cache_state(KEY_POSITIONS, {"rows": list(live.values())})
    except Exception as e:
        _cache_state(KEY_POSITIONS, {"rows": [], "error": str(e)})
        log.warning("position_watch: fetch failed: %s", e)
        return
    if not live:
        return

    with session_scope() as s:
        rows = list(s.exec(select(Episode).where(Episode.executed == True, Episode.closed_at == None)).all())  # noqa: E712

    for ep in rows:
        ae = ep.actual_entry or {}
        if not ae.get("paper"):
            continue
        pos = live.get(ep.symbol)
        if not pos:
            continue
        side = ep.decision
        entry = float(ae.get("price") or 0)
        if entry <= 0:
            continue
        plan = ep.entry_plan or {}
        sl = plan.get("stop_loss_pct")
        tps = plan.get("take_profit_pct") or []
        first_tp = tps[0] if tps else None
        mark = float(pos["mark_price"])
        chg_pct = (mark - entry) / entry * 100 * (1 if side == "long" else -1)

        hit_sl = sl is not None and chg_pct <= -abs(float(sl))
        hit_tp = first_tp is not None and chg_pct >= float(first_tp)

        if hit_sl or hit_tp:
            reason = "sl" if hit_sl else "tp1"
            log.info("position_watch: %s %s @ %+.2f%% -> closing (%s)", ep.symbol, side, chg_pct, reason)
            try:
                order = testnet.close_position(ep.symbol, side)
                HarnessAgent().close_trade(ep.id, exit_price=order.avg_price, reason=f"auto_{reason}")
            except Exception as e:
                log.warning("auto-close failed for %s: %s", ep.id, e)


def job_auto_execute() -> None:
    """Auto paper-trade fresh consensus episodes within safety guardrails:

      * dual:agree-* tag required (rule + LLM aligned)
      * decision must be long/short
      * not yet executed, not yet closed
      * created within auto_execute_max_age_minutes
      * symbol has no existing open testnet position
      * total open positions < auto_execute_max_concurrent
      * USDT free balance >= auto_execute_min_free_usdt

    Plan SL/TP are overridden to tight defaults so position_watch can close them
    on a realistic horizon.
    """
    if not settings.auto_execute_enabled:
        return
    if not (settings.binance_testnet_key and settings.binance_testnet_secret):
        return

    from .exchange import testnet

    # Account & live positions
    try:
        acct = testnet.account()
        live = testnet.all_open_positions()
        _cache_state(KEY_ACCOUNT, acct)
        _cache_state(KEY_POSITIONS, {"rows": live})
    except Exception as e:
        log.warning("auto_execute: account/positions fetch failed: %s", e)
        return

    free_usdt = float(acct.get("free_usdt") or 0)
    if free_usdt < settings.auto_execute_min_free_usdt:
        log.info("auto_execute: skip — free=$%.2f < min=$%.2f",
                 free_usdt, settings.auto_execute_min_free_usdt)
        return

    open_symbols = {p["symbol"] for p in live}
    if len(open_symbols) >= settings.auto_execute_max_concurrent:
        log.info("auto_execute: skip — at max concurrent (%d)", len(open_symbols))
        return

    cutoff = datetime.utcnow() - timedelta(minutes=settings.auto_execute_max_age_minutes)
    with session_scope() as s:
        rows = list(s.exec(
            select(Episode)
            .where(Episode.executed == False, Episode.closed_at == None)  # noqa: E712
            .where(Episode.created_at >= cutoff)
            .order_by(Episode.created_at.desc())
        ).all())

    tps = [float(x) for x in settings.auto_execute_tp_pcts.split(",") if x.strip()]
    sl_abs = abs(settings.auto_execute_sl_pct)
    notional = settings.auto_execute_notional_usdt
    slots = settings.auto_execute_max_concurrent - len(open_symbols)

    for ep in rows:
        if slots <= 0:
            break
        if ep.decision not in {"long", "short"}:
            continue
        tags = ep.tags or []
        if not any(
            (t or "").startswith("dual:agree") or (t or "").startswith("dual:llm_lead")
            for t in tags
        ):
            continue
        if ep.symbol in open_symbols:
            continue

        # Tighten plan
        with session_scope() as s:
            db_ep = s.get(Episode, ep.id)
            if not db_ep or db_ep.executed:
                continue
            plan = dict(db_ep.entry_plan or {})
            plan["stop_loss_pct"] = sl_abs
            plan["take_profit_pct"] = tps
            plan["side"] = db_ep.decision
            plan["size_pct"] = notional / max(free_usdt, 1.0)
            db_ep.entry_plan = plan
            s.add(db_ep)

        # Open paper position
        try:
            order = testnet.open_position(ep.symbol, ep.decision, size_usdt=notional)
        except Exception as e:
            log.warning("auto_execute: open %s %s failed: %s", ep.symbol, ep.decision, e)
            continue

        HarnessAgent().mark_executed(
            ep.id, price=order.avg_price, size=order.qty,
            extra={"paper": True, "order_id": order.id,
                   "leverage": settings.paper_leverage,
                   "notional_usdt": order.notional_usdt,
                   "auto": True},
        )
        log.info("auto_execute: opened %s %s qty=%s avg=$%s (ep=%s)",
                 ep.symbol, ep.decision, order.qty, order.avg_price, ep.id[:8])
        open_symbols.add(ep.symbol)
        slots -= 1


def job_auto_reflect() -> None:
    """Scan recently-closed episodes with empty reflection and ask the LLM
    to write one. Bounded per tick so a burst of closes can't monopolize."""
    from .llm_clients import is_configured
    if not is_configured("reflection"):
        return  # no LLM configured
    from .harness.memory import auto_reflect_episode

    max_per_tick = 3
    cutoff = datetime.utcnow() - timedelta(days=2)
    with session_scope() as s:
        rows = list(s.exec(
            select(Episode)
            .where(Episode.closed_at.is_not(None))
            .where(Episode.closed_at >= cutoff)
            .where((Episode.reflection == "") | (Episode.reflection.is_(None)))
            .order_by(Episode.closed_at.desc())
            .limit(max_per_tick)
        ).all())
        target_ids = [r.id for r in rows]

    for ep_id in target_ids:
        try:
            auto_reflect_episode(ep_id)
        except Exception as e:
            log.warning("auto_reflect failed for %s: %s", ep_id[:8], e)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(consensus_only_notify: bool = True) -> None:
    init_db()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("starting scheduler  scan_every=%dmin  watch_every=%ds",
             settings.scan_interval_minutes, settings.position_watch_interval_seconds)
    _mark_scheduler("starting")
    try:
        from .llm_clients import describe_routing
        log.info("LLM routing:\n%s", describe_routing())
    except Exception:
        pass

    sch = BackgroundScheduler(timezone="UTC")
    sch.add_job(
        job_scan_dual,
        "interval",
        minutes=settings.scan_interval_minutes,
        kwargs={"consensus_only_notify": consensus_only_notify},
        id="scan_dual",
        next_run_time=datetime.now(timezone.utc),  # run once immediately
        max_instances=1,
        coalesce=True,
    )
    sch.add_job(
        refresh_exchange_snapshot,
        "interval",
        seconds=max(30, settings.position_watch_interval_seconds),
        id="exchange_snapshot",
        next_run_time=datetime.now(timezone.utc),
        max_instances=1,
        coalesce=True,
    )
    sch.add_job(
        refresh_contract_rankings,
        "interval",
        seconds=max(300, settings.contract_rankings_interval_seconds),
        id="contract_rankings",
        next_run_time=datetime.now(timezone.utc),
        max_instances=1,
        coalesce=True,
    )
    if settings.binance_testnet_key and settings.binance_testnet_secret:
        sch.add_job(
            job_position_watch,
            "interval",
            seconds=settings.position_watch_interval_seconds,
            id="position_watch",
            max_instances=1,
            coalesce=True,
        )
        log.info("position_watch enabled")
        if settings.auto_execute_enabled:
            sch.add_job(
                job_auto_execute,
                "interval",
                seconds=settings.auto_execute_interval_seconds,
                id="auto_execute",
                next_run_time=datetime.now(timezone.utc) + timedelta(seconds=15),
                max_instances=1,
                coalesce=True,
            )
            log.info(
                "auto_execute enabled  notional=$%.0f  max_concurrent=%d  SL=-%.2f%% TP=%s",
                settings.auto_execute_notional_usdt,
                settings.auto_execute_max_concurrent,
                settings.auto_execute_sl_pct,
                settings.auto_execute_tp_pcts,
            )
    else:
        log.info("position_watch disabled (no testnet credentials)")

    # Auto-reflection job (runs regardless of testnet; only needs an LLM)
    from .llm_clients import is_configured
    if is_configured("reflection"):
        sch.add_job(
            job_auto_reflect,
            "interval",
            seconds=max(60, settings.auto_reflect_interval_seconds),
            id="auto_reflect",
            next_run_time=datetime.now(timezone.utc) + timedelta(seconds=45),
            max_instances=1,
            coalesce=True,
        )
        log.info("auto_reflect enabled  every=%ds", settings.auto_reflect_interval_seconds)

    sch.start()
    _mark_scheduler("idle")

    stop_evt = threading.Event()
    def _stop(*_a):
        log.info("shutting down...")
        _mark_scheduler("stopping")
        stop_evt.set()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        while not stop_evt.is_set():
            time.sleep(1)
    finally:
        sch.shutdown(wait=False)
        _mark_scheduler("stopped")
        log.info("bye")
