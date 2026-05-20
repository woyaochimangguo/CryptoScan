#!/usr/bin/env python3
"""End-to-end paper trade test:
  1. Pick a long/short episode (or one provided via --episode)
  2. Tighten its SL/TP plan to ±SL_PCT/±TP_PCT for fast triggering
  3. Open paper position on testnet
  4. Poll mark price every POLL_S seconds, log PnL%
  5. Auto-close when SL or TP hit (or after MAX_MIN minutes)
  6. Annotate with reflection

Usage:
  python scripts/test_full_trade.py --episode <id> --sl 0.4 --tp 0.6 --size 100 --leverage 5 --poll 5 --max-min 15
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

# ensure repo root on sys.path when run from project dir
sys.path.insert(0, ".")

from sqlmodel import select  # noqa: E402

from cryptoscan.db import init_db, session_scope  # noqa: E402
from cryptoscan.exchange import testnet  # noqa: E402
from cryptoscan.harness import HarnessAgent  # noqa: E402
from cryptoscan.models import Episode  # noqa: E402


def find_candidate() -> Episode | None:
    """Find an unexecuted long/short episode, prefer highest confidence."""
    with session_scope() as s:
        rows = list(
            s.exec(
                select(Episode)
                .where(
                    Episode.decision.in_(["long", "short"]),
                    Episode.executed == False,  # noqa: E712
                    Episode.closed_at == None,  # noqa: E712
                )
                .order_by(Episode.confidence.desc())
            ).all()
        )
    return rows[0] if rows else None


def tighten_plan(ep_id: str, sl_pct: float, tp_pct: float) -> None:
    with session_scope() as s:
        ep = s.get(Episode, ep_id)
        plan = dict(ep.entry_plan or {})
        plan["stop_loss_pct"] = sl_pct
        plan["take_profit_pct"] = [tp_pct]
        plan["timeframe"] = plan.get("timeframe", "test")
        plan["side"] = ep.decision
        plan["size_pct"] = plan.get("size_pct", 0.01)
        ep.entry_plan = plan
        s.add(ep)


def open_paper(ep: Episode, size_usdt: float, leverage: int):
    res = testnet.open_position(ep.symbol, ep.decision, size_usdt=size_usdt, leverage=leverage)
    HarnessAgent().mark_executed(
        ep.id,
        price=res.avg_price,
        size=res.qty,
        extra={
            "paper": True,
            "order_id": res.id,
            "leverage": leverage,
            "notional_usdt": res.notional_usdt,
        },
    )
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", help="episode id (default: pick highest-conf unexecuted)")
    p.add_argument("--sl", type=float, default=0.4, help="stop-loss pct (default 0.4)")
    p.add_argument("--tp", type=float, default=0.6, help="take-profit pct (default 0.6)")
    p.add_argument("--size", type=float, default=100.0, help="size USDT (default 100)")
    p.add_argument("--leverage", type=int, default=5, help="leverage (default 5)")
    p.add_argument("--poll", type=float, default=5.0, help="poll interval seconds (default 5)")
    p.add_argument("--max-min", type=float, default=15.0, help="max minutes before timeout-close (default 15)")
    args = p.parse_args()

    init_db()

    if args.episode:
        with session_scope() as s:
            ep = s.get(Episode, args.episode)
        if not ep:
            print(f"[error] episode {args.episode} not found")
            sys.exit(2)
    else:
        ep = find_candidate()
        if not ep:
            print("[error] no unexecuted long/short episode in DB. Run `cryptoscan scan --dual` first.")
            sys.exit(2)

    print(f"=== STEP 1: candidate episode ===")
    print(f"  id       : {ep.id}")
    print(f"  symbol   : {ep.symbol}")
    print(f"  decision : {ep.decision}  conf={ep.confidence:.2f}")
    print(f"  rationale: {(ep.rationale or '')[:140]}{'...' if len(ep.rationale or '')>140 else ''}")

    if ep.executed:
        print(f"[error] episode already executed (entry={ep.actual_entry}). pick another.")
        sys.exit(2)

    print(f"\n=== STEP 2: tighten plan to SL=-{args.sl}% / TP=+{args.tp}% (was {ep.entry_plan}) ===")
    tighten_plan(ep.id, sl_pct=args.sl, tp_pct=args.tp)

    print(f"\n=== STEP 3: open paper position size=${args.size} {args.leverage}x ===")
    t0 = time.time()
    res = open_paper(ep, args.size, args.leverage)
    print(f"  filled in {time.time()-t0:.1f}s  order={res.id}  avg=${res.avg_price}  qty={res.qty}")
    sl_price = res.avg_price * (1 - (1 if ep.decision == "long" else -1) * args.sl / 100)
    tp_price = res.avg_price * (1 + (1 if ep.decision == "long" else -1) * args.tp / 100)
    print(f"  SL price : ${sl_price:.6f}   TP price: ${tp_price:.6f}")

    print(f"\n=== STEP 4: poll every {args.poll}s (max {args.max_min} min) ===")
    deadline = time.time() + args.max_min * 60
    sign = 1 if ep.decision == "long" else -1
    last_print = 0.0
    closed_reason = None
    closed_price = None
    while time.time() < deadline:
        try:
            pos = testnet.current_position(ep.symbol)
        except Exception as e:
            print(f"  [poll-err] {e}")
            time.sleep(args.poll)
            continue
        if not pos:
            print(f"  [poll] position vanished — already closed elsewhere")
            break
        mark = pos["mark_price"]
        chg = (mark - res.avg_price) / res.avg_price * 100 * sign
        ts = datetime.utcnow().strftime("%H:%M:%S")
        print(f"  {ts}  mark=${mark:.6f}  pnl={chg:+.3f}%   uPnL=${pos['unrealized_pnl']:+.4f}")

        if chg <= -args.sl:
            closed_reason = "auto_sl"; break
        if chg >= args.tp:
            closed_reason = "auto_tp1"; break
        time.sleep(args.poll)
    else:
        closed_reason = "auto_timeout"

    print(f"\n=== STEP 5: closing ({closed_reason}) ===")
    t0 = time.time()
    co = testnet.close_position(ep.symbol, ep.decision)
    closed_price = co.avg_price
    HarnessAgent().close_trade(ep.id, exit_price=closed_price, reason=closed_reason)
    print(f"  closed in {time.time()-t0:.1f}s  exit=${closed_price}  qty={co.qty}")

    # Refresh and print final PnL
    with session_scope() as s:
        ep2 = s.get(Episode, ep.id)
    print(f"  PnL  : {ep2.pnl_pct:+.3f}%   outcome: {ep2.outcome_label}")

    print(f"\n=== STEP 6: writing reflection ===")
    entry_ts = ep2.actual_entry.get("ts") if ep2.actual_entry else None
    if isinstance(entry_ts, str):
        try:
            entry_ts = datetime.fromisoformat(entry_ts)
        except ValueError:
            entry_ts = None
    held = f"{(datetime.utcnow() - entry_ts).total_seconds():.0f}s" if isinstance(entry_ts, datetime) else "N/A"
    refl = (
        f"E2E auto-close test on {ep.symbol}. SL={args.sl}% TP={args.tp}% poll={args.poll}s. "
        f"Triggered: {closed_reason}. Final PnL {ep2.pnl_pct:+.3f}% over {held}. "
        f"Polling pipeline confirmed working."
    )
    HarnessAgent().annotate(
        ep.id,
        reflection=refl,
        lessons=[
            "auto-close-pipeline-works",
            f"trigger-{closed_reason}",
            f"SL{args.sl}_TP{args.tp}_realized_{ep2.pnl_pct:+.2f}pct",
        ],
    )
    print(f"  done. View at: http://127.0.0.1:8766/  (click {ep.symbol} row → detail)")


if __name__ == "__main__":
    main()
