#!/usr/bin/env python3
"""Quick demo: print full state of an episode after open+close+annotate."""
import sys
import httpx

ep_id = sys.argv[1] if len(sys.argv) > 1 else "5b050c948c7b"
base = "http://127.0.0.1:8766"

d = httpx.get(f"{base}/api/episode/{ep_id}", timeout=10).json()
plan = d["entry_plan"]
ae = d.get("actual_entry") or {}
ax = d.get("actual_exit") or {}

print(f"symbol     : {d['symbol']}")
print(f"plan       : SL={plan['stop_loss_pct']}%  TP={plan['take_profit_pct']}  TF={plan['timeframe']}")

if ae.get("price"):
    sign = 1 if plan["side"] == "long" else -1
    sl_price = ae["price"] * (1 - sign * abs(plan["stop_loss_pct"]) / 100)
    tp_prices = [ae["price"] * (1 + sign * p / 100) for p in plan["take_profit_pct"]]
    print(f"entry      : ${ae['price']}  qty={ae['size']}  order={ae.get('order_id')}")
    print(f"  SL price : ${sl_price:.6f}  ({plan['stop_loss_pct']}%)")
    for i, p in enumerate(tp_prices, 1):
        print(f"  TP{i} price: ${p:.6f}  (+{plan['take_profit_pct'][i-1]}%)")

if ax.get("price"):
    print(f"exit       : ${ax['price']}  reason: {ax.get('reason')}")
    print(f"PnL        : {d['pnl_pct']:+.3f}%   outcome: {d.get('outcome_label')}")

print()
print("--- 开仓理由 (rationale) ---")
print(d.get("rationale") or "(none)")
print()
print("--- 反思 (reflection) ---")
print(d.get("reflection") or "(none)")
print()
print(f"--- 经验 ({len(d.get('lessons') or [])} lessons) ---")
for l in d.get("lessons") or []:
    print(f"  - {l}")
