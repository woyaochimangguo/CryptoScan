from __future__ import annotations

from typing import Any

import httpx

from ..config import settings
from ..models import Episode


def _fmt_mcap(v: float) -> str:
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.1f}M"
    if v >= 1e3:
        return f"${v/1e3:.0f}K"
    return f"${v:.0f}" if v else "?"


def _fmt_views(v: float) -> str:
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}K"
    return str(int(v))


def send_text(text: str, parse_mode: str = "Markdown") -> bool:
    if not settings.tg_bot_token or not settings.tg_chat_id:
        print("[TG-DRY]\n" + text)
        return False
    url = f"https://api.telegram.org/bot{settings.tg_bot_token}/sendMessage"
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
    ok = True
    for chunk in chunks:
        try:
            r = httpx.post(
                url,
                json={"chat_id": settings.tg_chat_id, "text": chunk, "parse_mode": parse_mode},
                timeout=10,
            )
            if r.status_code != 200:
                # fallback no formatting
                httpx.post(url, json={"chat_id": settings.tg_chat_id, "text": chunk}, timeout=10)
        except Exception as e:
            print(f"[TG] send failed: {e}")
            ok = False
    return ok


def render_episode(ep: Episode) -> str:
    snap: dict[str, Any] = ep.snapshot or {}
    coin = snap.get("coin") or ep.symbol.replace("USDT", "")
    head = f"*[{ep.trigger}]* `{coin}`  →  *{ep.decision.upper()}* (conf {ep.confidence:.2f})"
    lines = [head, ""]
    lines.append("```")
    lines.append(f"price       {snap.get('price', 0):.6f}  24h {snap.get('price_chg_24h', 0):+.2f}%")
    lines.append(
        f"funding     {snap.get('prev_funding', 0):+.4%} -> {snap.get('curr_funding', 0):+.4%}"
    )
    if snap.get("oi_segments"):
        segs = " > ".join(f"{v/1e6:.1f}M" for v in snap["oi_segments"])
        lines.append(f"OI          {snap.get('oi_change_pct', 0):+.1f}%  ({segs})")
    lines.append(f"volume 24h  ${snap.get('volume_24h', 0)/1e6:.1f}M")
    lines.append(
        f"mcap        {_fmt_mcap(snap.get('market_cap_usd', 0))}   spot: {'yes' if snap.get('has_spot') else 'perp-only'}"
    )
    if snap.get("square_posts"):
        lines.append(
            f"square      {snap.get('square_posts')} posts / {_fmt_views(snap.get('square_views', 0))} views"
        )
    lines.append("```")
    if ep.rationale:
        lines.append(f"_{ep.rationale}_")
    if ep.entry_plan:
        ep_plan = ep.entry_plan
        lines.append("")
        lines.append(
            f"*plan*: {ep_plan.get('side', '?')} size~{ep_plan.get('size_pct', 0)*100:.1f}%"
            f"  SL {ep_plan.get('stop_loss_pct', 0)}%  TP {ep_plan.get('take_profit_pct')}"
        )
    lines.append("")
    lines.append(f"`/exec {ep.id} <price> <size>`   `/skip {ep.id}`")
    return "\n".join(lines)


def send_episode(ep: Episode) -> bool:
    return send_text(render_episode(ep))
