"""Episodic memory utilities.

Two capabilities:

1. retrieve_similar_episodes(symbol, snapshot) -> list of closed past episodes
   ranked by relevance (same symbol first, then same side/tag bucket). Used by
   LLMPolicy to give the model "what happened last time I saw this setup".

2. auto_reflect_episode(episode_id) -> asks the LLM to write a concise
   post-mortem for a just-closed trade and persists it back via
   HarnessAgent.annotate().

Both are best-effort: they swallow and log errors rather than raising, so they
can't break the trading loop.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any

from sqlmodel import select

from ..config import settings
from ..db import session_scope
from ..models import Episode

log = logging.getLogger("cryptoscan.memory")


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> dict | None:
    """Best-effort extract the first top-level {...} JSON object from a blob
    that might be wrapped with DeepSeek-R1 style <think>...</think>, markdown
    code fences, or trailing commentary. Returns None on failure."""
    if not text:
        return None
    # 1. strip R1-style reasoning
    text = _THINK_RE.sub("", text).strip()
    # 2. strip ``` or ```json fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    # 3. try direct parse
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    # 4. find the first balanced {...} block
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = text[start:i + 1]
                        try:
                            v = json.loads(chunk)
                            return v if isinstance(v, dict) else None
                        except Exception:
                            break
        start = text.find("{", start + 1)
    return None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_similar_episodes(
    symbol: str,
    snapshot: dict[str, Any] | None = None,
    limit: int = 5,
    lookback_days: int = 30,
) -> list[Episode]:
    """Return up to `limit` closed past episodes most relevant to this setup.

    Ranking (cheap, deterministic, no embeddings):
      1) same symbol first
      2) among same-symbol, most recent first
      3) backfill with different-symbol ones that share a tag prefix
         (e.g. both are 'funding_negative + perp_only')
    Only closed episodes (with pnl_pct) are returned — we need outcomes to
    learn from.
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    try:
        with session_scope() as s:
            rows = list(
                s.exec(
                    select(Episode)
                    .where(Episode.closed_at.is_not(None))
                    .where(Episode.pnl_pct.is_not(None))
                    .where(Episode.created_at >= cutoff)
                    .order_by(Episode.created_at.desc())
                ).all()
            )
    except Exception as e:
        log.warning("retrieve_similar_episodes: DB read failed: %s", e)
        return []

    same_sym = [r for r in rows if r.symbol == symbol]
    others = [r for r in rows if r.symbol != symbol]

    # Tag-based similarity for cross-symbol backfill
    snap_tags: set[str] = set()
    if snapshot:
        if float(snapshot.get("curr_funding") or 0) < 0:
            snap_tags.add("funding_negative")
        if snapshot.get("oi_rising"):
            snap_tags.add("oi_rising")
        if snapshot.get("has_spot"):
            snap_tags.add("has_spot")
        else:
            snap_tags.add("perp_only")

    def tag_overlap(ep: Episode) -> int:
        return len(set(ep.tags or []) & snap_tags)

    others_ranked = sorted(others, key=lambda r: (-tag_overlap(r), -r.created_at.timestamp()))

    return (same_sym + others_ranked)[:limit]


def summarize_for_prompt(eps: list[Episode], max_chars: int = 1800) -> str:
    """One-line-per-episode bullet summary suitable for inline LLM prompt."""
    if not eps:
        return "(no prior closed trades for this symbol/pattern)"
    lines = ["Past closed trades (most relevant first):"]
    for e in eps:
        ae = e.actual_entry or {}
        ax = e.actual_exit or {}
        snap = e.snapshot or {}
        when = e.created_at.strftime("%m-%d %H:%M") if e.created_at else "?"
        pnl = f"{e.pnl_pct:+.2f}%" if e.pnl_pct is not None else "?"
        side = (e.entry_plan or {}).get("side") or e.decision
        oi = snap.get("oi_change_pct")
        fr = snap.get("curr_funding")
        tag_str = ",".join(
            t for t in (e.tags or [])
            if t.startswith("dual:") or t in {"funding_negative", "oi_rising", "perp_only", "has_spot"}
        )
        reason = (ax.get("reason") or "").replace("auto_", "")
        lesson = "; ".join((e.lessons or [])[:2])
        refl = (e.reflection or "").strip().replace("\n", " ")
        if len(refl) > 120:
            refl = refl[:120] + "…"
        parts = [
            f"- {when} {e.symbol} {side} pnl={pnl} exit={reason or '?'}",
            f"  ctx: oi_chg={oi:+.1f}% fr={fr:+.4%}" if isinstance(oi, (int, float)) and isinstance(fr, (int, float)) else None,
            f"  tags: {tag_str}" if tag_str else None,
            f"  lessons: {lesson}" if lesson else None,
            f"  note: {refl}" if refl else None,
        ]
        lines.extend([p for p in parts if p])
        if sum(len(l) for l in lines) > max_chars:
            break
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-reflection
# ---------------------------------------------------------------------------

REFLECT_SYSTEM = """You are reviewing a closed crypto futures trade to help a
systematic trading bot learn. Be blunt, concrete, and brief.

Return STRICT JSON with exactly these keys:
  "reflection": 1-3 sentences explaining what actually happened and whether
                the original thesis was right/wrong/irrelevant.
  "lessons":    list of 2-4 short imperative lessons (each <= 60 chars), e.g.
                "avoid shorting into positive-funding spike" or
                "1.5% SL too tight for AXS-class alts".

Do not include any text outside the JSON object.
"""


def _build_reflect_prompt(ep: Episode) -> str:
    ae = ep.actual_entry or {}
    ax = ep.actual_exit or {}
    snap = ep.snapshot or {}
    plan = ep.entry_plan or {}

    # Keep snapshot compact — only fields that matter for learning.
    snap_compact = {
        "price": snap.get("price"),
        "price_chg_24h": snap.get("price_chg_24h"),
        "volume_24h_usd": snap.get("volume_24h"),
        "market_cap_usd": snap.get("market_cap_usd"),
        "has_spot": snap.get("has_spot"),
        "prev_funding": snap.get("prev_funding"),
        "curr_funding": snap.get("curr_funding"),
        "oi_change_pct": snap.get("oi_change_pct"),
        "oi_rising": snap.get("oi_rising"),
    }
    held_sec = None
    if ep.closed_at and isinstance(ae.get("ts"), str):
        try:
            entry_ts = datetime.fromisoformat(ae["ts"])
            held_sec = int((ep.closed_at - entry_ts).total_seconds())
        except Exception:
            pass

    payload = {
        "symbol": ep.symbol,
        "side": plan.get("side") or ep.decision,
        "snapshot_at_entry": snap_compact,
        "rule_llm_tags": [t for t in (ep.tags or []) if t.startswith("dual:")],
        "original_rationale": (ep.rationale or "")[:800],
        "plan": {
            "stop_loss_pct": plan.get("stop_loss_pct"),
            "take_profit_pct": plan.get("take_profit_pct"),
        },
        "actual_entry_price": ae.get("price"),
        "actual_exit_price": ax.get("price"),
        "exit_reason": ax.get("reason"),
        "held_seconds": held_sec,
        "pnl_pct": ep.pnl_pct,
        "pnl_usd": ep.pnl_usd,
        "outcome_label": ep.outcome_label,
    }
    return "CLOSED TRADE:\n" + json.dumps(payload, default=str, indent=2)


def auto_reflect_episode(episode_id: str, *, timeout_sec: float = 30.0) -> bool:
    """Generate + persist reflection/lessons for a closed episode.

    Returns True on success, False on any failure (which are logged).
    """
    with session_scope() as s:
        ep = s.get(Episode, episode_id)
        if not ep:
            return False
        if ep.closed_at is None:
            log.info("auto_reflect: %s not closed, skipping", episode_id[:8])
            return False
        if ep.reflection and len(ep.reflection.strip()) > 10:
            return False  # already reflected
        ep_copy = ep  # session is closed after this block; attrs are loaded
        ep_data = {
            "id": ep.id,
        }

    # Build prompt outside session
    prompt = _build_reflect_prompt(ep_copy)

    try:
        from ..llm_clients import get_client, resolve
    except ImportError:
        log.warning("auto_reflect: llm_clients module missing")
        return False

    try:
        client = get_client("reflection", timeout_sec=timeout_sec)
        model = resolve("reflection").model
    except Exception as e:
        log.warning("auto_reflect: client init failed for %s: %s", episode_id[:8], e)
        return False

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REFLECT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
    except Exception as e:
        log.warning("auto_reflect: LLM call failed for %s: %s", episode_id[:8], e)
        return False

    obj = _extract_json_object(raw)
    if obj is None:
        log.warning("auto_reflect: JSON parse failed for %s | raw=%r",
                    episode_id[:8], raw[:240])
        return False
    reflection = str(obj.get("reflection") or "").strip()
    lessons = [str(x).strip() for x in (obj.get("lessons") or []) if str(x).strip()]

    if not reflection:
        log.warning("auto_reflect: empty reflection for %s", episode_id[:8])
        return False

    # Persist — avoid importing HarnessAgent to prevent circular imports.
    with session_scope() as s:
        ep2 = s.get(Episode, episode_id)
        if not ep2:
            return False
        ep2.reflection = reflection
        if lessons:
            ep2.lessons = list({*(ep2.lessons or []), *lessons})
        s.add(ep2)

    log.info("auto_reflect: %s reflected (%d lessons)", episode_id[:8], len(lessons))
    return True
