from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from sqlmodel import select

from .db import init_db, session_scope
from .harness import HarnessAgent
from .models import Episode
from .notify import send_episode
from .tools.derivs_scanner import scan_oi_funding_flip

app = typer.Typer(help="cryptoscan — harness-agent trade journal & scanner")
console = Console()


@app.command()
def init() -> None:
    """Create the SQLite database and tables."""
    init_db()
    console.print("[green]db ready[/green]")


@app.command()
def scan(
    notify: bool = True,
    min_volume: Optional[float] = None,
    llm: bool = False,
    dual: bool = False,
    consensus_only_notify: bool = True,
) -> None:
    """Run the OI+funding-flip scanner once and create episodes for new signals.

    --llm   : pure LLM policy
    --dual  : run rule first; only escalate to LLM on long/short candidates;
              episode tagged with consensus / disagreement.
    --consensus-only-notify (default): in dual mode, only push TG when rule+LLM agree.
    """
    init_db()
    try:
        raw = scan_oi_funding_flip(min_volume_usdt=min_volume)
    except Exception as e:
        console.print(f"[red]scan failed:[/red] {type(e).__name__}: {e}")
        console.print("[dim]Check network reachability to fapi.binance.com (try `cryptoscan doctor`).[/dim]")
        raise typer.Exit(2)
    if not raw:
        console.print(f"[dim]{datetime.utcnow().isoformat(timespec='seconds')}Z  no flip detected[/dim]")
        return

    # Warm shared/per-coin caches in parallel before iterating.
    from .tools.binance_market import prefetch_square_hashtags, market_caps, spot_symbols
    market_caps(); spot_symbols()
    prefetch_square_hashtags([sig["symbol"].replace("USDT", "") for sig in raw])

    if dual:
        from .harness.llm_policy import LLMPolicy
        from .harness.dual_policy import DualPolicy
        agent = HarnessAgent(policy=DualPolicy(llm_policy=LLMPolicy()))
        console.print("[cyan]using DUAL policy (rule -> LLM only on long/short)[/cyan]")
    elif llm:
        from .harness.llm_policy import LLMPolicy
        agent = HarnessAgent(policy=LLMPolicy())
        console.print(f"[cyan]using LLM policy ({agent.policy.model or 'default'})[/cyan]")
    else:
        agent = HarnessAgent()
    created: list[Episode] = []
    for sig in raw:
        ep = agent.handle_signal(
            trigger="oi_funding_flip",
            symbol=sig["symbol"],
            base_signal=sig,
        )
        if ep:
            created.append(ep)

    console.print(f"[cyan]{len(created)} new episode(s) from {len(raw)} candidates[/cyan]")
    for ep in created:
        consensus = any(t.startswith("dual:agree") for t in (ep.tags or []))
        disagree = any(t.startswith("dual:disagree") for t in (ep.tags or []))
        marker = "✓" if consensus else ("≠" if disagree else " ")
        console.print(f"  {marker} {ep.id}  {ep.symbol}  -> {ep.decision} (conf {ep.confidence:.2f})")
        if not notify:
            continue
        # Gating: in dual mode require consensus; otherwise just long/short.
        push = ep.decision in {"long", "short"}
        if dual and consensus_only_notify:
            push = push and consensus
        if push:
            send_episode(ep)
            with session_scope() as s:
                ep_db = s.get(Episode, ep.id)
                if ep_db:
                    ep_db.notified = True
                    s.add(ep_db)


@app.command("review")
def review_today(hours: int = 24) -> None:
    """List recent episodes."""
    init_db()
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    with session_scope() as s:
        rows = s.exec(
            select(Episode).where(Episode.created_at >= cutoff).order_by(Episode.created_at.desc())
        ).all()

    table = Table(title=f"episodes (last {hours}h)")
    table.add_column("id")
    table.add_column("time")
    table.add_column("symbol")
    table.add_column("decision")
    table.add_column("conf")
    table.add_column("exec")
    table.add_column("pnl%")
    table.add_column("tags")
    for ep in rows:
        table.add_row(
            ep.id,
            ep.created_at.strftime("%m-%d %H:%M"),
            ep.symbol,
            ep.decision,
            f"{ep.confidence:.2f}",
            "Y" if ep.executed else "-",
            f"{ep.pnl_pct:+.2f}" if ep.pnl_pct is not None else "-",
            ",".join(ep.tags or []),
        )
    console.print(table)


@app.command("show")
def show(episode_id: str) -> None:
    """Show full detail of one episode."""
    init_db()
    with session_scope() as s:
        ep = s.get(Episode, episode_id)
    if not ep:
        console.print("[red]not found[/red]")
        raise typer.Exit(1)
    console.print_json(data=ep.model_dump(mode="json"))


@app.command("exec")
def exec_trade(
    episode_id: str,
    price: float = 0.0,
    size: float = 0.0,
    paper: bool = False,
    size_usdt: Optional[float] = None,
    leverage: Optional[int] = None,
) -> None:
    """Mark an episode as executed.

    Modes:
      manual: provide --price and --size, just journals the entry.
      --paper: place a real market order on Binance Futures testnet using --size-usdt
               (default from PAPER_DEFAULT_SIZE_USDT). Fill price/qty are recorded.
    """
    init_db()
    if paper:
        from .exchange import testnet
        from .config import settings as _s
        with session_scope() as s:
            ep = s.get(Episode, episode_id)
        if not ep:
            console.print("[red]episode not found[/red]"); raise typer.Exit(1)
        if ep.decision not in {"long", "short"}:
            console.print(f"[red]episode decision is {ep.decision!r}, refusing to paper-trade[/red]")
            raise typer.Exit(2)
        notional = size_usdt or _s.paper_default_size_usdt
        lev = leverage or _s.paper_leverage
        console.print(f"[cyan]paper-trade[/cyan]  {ep.symbol}  {ep.decision}  ${notional} @ {lev}x  (testnet)")
        try:
            order = testnet.open_position(ep.symbol, ep.decision, size_usdt=notional, leverage=lev)
        except Exception as e:
            console.print(f"[red]paper exec failed:[/red] {e}"); raise typer.Exit(3)
        HarnessAgent().mark_executed(episode_id, price=order.avg_price, size=order.qty,
                                     extra={"paper": True, "order_id": order.id, "leverage": lev,
                                            "notional_usdt": order.notional_usdt})
        console.print(f"[green]filled[/green]  order_id={order.id}  qty={order.qty}  avg=${order.avg_price}  notional=${order.notional_usdt:.2f}")
        return

    if price <= 0 or size <= 0:
        console.print("[red]manual mode requires --price and --size[/red]"); raise typer.Exit(2)
    HarnessAgent().mark_executed(episode_id, price=price, size=size)
    console.print(f"[green]marked executed[/green]  {episode_id}  @ {price}  size {size}")


@app.command("close")
def close_trade(
    episode_id: str,
    exit_price: float = 0.0,
    reason: str = "manual",
    paper: bool = False,
) -> None:
    """Close an executed episode.

    --paper: closes the live testnet position with a reduce-only market order;
             fill price drives the PnL calculation.
    """
    init_db()
    if paper:
        from .exchange import testnet
        with session_scope() as s:
            ep = s.get(Episode, episode_id)
        if not ep:
            console.print("[red]episode not found[/red]"); raise typer.Exit(1)
        if not ep.executed:
            console.print("[red]episode not executed yet[/red]"); raise typer.Exit(2)
        try:
            order = testnet.close_position(ep.symbol, ep.decision)
        except Exception as e:
            console.print(f"[red]paper close failed:[/red] {e}"); raise typer.Exit(3)
        exit_price = order.avg_price
        console.print(f"[cyan]paper close fill[/cyan]  ${exit_price}  qty={order.qty}")

    if exit_price <= 0:
        console.print("[red]exit_price required[/red]"); raise typer.Exit(2)
    HarnessAgent().close_trade(episode_id, exit_price=exit_price, reason=reason)
    with session_scope() as s:
        ep = s.get(Episode, episode_id)
    if ep and ep.pnl_pct is not None:
        console.print(
            f"[green]closed[/green]  {episode_id}  pnl {ep.pnl_pct:+.2f}%  ({ep.outcome_label})"
        )


@app.command("positions")
def positions() -> None:
    """List all open positions on Binance Futures testnet."""
    from .exchange import testnet
    try:
        acct = testnet.account()
        rows = testnet.all_open_positions()
    except Exception as e:
        console.print(f"[red]{e}[/red]"); raise typer.Exit(1)
    console.print(f"[cyan]testnet balance[/cyan]  total ${acct['total_usdt']:.2f}  free ${acct['free_usdt']:.2f}  used ${acct['used_usdt']:.2f}")
    if not rows:
        console.print("[dim]no open positions[/dim]"); return
    t = Table(title="open positions (testnet)")
    for col in ("symbol", "side", "qty", "entry", "mark", "uPnL"):
        t.add_column(col)
    for p in rows:
        upnl = p["unrealized_pnl"]
        color = "green" if upnl >= 0 else "red"
        t.add_row(p["symbol"], p["side"], f"{p['contracts']:+.6f}", f"{p['entry_price']:.6f}",
                  f"{p['mark_price']:.6f}", f"[{color}]${upnl:+.2f}[/{color}]")
    console.print(t)


@app.command("annotate")
def annotate(episode_id: str, reflection: str = "", lesson: list[str] = typer.Option(None)) -> None:
    """Attach reflection / lessons to an episode for later review."""
    init_db()
    HarnessAgent().annotate(episode_id, reflection=reflection, lessons=lesson or [])
    console.print(f"[green]annotated[/green]  {episode_id}")


@app.command()
def replay(episode_id: str, llm: bool = True, dual: bool = False, verbose: bool = True) -> None:
    """Re-run the policy on an existing episode's snapshot. Creates a NEW episode
    so you can compare rule / llm / dual decisions side-by-side."""
    init_db()
    with session_scope() as s:
        src = s.get(Episode, episode_id)
    if not src:
        console.print("[red]episode not found[/red]")
        raise typer.Exit(1)

    base = dict(src.snapshot or {})
    if dual:
        from .harness.llm_policy import LLMPolicy
        from .harness.dual_policy import DualPolicy
        policy = DualPolicy(llm_policy=LLMPolicy(verbose=verbose))
        console.print("[cyan]DUAL policy[/cyan]")
        agent = HarnessAgent(policy=policy, dedup_hours=0)
    elif llm:
        from .harness.llm_policy import LLMPolicy
        from .config import settings as _s
        policy = LLMPolicy(verbose=verbose)
        console.print(f"[cyan]LLM policy  model={policy.model or _s.llm_model}[/cyan]")
        agent = HarnessAgent(policy=policy, dedup_hours=0)
    else:
        agent = HarnessAgent(dedup_hours=0)

    # Bypass network enrichment by reusing the saved snapshot as-is.
    from .harness import agent as agent_mod
    original = agent_mod.build_snapshot
    agent_mod.build_snapshot = lambda sym, b: dict(b or {})
    try:
        ep = agent.handle_signal(f"replay:{src.trigger}", src.symbol, base)
    finally:
        agent_mod.build_snapshot = original

    if not ep:
        console.print("[yellow](dedup hit somehow)[/yellow]")
        return
    console.print(f"\n[green]new episode[/green]  {ep.id}  {ep.symbol}  -> {ep.decision} (conf {ep.confidence:.2f})")
    if src.decision != ep.decision:
        console.print(f"[bold yellow]DIFF[/bold yellow]  rule={src.decision}({src.confidence:.2f})  vs  llm={ep.decision}({ep.confidence:.2f})")


@app.command()
def run(consensus_only_notify: bool = True) -> None:
    """Run the long-running scheduler: dual-policy scan every N minutes
    + position watch every K seconds (auto SL/TP on testnet positions).
    Ctrl-C to stop. Configure intervals in .env (SCAN_INTERVAL_MINUTES,
    POSITION_WATCH_INTERVAL_SECONDS)."""
    from .scheduler import run as _run
    _run(consensus_only_notify=consensus_only_notify)


@app.command()
def web(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Launch the FastAPI dashboard at http://host:port (default 127.0.0.1:8765)."""
    from .web import run as _run
    console.print(f"[green]starting dashboard[/green]  http://{host}:{port}")
    _run(host=host, port=port)


@app.command()
def doctor() -> None:
    """Check Binance endpoint reachability and DB readiness."""
    import httpx as _h
    init_db()
    console.print(f"[green]db ok[/green]  -> {Episode.__tablename__} ready")
    for url in (
        "https://fapi.binance.com/fapi/v1/ping",
        "https://api.binance.com/api/v3/ping",
        "https://www.binance.com/bapi/composite/v1/public/marketing/symbol/list",
    ):
        try:
            r = _h.get(url, timeout=5)
            console.print(f"[green]{r.status_code}[/green]  {url}")
        except Exception as e:
            console.print(f"[red]FAIL[/red]  {url}  ({type(e).__name__}: {e})")


@app.command("llm-routing")
def llm_routing() -> None:
    """Show effective LLM model routing and configured switchable profiles."""
    from .llm_clients import describe_profiles, describe_routing
    console.rule("effective routing")
    console.print(describe_routing())
    console.rule("configured profiles")
    console.print(describe_profiles())


@app.command("self-test")
def self_test(llm: bool = False, verbose: bool = True, fresh: bool = False) -> None:
    """Inject a fake signal end-to-end (no network) — verifies harness + DB + render.

    With --llm, uses the LLM policy (requires OPENAI_API_KEY + network).
    """
    init_db()
    from .notify.telegram import render_episode

    fake = {
        "symbol": "DEMOUSDT",
        "price": 0.1234,
        "price_chg_24h": 4.2,
        "volume_24h": 8_500_000,
        "prev_funding": 0.0001,
        "curr_funding": -0.00025,
        "oi_change_pct": 14.7,
        "oi_segments": [10e6, 11e6, 12.5e6, 14e6],
        "oi_rising": True,
    }

    # Bypass network-dependent context enrichment by stubbing.
    from .harness import agent as agent_mod

    original = agent_mod.build_snapshot
    agent_mod.build_snapshot = lambda sym, base: {**(base or {}), "coin": sym.replace("USDT", ""), "market_cap_usd": 0, "has_spot": False, "square_posts": 0, "square_views": 0}
    try:
        if llm:
            from .harness.llm_policy import LLMPolicy
            policy = LLMPolicy(verbose=verbose)
            from .config import settings as _s
            console.print(f"[cyan]LLM policy  model={policy.model or _s.llm_model} base={policy.base_url or _s.llm_base_url or 'openai-default'}[/cyan]")
            agent = HarnessAgent(policy=policy, dedup_hours=0 if fresh else 24)
            ep = agent.handle_signal("self_test", "DEMOUSDT", fake)
        else:
            agent = HarnessAgent(dedup_hours=0 if fresh else 24)
            ep = agent.handle_signal("self_test", "DEMOUSDT", fake)
    finally:
        agent_mod.build_snapshot = original

    if not ep:
        console.print("[yellow]dedup hit — already have a recent self_test episode[/yellow]")
        return
    console.print(f"[green]created[/green]  {ep.id}  decision={ep.decision} conf={ep.confidence:.2f}")
    console.rule("rendered telegram message")
    console.print(render_episode(ep))


if __name__ == "__main__":
    app()
