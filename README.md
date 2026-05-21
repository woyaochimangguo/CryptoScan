# cryptoscan

Personal harness-agent for crypto signal scanning, decision journaling, and post-trade reflection.

## Idea

Every signal → snapshot → decision → execution → outcome → reflection cycle is captured as one **Episode**.
The agent is a thin loop with swappable **policies**: P1 ships rule-based policies; P2 will plug in an LLM
ReAct policy that calls the same tools.

```
signal trigger → build snapshot → policy decides → persist Episode → TG push
                                                                ↓
                              you execute manually → /exec → /close → annotate → review
```

## Layout

```
cryptoscan/
├── tools/         tool registry; Binance market + OI/funding scanner
├── strategies/    strategy plugins + registry
├── harness/       Agent main loop, snapshot builder, policies
├── notify/        Telegram renderer
├── models.py      Episode SQLModel
├── db.py          SQLite engine + session scope
├── config.py      env-driven settings
└── cli.py         typer entry point
```

## Setup

```bash
# in repo root
python -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env
# fill TG_BOT_TOKEN / TG_CHAT_ID if you want push (optional, falls back to stdout)

cryptoscan init                # create data/cryptoscan.db
```

## Strategies

The scanner now runs through a small strategy registry instead of hard-coding
one setup. The first registered strategy is:

```bash
cryptoscan strategies
cryptoscan scan --strategy oi_funding_flip --dual
```

To add a strategy, create a module under `cryptoscan/strategies/` that exposes
the same shape as `OiFundingFlipStrategy`:

```python
class MyStrategy:
    id = "my_strategy"
    name = "My strategy"
    version = "1.0.0"
    trigger = "my_trigger"
    default_policy_id = "dual"

    def scan(self, **kwargs) -> list[Signal]: ...
    def warm(self, signals: list[Signal]) -> None: ...
    def policy(self, mode: str = "dual") -> Policy: ...
```

Then register it in `cryptoscan/strategies/registry.py` and enable it via:

```env
ENABLED_STRATEGIES=oi_funding_flip,my_strategy
```

Each Episode persists `strategy_id`, `strategy_version`, `policy_id`,
`model_profile`, and `risk_profile`, so dashboard stats can compare strategies
instead of mixing every signal into one bucket.

## Daily use

```bash
# Run scanner (cron every minute, or just on demand)
cryptoscan scan

# Review last 24h of episodes
cryptoscan review

# Drill into one
cryptoscan show <episode_id>

# Manual lifecycle (because P1 doesn't auto-trade)
cryptoscan exec <episode_id> <entry_price> <size>
cryptoscan close <episode_id> <exit_price> --reason tp1
cryptoscan annotate <episode_id> --reflection "shorts capitulated as predicted" --lesson "FR flip + OI rising = strong"
```

## Runtime split

`cryptoscan run` and `cryptoscan web` are separate processes:

- `cryptoscan run` owns scheduled scans, paper-position watching, auto execution, and runtime cache refresh.
- `cryptoscan web` owns only the dashboard/API. Its read endpoints use SQLite snapshots written by the scheduler, so the page stays usable even if Binance, the scanner, or the LLM path is slow or down.
- Manual trading actions from the dashboard (`/api/exec/*`, `/api/close/*`) still call Binance testnet live because they perform side effects.

Both processes share `data/cryptoscan.db`; SQLite runs in WAL mode so scheduler writes do not block dashboard reads under normal load.

### Cron

```cron
* * * * * cd /path/to/cryptoscan && /path/to/.venv/bin/cryptoscan scan >> data/scan.log 2>&1
```

## Roadmap

- **P1 (MVP)** — OI+funding scanner, rule policy, episode journal, TG push, manual lifecycle
- **P2 (current)** — LLM policy (ReAct via OpenAI-compatible tool-calling), structured decisions via pydantic
  ```bash
  # .env: OPENAI_API_KEY=sk-...   LLM_MODEL=gpt-4o-mini   LLM_BASE_URL=  (or DAPI/Ollama URL)
  cryptoscan self-test --llm     # smoke-test LLM policy with a fake snapshot
  cryptoscan scan --llm          # use LLM for real signals
  ```
  Opening decisions use the `decision` LLM route. Check it with:
  ```bash
  cryptoscan llm-routing
  ```
  For quick model switching, define profiles in `.env`:
  ```env
  LLM_DECISION_PROFILE=minimax
  LLM_PROFILE_MINIMAX_MODEL=coding-minimax-m2.7-free
  LLM_PROFILE_MINIMAX_BASE_URL=https://aihubmix.com/v1
  LLM_PROFILE_MINIMAX_API_KEY=...

  LLM_PROFILE_DEEPSEEK_MODEL=deepseek-chat
  LLM_PROFILE_DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
  LLM_PROFILE_DEEPSEEK_API_KEY=...
  ```
  Then switch opening-decision models by changing only `LLM_DECISION_PROFILE`.
  The LLM may call: `get_oi_history`, `get_long_short_ratio`, `get_square_hashtag`,
  `get_spot_listed`, then **must** finalize via `submit_decision`. All tool calls and
  the final rationale are persisted to the Episode for replay/review.
- **P3** — Vector memory: retrieve similar past episodes at decision time
- **P4** — PnL watcher (ccxt websocket) auto-fills outcome on SL/TP
- **P5** — Reflection engine: weekly LLM reviews → lessons → fed back into prompt
- **P6** — More tools: Binance Square hashtag stream, CryptoPanic, DexScreener, on-chain
