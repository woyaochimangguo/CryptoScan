from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Telegram
    tg_bot_token: str = ""
    tg_chat_id: str = ""

    # Storage
    cryptoscan_db_path: str = "./data/cryptoscan.db"
    cryptoscan_data_dir: str = "./data"

    # Scanner
    oi_min_change_pct: float = 8.0
    oi_dedup_hours: int = 24
    min_volume_usdt: float = 1_000_000

    # LLM — default (role-agnostic) settings.
    # Individual roles can override any of {api_key, model, base_url} via the
    # LLM_DECISION_* and LLM_REFLECTION_* keys below. Anything left empty falls
    # back to these defaults. Keeping all three keys separate lets you route
    # expensive decisions to a strong model and cheap reflections to a mini one.
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = ""
    llm_profile: str = ""  # optional default profile name for all LLM roles

    # Decision-role LLM (gets the full ReAct + tools + memory context)
    llm_decision_model: str = ""
    llm_decision_base_url: str = ""
    llm_decision_api_key: str = ""
    llm_decision_profile: str = ""

    # Reflection-role LLM (simple JSON post-mortem; safe to use a cheap model)
    llm_reflection_model: str = ""
    llm_reflection_base_url: str = ""
    llm_reflection_api_key: str = ""
    llm_reflection_profile: str = ""

    # 6551 MCP servers
    opennews_token: str = ""
    twitter_token: str = ""
    opennews_mcp_dir: str = ""
    opentwitter_mcp_dir: str = ""

    # Binance Futures Testnet (paper trading)
    binance_testnet_key: str = ""
    binance_testnet_secret: str = ""
    paper_default_size_usdt: float = 100.0   # default notional per paper trade
    paper_leverage: int = 5                   # default isolated leverage

    # Scheduler
    scan_interval_minutes: int = 5
    position_watch_interval_seconds: int = 60

    # Auto-execute (paper) — when enabled, scheduler will auto-open paper positions
    # for high-confidence consensus episodes within the safety guardrails below.
    auto_execute_enabled: bool = False
    auto_execute_notional_usdt: float = 20.0
    auto_execute_max_concurrent: int = 3
    auto_execute_min_free_usdt: float = 50.0
    auto_execute_sl_pct: float = 1.5            # absolute %, becomes -1.5
    auto_execute_tp_pcts: str = "2.5,5.0"        # comma-separated absolute %
    auto_execute_max_age_minutes: int = 10       # only act on fresh episodes
    auto_execute_interval_seconds: int = 30      # poll cadence

    # Auto-reflection (LLM post-mortem for closed episodes)
    auto_reflect_interval_seconds: int = 120

    @property
    def data_dir(self) -> Path:
        p = Path(self.cryptoscan_data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        p = Path(self.cryptoscan_db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
