"""LLM-backed policy using OpenAI-compatible chat completions with tool-calling.

The model receives the snapshot and may call any of the exposed tools to
gather extra context. After at most `max_steps` ReAct iterations it MUST emit
a final `submit_decision` tool call whose arguments match `DecisionSchema`.

Works with any OpenAI-compatible endpoint:
  - OpenAI:     LLM_BASE_URL="" (default) + OPENAI_API_KEY
  - DAPI:       LLM_BASE_URL="https://dapicloud.com/v1"
  - Ollama:     LLM_BASE_URL="http://localhost:11434/v1", LLM_MODEL="qwen2.5:7b"
  - Anthropic:  via DAPI or any OpenAI-compatible proxy
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field

from ..config import settings
from ..tools import binance_market as bm
from .agent import Decision


# ---------------------------------------------------------------------------
# Tools exposed to the LLM
# ---------------------------------------------------------------------------

TOOL_IMPL: dict[str, Callable[..., Any]] = {
    "get_oi_history": lambda symbol, limit=48: bm.oi_history(symbol, "1h", int(limit)),
    "get_long_short_ratio": lambda symbol, limit=24: bm.long_short_ratio(symbol, "1h", int(limit)),
    "get_square_hashtag": lambda coin: dict(zip(("posts", "views"), bm.square_hashtag(coin))),
    "get_spot_listed": lambda coin: coin in bm.spot_symbols(),
}


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_oi_history",
            "description": "Fetch hourly open-interest history for a Binance USDT perpetual. Returns up to `limit` points with sumOpenInterestValue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "e.g. BTCUSDT"},
                    "limit": {"type": "integer", "default": 48, "minimum": 12, "maximum": 200},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_long_short_ratio",
            "description": "Top-trader long/short account ratio on Binance perps (hourly).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "limit": {"type": "integer", "default": 24},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_square_hashtag",
            "description": "Binance Square attention for a coin: post count and view count of the hashtag.",
            "parameters": {
                "type": "object",
                "properties": {"coin": {"type": "string", "description": "e.g. BTC, not BTCUSDT"}},
                "required": ["coin"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_spot_listed",
            "description": "Return true if the coin has a Binance spot USDT pair (vs perp-only).",
            "parameters": {
                "type": "object",
                "properties": {"coin": {"type": "string"}},
                "required": ["coin"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_decision",
            "description": "Finalize the trading decision. Call this exactly once when you have enough information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["long", "short", "skip", "watch"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string", "description": "Concise reasoning for the decision."},
                    "entry_plan": {
                        "type": "object",
                        "properties": {
                            "side": {"type": "string", "enum": ["long", "short"]},
                            "size_pct": {"type": "number", "description": "Fraction of bankroll, e.g. 0.02 = 2%."},
                            "stop_loss_pct": {"type": "number"},
                            "take_profit_pct": {"type": "array", "items": {"type": "number"}},
                            "timeframe": {"type": "string"},
                        },
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["decision", "confidence", "rationale"],
            },
        },
    },
]


class DecisionSchema(BaseModel):
    decision: str
    confidence: float = Field(ge=0, le=1)
    rationale: str
    entry_plan: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


SYSTEM_PROMPT = """You are a disciplined crypto derivatives trading analyst.

You will receive a market SNAPSHOT describing a symbol where some trigger fired
(e.g. funding-rate flip + OI surge). Your job: decide LONG / SHORT / SKIP / WATCH.

Guidelines:
- Favor SKIP when signals are mixed or volume is thin.
- Negative funding + rising OI + healthy volume → short-squeeze setup (bias long).
- Negative funding + falling OI → shorts already covering → weaker long.
- Overheated positive funding + OI spike → crowded longs → bias short/watch.
- Perp-only micro-caps are higher variance; demand stronger evidence.
- Always produce an entry_plan with concrete SL/TP if you go long/short.
- Be concise. Call tools only when they materially change your view.
- End by calling the `submit_decision` tool exactly once.

If a MEMORY section is provided with past closed trades on this symbol or
pattern, weigh their outcomes heavily: repeat what worked, avoid repeating
what lost. If memory shows a losing streak on a given setup, lean SKIP/WATCH
unless the current snapshot is materially different — and say *why* it's
different in your rationale.
"""


@dataclass
class LLMPolicy:
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    max_steps: int = 4
    verbose: bool = False
    use_memory: bool = True              # inject past closed episodes into prompt
    memory_limit: int = 5
    tools_called: list[dict[str, Any]] = None  # populated per-call for the agent to persist
    similar_episode_ids: list[str] = None      # populated per-call for the agent to persist

    def _log(self, *parts: Any) -> None:
        if self.verbose:
            print(*parts, flush=True)

    def _client(self):
        """Resolve the decision-role client. Explicit constructor args on this
        policy (model/base_url/api_key) still win for ad-hoc experiments; if
        none are set we fall through to the role-based routing.
        """
        if self.model or self.base_url or self.api_key:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError("openai package not installed") from e
            return OpenAI(
                api_key=self.api_key or settings.openai_api_key or "sk-noop",
                base_url=(self.base_url or settings.llm_base_url) or None,
            )
        from ..llm_clients import get_client
        return get_client("decision")

    def _resolved_model(self) -> str:
        if self.model:
            return self.model
        from ..llm_clients import resolve
        return resolve("decision").model

    def __call__(self, snapshot: dict[str, Any]) -> Decision:
        self.tools_called = []
        self.similar_episode_ids = []
        client = self._client()
        model = self._resolved_model()

        # Pull a few closed past episodes for this symbol/pattern and inline
        # them as a compact "memory" block the model can ground its reasoning on.
        memory_block = ""
        if self.use_memory:
            try:
                from .memory import retrieve_similar_episodes, summarize_for_prompt
                sym = snapshot.get("symbol") or ""
                past = retrieve_similar_episodes(sym, snapshot, limit=self.memory_limit)
                self.similar_episode_ids = [p.id for p in past]
                memory_block = summarize_for_prompt(past)
            except Exception as e:
                self._log(f"[memory] retrieval failed: {e}")
                memory_block = "(memory retrieval failed)"

        user_content = "SNAPSHOT:\n" + json.dumps(snapshot, default=str, indent=2)
        if memory_block:
            user_content += "\n\nMEMORY:\n" + memory_block

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        final: DecisionSchema | None = None

        for step in range(self.max_steps):
            self._log(f"\n========== ReAct step {step} ==========")
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if msg.content:
                self._log(f"[thought]\n{msg.content}")

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                self._log("[no tool call] -> nudging to finalize")
                messages.append({
                    "role": "user",
                    "content": "Please finalize by calling `submit_decision`.",
                })
                continue

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}

                self._log(f"[tool_call] {name}({json.dumps(args, ensure_ascii=False)})")

                if name == "submit_decision":
                    try:
                        final = DecisionSchema(**args)
                    except Exception as e:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"schema_error: {e}. Please re-submit with corrected fields.",
                        })
                        continue
                    self.tools_called.append({"name": name, "args": args, "step": step})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "ok",
                    })
                    break

                impl = TOOL_IMPL.get(name)
                if impl is None:
                    result = {"error": f"unknown tool {name}"}
                else:
                    try:
                        result = impl(**args)
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {e}"}

                # Truncate very long tool outputs to keep context small
                content = json.dumps(result, default=str)
                if len(content) > 4000:
                    content = content[:4000] + " ...[truncated]"

                self.tools_called.append({"name": name, "args": args, "step": step, "result_len": len(content)})
                preview = content if len(content) <= 300 else content[:300] + "...[truncated]"
                self._log(f"[tool_result] {preview}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                })

            if final is not None:
                break

        if final is None:
            # Fallback: conservative skip with trace for debugging
            return Decision(
                decision="skip",
                confidence=0.0,
                rationale=f"LLM did not submit a decision within {self.max_steps} steps.",
                tags=["llm_timeout"],
            )

        return Decision(
            decision=final.decision,
            confidence=final.confidence,
            rationale=final.rationale,
            entry_plan=final.entry_plan,
            tags=final.tags,
        )
