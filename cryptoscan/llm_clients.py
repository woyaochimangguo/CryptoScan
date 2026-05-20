"""Centralized LLM client factory with role-based routing.

Two roles today:
  * "decision"   -> strong model, used by LLMPolicy
  * "reflection" -> cheap model, used by auto_reflect

Each role can independently override api_key / model / base_url via
LLM_DECISION_* and LLM_REFLECTION_* env keys. Anything left empty falls back to
the role-agnostic defaults (LLM_MODEL / LLM_BASE_URL / OPENAI_API_KEY).

Returns plain OpenAI SDK clients; all providers we talk to are
OpenAI-compatible (OpenAI, DAPI, DeepSeek, Ollama, Together, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .config import settings

Role = Literal["decision", "reflection"]


@dataclass
class ResolvedLLMConfig:
    role: Role
    model: str
    base_url: str | None  # None means OpenAI default
    api_key: str


def resolve(role: Role) -> ResolvedLLMConfig:
    """Resolve the effective (model, base_url, api_key) for a role.

    Precedence per field: role-specific env > global env > package default.
    """
    defaults = (
        settings.llm_model,
        settings.llm_base_url,
        settings.openai_api_key,
    )
    if role == "decision":
        override = (
            settings.llm_decision_model,
            settings.llm_decision_base_url,
            settings.llm_decision_api_key,
        )
    elif role == "reflection":
        override = (
            settings.llm_reflection_model,
            settings.llm_reflection_base_url,
            settings.llm_reflection_api_key,
        )
    else:
        raise ValueError(f"unknown LLM role: {role!r}")

    model = override[0] or defaults[0] or "gpt-4o-mini"
    base_url = override[1] or defaults[1] or ""
    api_key = override[2] or defaults[2] or "sk-noop"

    return ResolvedLLMConfig(
        role=role,
        model=model,
        base_url=base_url or None,
        api_key=api_key,
    )


def get_client(role: Role, *, timeout_sec: float | None = None):
    """Return a configured OpenAI SDK client for the given role.

    Usage:
        client = get_client("decision")
        resp = client.chat.completions.create(model=..., messages=...)

    The model name can be read back from `resolve(role).model` (we pass it as
    an attribute on the returned object for convenience).
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai package not installed; run `pip install openai`") from e

    cfg = resolve(role)
    kwargs: dict = {"api_key": cfg.api_key, "base_url": cfg.base_url}
    if timeout_sec is not None:
        kwargs["timeout"] = timeout_sec
    client = OpenAI(**kwargs)
    # attach the resolved model so callers don't need to re-resolve
    client._cryptoscan_model = cfg.model  # type: ignore[attr-defined]
    client._cryptoscan_role = cfg.role    # type: ignore[attr-defined]
    return client


def describe_routing() -> str:
    """Human-readable summary of which model each role resolves to — used by
    CLI/scheduler startup logs so the user immediately sees the routing."""
    lines = []
    for role in ("decision", "reflection"):
        c = resolve(role)  # type: ignore[arg-type]
        host = c.base_url or "api.openai.com"
        lines.append(f"  {role:>10s}  →  {c.model}  @  {host}")
    return "\n".join(lines)
