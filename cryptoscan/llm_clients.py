"""Centralized LLM client factory with role/profile-based routing.

Two roles today:
  * "decision"   -> strong model, used by LLMPolicy
  * "reflection" -> cheap model, used by auto_reflect

Routing can be changed in two ways:
  * direct role env: LLM_DECISION_MODEL / LLM_DECISION_BASE_URL / ...
  * named profiles: LLM_DECISION_PROFILE=deepseek, then
    LLM_PROFILE_DEEPSEEK_MODEL / LLM_PROFILE_DEEPSEEK_BASE_URL / ...

Direct role env wins over profiles. Anything left empty falls back to the
role-agnostic defaults (LLM_MODEL / LLM_BASE_URL / OPENAI_API_KEY).

Returns plain OpenAI SDK clients; all providers we talk to are
OpenAI-compatible (OpenAI, DAPI, DeepSeek, Ollama, Together, etc.).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from .config import settings

Role = Literal["decision", "reflection"]


@dataclass
class ResolvedLLMConfig:
    role: Role
    model: str
    base_url: str | None  # None means OpenAI default
    api_key: str
    profile: str | None = None


def _profile_env_name(profile: str) -> str:
    """Normalize a user-facing profile name into an env var suffix."""
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", profile.strip()).strip("_").upper()
    return suffix


@lru_cache(maxsize=1)
def _dotenv_values() -> dict[str, str]:
    try:
        from dotenv import dotenv_values
    except Exception:
        return {}
    return {k: str(v) for k, v in dotenv_values(".env").items() if k and v is not None}


def _env(key: str) -> str:
    return os.getenv(key) or _dotenv_values().get(key, "")


def _profile_values(profile: str) -> tuple[str, str, str]:
    suffix = _profile_env_name(profile)
    if not suffix:
        return "", "", ""
    return (
        _env(f"LLM_PROFILE_{suffix}_MODEL"),
        _env(f"LLM_PROFILE_{suffix}_BASE_URL"),
        _env(f"LLM_PROFILE_{suffix}_API_KEY"),
    )


def available_profiles() -> dict[str, dict[str, str]]:
    """Return configured LLM profiles without exposing API keys."""
    out: dict[str, dict[str, str]] = {}
    pattern = re.compile(r"^LLM_PROFILE_([A-Z0-9_]+)_(MODEL|BASE_URL|API_KEY)$")
    env_items = {**_dotenv_values(), **os.environ}
    for key, value in env_items.items():
        m = pattern.match(key)
        if not m:
            continue
        profile, field = m.groups()
        row = out.setdefault(profile.lower(), {"model": "", "base_url": "", "has_api_key": "false"})
        if field == "MODEL":
            row["model"] = value
        elif field == "BASE_URL":
            row["base_url"] = value
        elif field == "API_KEY":
            row["has_api_key"] = "true" if value else "false"
    return dict(sorted(out.items()))


def resolve(role: Role) -> ResolvedLLMConfig:
    """Resolve the effective (model, base_url, api_key) for a role.

    Precedence per field:
      role-specific env > role profile > global profile > global env > package default.
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
        profile = settings.llm_decision_profile or settings.llm_profile
    elif role == "reflection":
        override = (
            settings.llm_reflection_model,
            settings.llm_reflection_base_url,
            settings.llm_reflection_api_key,
        )
        profile = settings.llm_reflection_profile or settings.llm_profile
    else:
        raise ValueError(f"unknown LLM role: {role!r}")

    profile_override = _profile_values(profile) if profile else ("", "", "")

    model = override[0] or profile_override[0] or defaults[0] or "gpt-4o-mini"
    base_url = override[1] or profile_override[1] or defaults[1] or ""
    api_key = override[2] or profile_override[2] or defaults[2] or "sk-noop"

    return ResolvedLLMConfig(
        role=role,
        model=model,
        base_url=base_url or None,
        api_key=api_key,
        profile=profile or None,
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
        profile = f"  profile={c.profile}" if c.profile else ""
        lines.append(f"  {role:>10s}  →  {c.model}  @  {host}{profile}")
    return "\n".join(lines)


def describe_profiles() -> str:
    profiles = available_profiles()
    if not profiles:
        return "(no LLM_PROFILE_* profiles configured)"
    lines = []
    for name, cfg in profiles.items():
        model = cfg.get("model") or "(model not set)"
        host = cfg.get("base_url") or "api.openai.com/default"
        key = "key=yes" if cfg.get("has_api_key") == "true" else "key=no"
        lines.append(f"  {name:<16s}  {model}  @  {host}  {key}")
    return "\n".join(lines)


def is_configured(role: Role) -> bool:
    cfg = resolve(role)
    return bool(cfg.api_key and cfg.api_key != "sk-noop") or bool(cfg.base_url)
