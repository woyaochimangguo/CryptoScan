"""Bridge stdio MCP servers into the cryptoscan tool registry.

The MCP Python SDK is async. We expose a sync facade so the rest of the codebase
(LLMPolicy, CLI) keeps working without an event loop. Each bridge owns a long-
lived background thread running an asyncio loop with a connected ClientSession.
Tools discovered via `list_tools` are auto-registered into the global @tool
registry under names like `news.get_high_score_news`.
"""
from __future__ import annotations

import asyncio
import atexit
import json
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .tools.registry import Tool, registry


@dataclass
class MCPBridge:
    name: str                       # short prefix, e.g. "news"
    command: str                    # e.g. "uv"
    args: list[str]                 # e.g. ["--directory", "...", "run", "opennews-mcp"]
    env: dict[str, str]             # extra env vars (token)

    _loop: asyncio.AbstractEventLoop | None = None
    _thread: threading.Thread | None = None
    _session: ClientSession | None = None
    _ready: threading.Event | None = None
    _stop: asyncio.Event | None = None
    _tools_meta: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._ready = threading.Event()
        err: dict[str, Any] = {}

        def _runner() -> None:
            try:
                asyncio.run(self._serve())
            except Exception as e:
                err["e"] = e
                self._ready.set()

        self._thread = threading.Thread(target=_runner, name=f"mcp-{self.name}", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=20):
            raise RuntimeError(f"MCP bridge {self.name} did not become ready in 20s")
        if "e" in err:
            raise err["e"]
        atexit.register(self.stop)

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                self._session = session
                self._tools_meta = [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "inputSchema": t.inputSchema or {"type": "object", "properties": {}},
                    }
                    for t in listed.tools
                ]
                self._ready.set()  # type: ignore[union-attr]
                await self._stop.wait()

    def stop(self) -> None:
        if self._loop and self._stop and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3)

    # ------------------------------------------------------------------
    # Sync call facade
    # ------------------------------------------------------------------

    def call(self, tool_name: str, **arguments: Any) -> Any:
        if not self._session or not self._loop:
            raise RuntimeError(f"bridge {self.name} not started")

        async def _invoke() -> Any:
            result = await self._session.call_tool(tool_name, arguments=arguments)
            # Concatenate text blocks; many MCP tools return JSON-as-text.
            parts: list[str] = []
            for c in result.content:
                if hasattr(c, "text") and c.text is not None:
                    parts.append(c.text)
            blob = "\n".join(parts) if parts else ""
            if blob:
                try:
                    return json.loads(blob)
                except Exception:
                    return blob
            return None

        fut: Future[Any] = asyncio.run_coroutine_threadsafe(_invoke(), self._loop)
        return fut.result(timeout=60)

    # ------------------------------------------------------------------
    # Auto-register into cryptoscan tool registry
    # ------------------------------------------------------------------

    def register_all(self) -> list[str]:
        if self._tools_meta is None:
            raise RuntimeError(f"bridge {self.name} has no tools list (not started?)")
        registered: list[str] = []
        for meta in self._tools_meta:
            local_name = f"{self.name}.{meta['name']}"
            description = meta["description"]
            schema = meta["inputSchema"]
            remote_name = meta["name"]

            def _make_caller(rn: str):
                def _caller(**kwargs: Any) -> Any:
                    return self.call(rn, **kwargs)
                _caller.__name__ = rn
                _caller.__doc__ = description
                return _caller

            tool_fn = _make_caller(remote_name)
            registry._tools[local_name] = Tool(name=local_name, description=description, func=tool_fn)
            # Stash schema for LLMPolicy dynamic exposure
            registry._tools[local_name].input_schema = schema  # type: ignore[attr-defined]
            registered.append(local_name)
        return registered


# ---------------------------------------------------------------------------
# Convenience: build bridges from cryptoscan settings
# ---------------------------------------------------------------------------

def build_bridges_from_settings() -> list[MCPBridge]:
    from .config import settings

    bridges: list[MCPBridge] = []
    if settings.opennews_token and settings.opennews_mcp_dir:
        bridges.append(MCPBridge(
            name="news",
            command="uv",
            args=["--directory", settings.opennews_mcp_dir, "run", "opennews-mcp"],
            env={"OPENNEWS_TOKEN": settings.opennews_token},
        ))
    if settings.twitter_token and settings.opentwitter_mcp_dir:
        bridges.append(MCPBridge(
            name="twitter",
            command="uv",
            args=["--directory", settings.opentwitter_mcp_dir, "run", "opentwitter-mcp"],
            env={"TWITTER_TOKEN": settings.twitter_token},
        ))
    return bridges


_started: list[MCPBridge] = []


def start_all() -> list[MCPBridge]:
    """Idempotent: start every configured bridge and register tools."""
    global _started
    if _started:
        return _started
    bridges = build_bridges_from_settings()
    for b in bridges:
        b.start()
        b.register_all()
    _started = bridges
    return bridges
