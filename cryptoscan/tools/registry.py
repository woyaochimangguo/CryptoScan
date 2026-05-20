from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    @property
    def signature(self) -> str:
        return f"{self.name}{inspect.signature(self.func)}"


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str = "") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def deco(func: Callable[..., Any]) -> Callable[..., Any]:
            self._tools[name] = Tool(name=name, description=description or (func.__doc__ or "").strip(), func=func)
            return func
        return deco

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def all(self) -> list[Tool]:
        return list(self._tools.values())


registry = Registry()
tool = registry.register
