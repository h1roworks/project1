"""Splitter 工厂：按配置中的 strategy 创建对应的 BaseSplitter 实现。"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from libs.splitter.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from core.settings import SplitterSettings

_BUILTIN_STRATEGIES: dict[str, str] = {
    "recursive": "libs.splitter.recursive_splitter",
    "semantic": "libs.splitter.semantic_splitter",
    "fixed": "libs.splitter.fixed_length_splitter",
}


class SplitterFactory:
    """根据 settings.splitter.strategy 路由到具体切分实现。"""

    _registry: dict[str, type[BaseSplitter]] = {}

    @classmethod
    def register(cls, strategy: str, impl: type[BaseSplitter]) -> None:
        cls._registry[strategy] = impl

    @classmethod
    def _ensure_builtin(cls, strategy: str) -> None:
        if strategy in cls._registry or strategy not in _BUILTIN_STRATEGIES:
            return
        importlib.import_module(_BUILTIN_STRATEGIES[strategy])

    @classmethod
    def create(cls, settings: Any) -> BaseSplitter:
        strategy = getattr(settings, "strategy", "")
        if not strategy:
            raise ValueError("Splitter 策略未配置（settings.splitter.strategy 为空）")

        cls._ensure_builtin(strategy)
        impl = cls._registry.get(strategy)
        if impl is None:
            raise ValueError(
                f"未知的 Splitter 策略: '{strategy}'。可选: {sorted(cls._registry)}"
            )
        return impl(settings)
