"""Reranker 工厂：按配置中的 provider 创建对应的 BaseReranker 实现。

- provider=none（或 enabled=false）：返回 NoneReranker，保持原顺序。
- provider=cross_encoder / llm：路由到对应实现（B7.7 / B7.8）。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from libs.reranker.base_reranker import BaseReranker, NoneReranker

if TYPE_CHECKING:
    from core.settings import RerankSettings

_BUILTIN_BACKENDS: dict[str, str] = {
    "cross_encoder": "libs.reranker.cross_encoder_reranker",
    "llm": "libs.reranker.llm_reranker",
}


class RerankerFactory:
    """根据 settings.rerank 路由到具体重排实现。"""

    _registry: dict[str, type[BaseReranker]] = {}

    @classmethod
    def register(cls, backend: str, impl: type[BaseReranker]) -> None:
        cls._registry[backend] = impl

    @classmethod
    def _ensure_builtin(cls, backend: str) -> None:
        if backend in cls._registry or backend not in _BUILTIN_BACKENDS:
            return
        importlib.import_module(_BUILTIN_BACKENDS[backend])

    @classmethod
    def create(cls, settings: Any) -> BaseReranker:
        enabled = getattr(settings, "enabled", True)
        provider = getattr(settings, "provider", "none")

        # 未启用或 provider 为 none：使用保持原顺序的回退实现
        if not enabled or provider in ("", "none"):
            return NoneReranker()

        cls._ensure_builtin(provider)
        impl = cls._registry.get(provider)
        if impl is None:
            raise ValueError(
                f"未知的 Reranker backend: '{provider}'。可选: {sorted(cls._registry) + ['none']}"
            )
        return impl(settings)
