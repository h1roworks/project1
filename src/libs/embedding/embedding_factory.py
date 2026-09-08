"""Embedding 工厂：按配置中的 provider 创建对应的 BaseEmbedding 实现。"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from core.settings import EmbeddingSettings

_BUILTIN_PROVIDERS: dict[str, str] = {
    "openai": "libs.embedding.openai_embedding",
    "azure": "libs.embedding.azure_embedding",
    "ollama": "libs.embedding.ollama_embedding",
    "dashscope": "libs.embedding.dashscope_embedding",
}


class EmbeddingFactory:
    """根据 settings.embedding.provider 路由到具体 Embedding 实现。"""

    _registry: dict[str, type[BaseEmbedding]] = {}

    @classmethod
    def register(cls, provider: str, impl: type[BaseEmbedding]) -> None:
        cls._registry[provider] = impl

    @classmethod
    def _ensure_builtin(cls, provider: str) -> None:
        if provider in cls._registry or provider not in _BUILTIN_PROVIDERS:
            return
        importlib.import_module(_BUILTIN_PROVIDERS[provider])

    @classmethod
    def create(cls, settings: Any) -> BaseEmbedding:
        provider = getattr(settings, "provider", "")
        if not provider:
            raise ValueError("Embedding provider 未配置（settings.embedding.provider 为空）")

        cls._ensure_builtin(provider)
        impl = cls._registry.get(provider)
        if impl is None:
            raise ValueError(
                f"未知的 Embedding provider: '{provider}'。可选: {sorted(cls._registry)}"
            )
        return impl(settings)
