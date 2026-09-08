"""VectorStore 工厂：按配置中的 provider 创建对应的 BaseVectorStore 实现。"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from core.settings import VectorStoreSettings

_BUILTIN_PROVIDERS: dict[str, str] = {
    "chroma": "libs.vector_store.chroma_store",
}


class VectorStoreFactory:
    """根据 settings.vector_store.provider 路由到具体向量库实现。"""

    _registry: dict[str, type[BaseVectorStore]] = {}

    @classmethod
    def register(cls, provider: str, impl: type[BaseVectorStore]) -> None:
        cls._registry[provider] = impl

    @classmethod
    def _ensure_builtin(cls, provider: str) -> None:
        if provider in cls._registry or provider not in _BUILTIN_PROVIDERS:
            return
        importlib.import_module(_BUILTIN_PROVIDERS[provider])

    @classmethod
    def create(cls, settings: Any) -> BaseVectorStore:
        provider = getattr(settings, "provider", "")
        if not provider:
            raise ValueError(
                "VectorStore provider 未配置（settings.vector_store.provider 为空）"
            )

        cls._ensure_builtin(provider)
        impl = cls._registry.get(provider)
        if impl is None:
            raise ValueError(
                f"未知的 VectorStore provider: '{provider}'。可选: {sorted(cls._registry)}"
            )
        return impl(settings)
