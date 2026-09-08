"""LLM 工厂：按配置中的 provider 创建对应的 BaseLLM 实现。

通过注册表 + 惰性加载内置实现，实现"改配置不改代码"的 provider 切换。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from libs.llm.base_llm import BaseLLM

if TYPE_CHECKING:
    from core.settings import LLMSettings

# 内置 provider → 实现模块路径。模块在首次使用时才 import，避免硬依赖。
_BUILTIN_PROVIDERS: dict[str, str] = {
    "openai": "libs.llm.openai_llm",
    "azure": "libs.llm.azure_llm",
    "deepseek": "libs.llm.deepseek_llm",
    "ollama": "libs.llm.ollama_llm",
    "dashscope": "libs.llm.dashscope_llm",
}

class LLMFactory:
    """根据 settings.llm.provider 路由到具体 LLM 实现。"""

    _registry: dict[str, type[BaseLLM]] = {}

    @classmethod
    def register(cls, provider: str, impl: type[BaseLLM]) -> None:
        """注册 provider 对应的实现类（测试可用 Fake 注册）。"""
        cls._registry[provider] = impl

    @classmethod
    def _ensure_builtin(cls, provider: str) -> None:
        """首次使用时加载内置 provider 模块（模块导入时自行注册）。"""
        if provider in cls._registry or provider not in _BUILTIN_PROVIDERS:
            return
        importlib.import_module(_BUILTIN_PROVIDERS[provider])

    @classmethod
    def create(cls, settings: Any) -> BaseLLM:
        """根据配置创建 LLM 实例。

        Args:
            settings: 配置对象，至少包含 ``provider`` 字段（通常是 LLMSettings）。

        Returns:
            BaseLLM 实现实例。

        Raises:
            ValueError: provider 为空或未注册时抛出可读错误。
        """
        provider = getattr(settings, "provider", "")
        if not provider:
            raise ValueError("LLM provider 未配置（settings.llm.provider 为空）")

        cls._ensure_builtin(provider)
        impl = cls._registry.get(provider)
        if impl is None:
            raise ValueError(
                f"未知的 LLM provider: '{provider}'。可选: {sorted(cls._registry)}"
            )
        return impl(settings)

