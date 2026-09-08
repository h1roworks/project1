"""Evaluator 工厂：按配置中的 provider 创建对应的 BaseEvaluator 实现。"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator

if TYPE_CHECKING:
    from core.settings import EvaluationSettings

# 内置 provider → 实现模块路径。ragas 依赖较重，仅在使用时才导入（阶段 H 落地）。
_BUILTIN_PROVIDERS: dict[str, str] = {
    "custom": "libs.evaluator.custom_evaluator",
    "ragas": "libs.evaluator.ragas_evaluator",
}


class EvaluatorFactory:
    """根据 settings.evaluation.provider 路由到具体评估实现。"""

    _registry: dict[str, type[BaseEvaluator]] = {
        "custom": CustomEvaluator,
    }

    @classmethod
    def register(cls, provider: str, impl: type[BaseEvaluator]) -> None:
        cls._registry[provider] = impl

    @classmethod
    def _ensure_builtin(cls, provider: str) -> None:
        if provider in cls._registry or provider not in _BUILTIN_PROVIDERS:
            return
        importlib.import_module(_BUILTIN_PROVIDERS[provider])

    @classmethod
    def create(cls, settings: Any) -> BaseEvaluator:
        provider = getattr(settings, "provider", "")
        if not provider:
            raise ValueError("Evaluator provider 未配置（settings.evaluation.provider 为空）")

        cls._ensure_builtin(provider)
        impl = cls._registry.get(provider)
        if impl is None:
            raise ValueError(
                f"未知的 Evaluator provider: '{provider}'。可选: {sorted(cls._registry)}"
            )
        return impl()
