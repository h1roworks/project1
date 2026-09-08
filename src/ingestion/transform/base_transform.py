"""Transform 抽象基类。

定义所有 Transform（增强处理）必须实现的最小接口。上层业务只依赖
``BaseTransform``，不关心具体是 ChunkRefiner / MetadataEnricher / ImageCaptioner。

工程契约（对齐 DEV_SPEC 3.1.1 "Transform 工程特性"）：
- **原子化**：``transform_many`` 逐 chunk 独立处理，单个失败不拖垮整批；
- **幂等**：``transform`` 对同一输入重复执行必须产出相同结果；
- **失败降级**：处理失败的 chunk 原样返回，并在 ``metadata.transform_errors`` 记录原因，
  保证 LLM 等外部依赖失败时不阻塞摄取流水线。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any

from core.types import Chunk


class BaseTransform(ABC):
    """所有 Transform 的抽象基类。

    ``transform`` 处理单个 Chunk 并返回新 Chunk（不修改入参）；``transform_many``
    提供批处理编排：逐 chunk 原子调用 transform，失败时降级为原样返回 + 记录错误，
    从而支持针对特定 chunk 的独立重试与增量更新。
    """

    name: str = "base"

    @abstractmethod
    def transform(self, chunk: Chunk, trace: Any = None) -> Chunk:
        """增强单个 Chunk。实现必须幂等，且不得修改入参 chunk。"""
        raise NotImplementedError

    def transform_many(self, chunks: list[Chunk], trace: Any = None) -> list[Chunk]:
        """批处理：对每个 chunk 原子执行 transform，失败降级不阻塞。

        返回列表与输入一一对应（保序）。单 chunk 抛异常时，返回原样副本并在
        ``metadata.transform_errors`` 追加 ``"{name}: {error}"`` 记录。
        """
        results: list[Chunk] = []
        for chunk in chunks:
            try:
                results.append(self.transform(chunk, trace=trace))
            except Exception as exc:  # noqa: BLE001 - 降级兜底：记录后继续处理
                results.append(_mark_transform_error(chunk, self.name, exc))
        return results


def _mark_transform_error(chunk: Chunk, transform_name: str, exc: Exception) -> Chunk:
    """把处理失败的 chunk 原样返回，并在 metadata 中追加一条错误记录。"""
    metadata = dict(chunk.metadata)
    errors = list(metadata.get("transform_errors", []))
    errors.append(f"{transform_name}: {exc}")
    metadata["transform_errors"] = errors
    return replace(chunk, metadata=metadata)
