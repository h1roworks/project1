"""Embedding 抽象基类。

定义所有 Embedding provider 必须实现的最小接口，支持批量向量化。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbedding(ABC):
    """所有 Embedding provider 的抽象基类。

    ``embed`` 接收一批文本，返回等长的向量列表（每个向量为 float 列表）。
    """

    provider: str = "base"

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        trace: Any = None,
    ) -> list[list[float]]:
        """批量向量化。

        Args:
            texts: 待向量化的文本列表。
            trace: 可选的 TraceContext（阶段 F 落地，目前透传忽略）。

        Returns:
            与 ``texts`` 等长的向量列表，每条向量维度由具体模型决定。
        """
        raise NotImplementedError
