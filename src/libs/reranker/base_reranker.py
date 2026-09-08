"""Reranker 抽象基类与 NoneReranker 回退实现。

Reranker 对检索阶段召回的候选集做二次精排。``NoneReranker`` 保持原顺序，
作为系统默认回退：精排不可用/超时/失败时，直接采用融合阶段的排名。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RerankCandidate:
    """参与重排的一条候选记录。"""

    id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """所有重排后端的抽象基类。"""

    backend: str = "base"

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any = None,
    ) -> list[RerankCandidate]:
        """对候选集重新排序。

        Args:
            query: 用户原始查询。
            candidates: 融合阶段输出的候选集。
            trace: 可选的 TraceContext（阶段 F 落地，目前透传忽略）。

        Returns:
            重新排序后的候选集（不改变候选对象本身，返回新列表）。
        """
        raise NotImplementedError


class NoneReranker(BaseReranker):
    """保持原顺序的回退重排器。"""

    backend = "none"

    def rerank(self, query, candidates, trace=None) -> list[RerankCandidate]:
        return list(candidates)
