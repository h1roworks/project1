"""Evaluator 抽象基类。

定义评估器的统一接口：输入一次检索的 query + 召回结果 + 黄金答案，
输出标准化的指标字典。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """所有评估器（Ragas/Custom/DeepEval 等）的抽象基类。"""

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any = None,
    ) -> dict[str, float]:
        """评估一次检索的质量。

        Args:
            query: 用户查询。
            retrieved_ids: 系统召回的结果 id（按排名顺序）。
            golden_ids: 黄金标准中预期的 id 集合。
            trace: 可选的 TraceContext（阶段 F 落地，目前透传忽略）。

        Returns:
            指标名 -> 分数 的字典（如 {"hit_rate": 1.0, "mrr": 0.5}）。
        """
        raise NotImplementedError
