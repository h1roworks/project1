"""VectorStore 抽象基类。

定义向量存储后端（Chroma/Qdrant/Pinecone 等）必须实现的最小契约。
上层业务代码只依赖 BaseVectorStore，不关心具体后端。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorRecord:
    """待写入向量库的一条记录。

    ``vector`` 为稠密向量；``metadata`` 携带 Chunk 原文与元数据，检索命中后
    可直接取回正文，无需额外查库。
    """

    id: str
    text: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorMatch:
    """一次向量查询命中的结果。"""

    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """所有向量存储后端的抽象基类。"""

    provider: str = "base"

    @abstractmethod
    def upsert(
        self,
        records: list[VectorRecord],
        trace: Any = None,
    ) -> int:
        """幂等写入一批记录，返回写入条数。"""
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        trace: Any = None,
    ) -> list[VectorMatch]:
        """按向量相似度查询 Top-K 结果。"""
        raise NotImplementedError

    @abstractmethod
    def get_by_ids(
        self,
        ids: list[str],
        trace: Any = None,
    ) -> list[VectorMatch]:
        """按 chunk ID 批量取回正文和元数据。

        稀疏检索的 BM25 索引只保存 ``chunk_id`` 和评分；该方法用于把命中
        ID 补全为可直接展示、可生成引用的文本结果。
        """
        raise NotImplementedError
