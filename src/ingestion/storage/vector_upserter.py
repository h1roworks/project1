"""向量库幂等 Upsert（C12：VectorUpserter）。

消费 C10 BatchProcessor 产出的 ChunkRecord，把"稠密向量 + 稀疏向量 + 正文 + 富
Metadata"原子化写入向量存储，并同步更新 C11 BM25Indexer 稀疏索引，完成摄取链路的
最后一环（Embed → Upsert）。

对齐 DEV_SPEC 3.1.1 "Upsert & Storage (索引存储)"：
- **All-in-One 存储策略**：每条记录同时携带 Dense Vector（写入向量库的 embeddings）、
  Sparse Vector（写入 BM25 倒排索引，并以 JSON 存入 metadata 保持 payload 完整）、
  以及 Chunk 原文与富 Metadata；
- **幂等性设计**：Chroma upsert 按 id 覆盖（原生幂等），BM25Indexer.add 同为
  Upsert 语义（重复 chunk_id 先移除旧条目再写入），重复摄取不会产生重复索引；
- **metadata 完整**：Chroma 的 metadata 值仅支持标量/标量列表，本模块对富 Metadata
  做安全序列化（复杂结构 → JSON 字符串），保证 ``source_path`` 等可过滤字段原样保留。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord

_PRIMITIVE = (str, int, float, bool)
_SPARSE_META_KEY = "sparse_vector"


@dataclass
class UpsertResult:
    """一次 ``upsert`` 的统计结果，供日志/追踪/进度回调展示。

    - ``total``: 输入 ChunkRecord 总数。
    - ``vector_store_count``: 实际写入向量库的条数（仅含携带稠密向量的记录）。
    - ``bm25_count``: 更新到 BM25 索引的条数（含仅稀疏的记录）。
    - ``skipped_no_dense``: 因缺少稠密向量、未写入向量库的条数。
    """

    total: int = 0
    vector_store_count: int = 0
    bm25_count: int = 0
    skipped_no_dense: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "vector_store_count": self.vector_store_count,
            "bm25_count": self.bm25_count,
            "skipped_no_dense": self.skipped_no_dense,
        }


class VectorUpserter:
    """把 ChunkRecord 双路写入向量库 + BM25 索引的编排器。"""

    name = "vector_upserter"

    def __init__(
        self,
        vector_store: BaseVectorStore,
        bm25_indexer: Any | None = None,
    ) -> None:
        """初始化。

        Args:
            vector_store: 向量库后端（ChromaStore 等 BaseVectorStore 实现），
                接收稠密向量 + 正文 + 规约后的 metadata。
            bm25_indexer: BM25Indexer（或 None）。为 None 时跳过稀疏索引
                （dense-only 模式）；提供且配置了 ``index_path`` 时，每次
                upsert 后自动保存到磁盘（增量持久化）。
        """
        self.vector_store = vector_store
        self.bm25_indexer = bm25_indexer

    def upsert(self, records: list[Any], trace: Any = None) -> UpsertResult:
        """把 ChunkRecord 列表双路写入存储，返回统计结果（幂等）。

        稠密路径：有 ``dense_vector`` 的记录转为 ``VectorRecord`` 写入向量库；
        稀疏路径：全部记录送入 BM25Indexer（其内部对 ``sparse_vector=None`` 的
        chunk 仍计入文档表、不产生词项）。
        """
        if not records:
            return UpsertResult()

        dense_records = [r for r in records if r.dense_vector is not None]
        result = UpsertResult(
            total=len(records),
            skipped_no_dense=len(records) - len(dense_records),
        )

        if dense_records:
            result.vector_store_count = self.vector_store.upsert(
                [self._to_vector_record(r) for r in dense_records],
                trace=trace,
            )

        if self.bm25_indexer is not None:
            result.bm25_count = self.bm25_indexer.add(records)
            index_path = getattr(self.bm25_indexer, "index_path", None)
            if index_path is not None:
                self.bm25_indexer.save()

        return result

    def _to_vector_record(self, record: Any) -> VectorRecord:
        """把 ChunkRecord 转为 VectorRecord，并把富 metadata 规约到 Chroma 兼容类型。

        稀疏向量以 JSON 存入 metadata（键 ``sparse_vector``），使单条记录同时承载
        Dense + Sparse 双路索引数据，满足 All-in-One 存储约定。
        """
        metadata = dict(record.metadata)
        if record.sparse_vector:
            metadata[_SPARSE_META_KEY] = json.dumps(
                record.sparse_vector, ensure_ascii=False, sort_keys=True
            )
        return VectorRecord(
            id=record.id,
            text=record.text,
            vector=record.dense_vector,
            metadata=_sanitize_metadata(metadata),
        )


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """把富 Metadata 规约为 Chroma 兼容的标量/标量列表，保证 upsert 不报错。"""
    return {key: _to_primitive(value) for key, value in metadata.items()}


def _to_primitive(value: Any) -> Any:
    """递归规约：标量保留、非空标量列表保留，其余（含空列表）转 JSON 字符串。

    chroma 要求 metadata 列表值非空（``ValueError: ... to be non-empty``），
    因此空列表也转为 ``"[]"`` 字符串以通过校验。
    """
    if _is_scalar(value):
        return value
    if isinstance(value, list) and value and all(_is_scalar(item) for item in value):
        return value
    return _to_json(value)


def _is_scalar(value: Any) -> bool:
    """是否为 Chroma 可接受的 metadata 标量（str/int/float/bool）。"""
    return isinstance(value, _PRIMITIVE)


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
