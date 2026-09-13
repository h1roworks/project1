"""D3：基于 BM25 的稀疏（关键词）检索器。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.vector_store.base_vector_store import BaseVectorStore, VectorMatch
from libs.vector_store.vector_store_factory import VectorStoreFactory

_DEFAULT_BM25_INDEX_PATH = Path("data/db/bm25/index.pkl")


class SparseRetriever:
    """编排“关键词 → BM25 排名 → 回取 chunk 正文”的稀疏召回流程。

    BM25 索引为轻量倒排表，只保存 chunk ID 和评分所需统计；而正文和完整
    元数据保存在 VectorStore。将两者分开能让 BM25 索引保持小巧，也能让
    Dense/Sparse 两条路径返回同一种 ``RetrievalResult``。
    """

    name = "sparse_retriever"

    def __init__(
        self,
        settings: Any,
        bm25_indexer: BM25Indexer | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.bm25_indexer = bm25_indexer or BM25Indexer(_DEFAULT_BM25_INDEX_PATH)
        self.vector_store = vector_store or VectorStoreFactory.create(
            settings.vector_store
        )

    def retrieve(
        self,
        keywords: list[str] | dict[str, float],
        top_k: int = 10,
        trace: Any = None,
    ) -> list[RetrievalResult]:
        """返回包含任一关键词的 BM25 Top-K 文本块。

        参数可以是关键词列表，也可以是 QueryProcessor 生成的加权
        ``sparse_terms``。空关键词直接返回空列表，避免查询整个索引。
        """
        if not isinstance(keywords, (list, dict)):
            raise TypeError("keywords 必须是关键词列表或词权重字典")
        if not keywords:
            return []
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        candidates = self.bm25_indexer.query(keywords, top_k=top_k, trace=trace)
        if not candidates:
            return []

        scores = {
            candidate["chunk_id"]: float(candidate["score"])
            for candidate in candidates
            if isinstance(candidate.get("chunk_id"), str)
        }
        if not scores:
            return []
        matches = self.vector_store.get_by_ids(list(scores), trace=trace)
        matches_by_id = {
            chunk_id: (text, metadata)
            for match in matches
            for chunk_id, text, metadata in [self._unpack_match(match)]
        }

        # 以 BM25 的候选顺序构造结果，不能依赖底层向量库的返回顺序。
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                score=scores[chunk_id],
                text=matches_by_id[chunk_id][0],
                metadata=matches_by_id[chunk_id][1],
            )
            for chunk_id in scores
            if chunk_id in matches_by_id
        ]

    @staticmethod
    def _unpack_match(
        match: VectorMatch | Mapping[str, Any],
    ) -> tuple[str, str, dict[str, Any]]:
        """兼容 VectorMatch 以及遵循同字段约定的字典式后端。"""
        if isinstance(match, Mapping):
            chunk_id = match.get("chunk_id", match.get("id"))
            text = match.get("text")
            metadata = match.get("metadata", {})
        else:
            chunk_id = match.id
            text = match.text
            metadata = match.metadata

        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("向量库回取结果缺少有效的 chunk_id")
        if not isinstance(text, str):
            raise ValueError("向量库回取结果缺少 text")
        if not isinstance(metadata, dict):
            raise ValueError("向量库回取结果的 metadata 必须是字典")
        return chunk_id, text, metadata
