"""D2：稠密（语义）检索器。

它负责把用户问题转换成向量，再交给 ``VectorStore`` 做近邻搜索。这里不
关心 ChromaDB 等具体实现：Embedding 和 VectorStore 都通过工厂创建，也能
在测试中注入替身对象。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.types import RetrievalResult
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.base_vector_store import BaseVectorStore, VectorMatch
from libs.vector_store.vector_store_factory import VectorStoreFactory


class DenseRetriever:
    """编排“查询文本 → 查询向量 → 向量检索”的语义召回流程。

    Args:
        settings: 完整的 ``Settings`` 对象，提供 ``embedding`` 和
            ``vector_store`` 两段配置。
        embedding_client: 可选的 Embedding 实现；传入后不会创建真实客户端，
            便于单元测试。
        vector_store: 可选的向量存储实现；同样支持依赖注入。
    """

    name = "dense_retriever"

    def __init__(
        self,
        settings: Any,
        embedding_client: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client or EmbeddingFactory.create(
            settings.embedding
        )
        self.vector_store = vector_store or VectorStoreFactory.create(
            settings.vector_store
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        trace: Any = None,
    ) -> list[RetrievalResult]:
        """返回与 ``query`` 语义最接近的至多 ``top_k`` 个文本块。

        空白查询没有可向量化的语义，直接返回空列表；正整数 ``top_k`` 是
        检索数量，非法值会尽早报错，避免底层后端出现不一致的行为。
        """
        if not isinstance(query, str):
            raise TypeError("query 必须是字符串")
        if not query.strip():
            return []
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        vectors = self.embedding_client.embed([query], trace=trace)
        if not vectors or not vectors[0]:
            raise ValueError("Embedding 客户端未返回查询向量")

        matches = self.vector_store.query(
            vectors[0], top_k=top_k, filters=filters, trace=trace
        )
        return [self._to_retrieval_result(match) for match in matches]

    @staticmethod
    def _to_retrieval_result(
        match: VectorMatch | Mapping[str, Any],
    ) -> RetrievalResult:
        """将向量库契约转换为上层通用的 ``RetrievalResult``。"""
        if isinstance(match, Mapping):
            chunk_id = match.get("chunk_id", match.get("id"))
            score = match.get("score")
            text = match.get("text")
            metadata = match.get("metadata", {})
        else:
            chunk_id = match.id
            score = match.score
            text = match.text
            metadata = match.metadata

        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("向量检索结果缺少有效的 chunk_id")
        if not isinstance(text, str):
            raise ValueError("向量检索结果缺少 text")
        if not isinstance(metadata, dict):
            raise ValueError("向量检索结果的 metadata 必须是字典")

        return RetrievalResult(
            chunk_id=chunk_id,
            score=float(score),
            text=text,
            metadata=metadata,
        )
