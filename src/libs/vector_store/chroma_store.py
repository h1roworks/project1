"""ChromaStore：基于 chromadb 的默认向量存储后端。

支持最小 ``upsert(records)`` / ``query(vector, top_k, filters)``，并支持本地
持久化目录（``persist_dir``）。chroma 返回余弦距离，统一转换为相似度
（``score = 1 - distance``，越大越相关），与内存实现的契约保持一致。
"""

from __future__ import annotations

from typing import Any

import chromadb

from libs.vector_store.base_vector_store import BaseVectorStore, VectorMatch, VectorRecord
from libs.vector_store.vector_store_factory import VectorStoreFactory


class ChromaStore(BaseVectorStore):
    provider = "chroma"

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._client = chromadb.PersistentClient(
            path=settings.persist_dir or "data/db/chroma"
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection or "default",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        records: list[VectorRecord],
        trace: Any = None,
    ) -> int:
        self._collection.upsert(
            ids=[r.id for r in records],
            embeddings=[r.vector for r in records],
            documents=[r.text for r in records],
            metadatas=[r.metadata for r in records],
        )
        return len(records)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        trace: Any = None,
    ) -> list[VectorMatch]:
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=max(top_k, 1),
            where=filters or None,
        )
        ids = (result["ids"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]

        matches: list[VectorMatch] = []
        for i, cid in enumerate(ids):
            matches.append(
                VectorMatch(
                    id=cid,
                    score=round(1.0 - distances[i], 6),
                    text=documents[i],
                    metadata=metadatas[i] or {},
                )
            )
        return matches


VectorStoreFactory.register("chroma", ChromaStore)
