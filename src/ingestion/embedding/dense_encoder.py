"""稠密向量编码（C8：DenseEncoder）。

把 ``chunks.text`` 批量送入 ``libs.embedding.BaseEmbedding``，产出与输入 chunks
等长、等维度对齐的稠密向量列表，供 BatchProcessor（C10）/ VectorUpserter（C12）消费。

对齐 DEV_SPEC 5.1.3 "差量计算 (Incremental Embedding / Cost Optimization)"：
- 在调用昂贵的 Embedding API 之前先计算 Chunk 内容哈希（Content Hash）；
- 同一运行内已编码过的相同内容直接复用缓存向量，显著降低 API 调用成本；
- 缓存为进程内内存缓存；跨运行的去重由 C12 幂等 upsert 兜底。
"""

from __future__ import annotations

import hashlib
from typing import Any

from libs.embedding.base_embedding import BaseEmbedding

_DEFAULT_BATCH_SIZE = 32  # 与 EmbeddingSettings.batch_size 默认值一致


def content_hash(text: str) -> str:
    """按文本内容生成稳定哈希（UTF-8 + SHA256 全量 hex）。内容不变则哈希不变。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DenseEncoder:
    """稠密向量编码器：批量 embed chunks，带内容哈希缓存与分批控制。"""

    name = "dense_encoder"

    def __init__(
        self,
        embedding: BaseEmbedding,
        settings: Any = None,
        batch_size: int | None = None,
        use_cache: bool = True,
    ) -> None:
        """初始化。

        Args:
            embedding: BaseEmbedding 实现（测试传 Fake，生产传 EmbeddingFactory 产物）。
            settings: 可选 EmbeddingSettings，也可传完整 Settings（自动取 ``.embedding``
                子段），提供 ``batch_size`` 默认值。
            batch_size: 每次调用 embedding.embed 的文本条数；显式传值时优先于 settings。
            use_cache: 是否启用内容哈希缓存（增量编码）。False 则每次全量 embed。
        """
        self.embedding = embedding
        self.batch_size = batch_size or _batch_size_from(settings)
        self.use_cache = use_cache
        self._cache: dict[str, list[float]] = {}
        self.embedded_count = 0  # 实际调用 Embedding API 编码的向量条数
        self.cache_hit_count = 0  # 命中内容哈希缓存、直接复用的向量条数

    def encode(self, chunks: list[Any], trace: Any = None) -> list[list[float]]:
        """对 chunks 批量编码，返回与输入等长、顺序对齐的稠密向量列表。

        Args:
            chunks: 待编码的 Chunk 列表（读取 ``.text``）。
            trace: 可选 TraceContext，透传给 embedding.embed（阶段 F 落地）。

        Returns:
            ``len(chunks)`` 条向量，``result[i]`` 对应 ``chunks[i]`` 的向量。
        """
        texts = [c.text for c in chunks]

        # 先按内容哈希拆分：命中缓存直接复用，未命中的标记待编码
        vectors: list[list[float] | None] = [None] * len(chunks)
        pending: list[int] = []
        for i, text in enumerate(texts):
            cached = self._cache.get(content_hash(text)) if self.use_cache else None
            if cached is not None:
                vectors[i] = cached
                self.cache_hit_count += 1
            else:
                pending.append(i)

        # 未命中的文本按 batch_size 分批送入 Embedding，逐批校验数量
        for start in range(0, len(pending), self.batch_size):
            batch_idx = pending[start : start + self.batch_size]
            batch_vectors = self.embedding.embed([texts[i] for i in batch_idx], trace=trace)
            if len(batch_vectors) != len(batch_idx):
                raise ValueError(
                    f"Embedding 返回 {len(batch_vectors)} 条向量，"
                    f"与输入 {len(batch_idx)} 条文本不一致"
                )
            for i, vec in zip(batch_idx, batch_vectors):
                vectors[i] = vec
                self.embedded_count += 1
                if self.use_cache:
                    self._cache[content_hash(texts[i])] = vec

        result: list[list[float]] = []
        for vec in vectors:
            if vec is None:
                raise ValueError("Embedding 输出缺失部分向量，无法与 chunks 对齐")
            result.append(vec)
        return result


def _batch_size_from(settings: Any) -> int:
    """从 settings 解析批大小：完整 Settings 取 ``.embedding`` 子段，缺省 32。"""
    if settings is None:
        return _DEFAULT_BATCH_SIZE
    sub = settings.embedding if hasattr(settings, "embedding") else settings
    value = getattr(sub, "batch_size", 0) or _DEFAULT_BATCH_SIZE
    return max(1, value)
