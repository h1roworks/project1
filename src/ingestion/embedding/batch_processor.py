"""批处理优化（C10：BatchProcessor）。

把 chunks 以 ``batch_size`` 驱动的批次送入 Dense + Sparse 双路编码，
逐批组装成等长、顺序对齐的 ``ChunkRecord`` 列表，供 C12 VectorUpserter 消费。

对齐 DEV_SPEC 3.1.2 "Embedding (双路向量化)"：
- 核心策略：系统对每个 Chunk 并行执行双路编码计算（Dense 语义向量 + Sparse 关键词权重）；
- 批处理优化：所有计算均采用 ``batch_size`` 驱动的批处理模式，
  最大化 CPU 利用率并减少网络 RTT；
- 组装为 ``ChunkRecord``（C1 定义的存储/检索载体），使上游只理解
  "chunks → ChunkRecord[]" 这一条统一流水线，不关心 Dense/Sparse 编码细节。
"""

from __future__ import annotations

from typing import Any

from core.types import Chunk, ChunkRecord

_DEFAULT_BATCH_SIZE = 32  # 与 EmbeddingSettings.batch_size 默认值一致


class BatchProcessor:
    """双路编码批处理器：Dense + Sparse 逐批编码并组装 ChunkRecord。

    DenseEncoder 内部已含内容哈希缓存（C8，增量编码）与 Embedding API 批处理，
    SparseEncoder 为纯本地词频统计（C9）；本模块负责把二者编排为统一的
    "chunks → ChunkRecord[]" 批处理流水线。
    """

    name = "batch_processor"

    def __init__(
        self,
        dense_encoder: Any,
        sparse_encoder: Any,
        batch_size: int | None = None,
    ) -> None:
        """初始化。

        Args:
            dense_encoder: DenseEncoder 实例（或任何实现 ``encode(chunks, trace)`` 的对象）。
            sparse_encoder: SparseEncoder 实例（或任何实现 ``encode(chunks, trace)`` 的对象）。
            batch_size: 每批处理的 chunk 数；默认继承 ``dense_encoder.batch_size``
                （该值已由 DenseEncoder 从 EmbeddingSettings 解析），显式传值优先。
        """
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = max(
            1, batch_size or getattr(dense_encoder, "batch_size", _DEFAULT_BATCH_SIZE)
        )

    def batch_ranges(self, total: int) -> list[tuple[int, int]]:
        """把 ``total`` 个元素切分为 ``[(start, end), ...]`` 批次区间（左闭右开）。"""
        if total <= 0:
            return []
        return [
            (start, min(start + self.batch_size, total))
            for start in range(0, total, self.batch_size)
        ]

    def process(self, chunks: list[Chunk], trace: Any = None) -> list[ChunkRecord]:
        """对 chunks 执行双路编码，返回等长、顺序对齐的 ChunkRecord 列表。

        Args:
            chunks: 待编码的 Chunk 列表（读取 ``.id``/``.text``/``.metadata``）。
            trace: 可选 TraceContext，透传给 Dense/Sparse 编码器（阶段 F 落地）。

        Returns:
            ``len(chunks)`` 条 ChunkRecord，``result[i]`` 对应 ``chunks[i]``，
            每条携带 ``dense_vector``（语义向量）与 ``sparse_vector``（关键词权重）。
        """
        if not chunks:
            return []

        records: list[ChunkRecord] = []
        for start, end in self.batch_ranges(len(chunks)):
            batch = chunks[start:end]
            dense = self.dense_encoder.encode(batch, trace=trace)
            sparse = self.sparse_encoder.encode(batch, trace=trace)
            if len(dense) != len(batch) or len(sparse) != len(batch):
                raise ValueError(
                    f"编码结果与批次长度不一致: "
                    f"dense={len(dense)}, sparse={len(sparse)}, batch={len(batch)}"
                )
            records.extend(
                ChunkRecord(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    dense_vector=dvec,
                    sparse_vector=svec,
                )
                for chunk, dvec, svec in zip(batch, dense, sparse)
            )
        return records
