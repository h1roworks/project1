"""D5：编排 Dense、Sparse 与 RRF 的混合检索。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import Fusion
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.sparse_retriever import SparseRetriever
from core.types import RetrievalResult


class HybridSearch:
    """执行“预处理 → 并行双路召回 → RRF → 过滤 → Top-K”的查询流程。"""

    name = "hybrid_search"

    def __init__(
        self,
        settings: Any,
        query_processor: QueryProcessor | None = None,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        fusion: Fusion | None = None,
    ) -> None:
        self.settings = settings
        self.query_processor = query_processor or QueryProcessor()
        self.dense_retriever = dense_retriever or DenseRetriever(settings)
        self.sparse_retriever = sparse_retriever or SparseRetriever(settings)
        fusion_k = getattr(getattr(settings, "retrieval", None), "fusion_k", 60)
        self.fusion = fusion or Fusion(k=fusion_k)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        trace: Any = None,
    ) -> list[RetrievalResult]:
        """执行一次完整混合检索，并返回经过兜底过滤的 Top-K 结果。

        任一路召回失败时仍使用另一路的结果；两路都不可用时返回空列表，
        让上层能给用户友好的“未找到结果”提示。
        """
        if not isinstance(query, str):
            raise TypeError("query 必须是字符串")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        if filters is not None and not isinstance(filters, dict):
            raise TypeError("filters 必须是字典或 None")

        processed = self.query_processor.process(query, filters=filters, trace=trace)
        dense_top_k, sparse_top_k = self._candidate_limits(top_k)
        pre_filters = self._hard_filters(processed.filters)

        dense_results, sparse_results = self._retrieve_in_parallel(
            dense_query=processed.dense_query,
            sparse_terms=processed.sparse_terms,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            pre_filters=pre_filters,
            trace=trace,
        )
        if not dense_results and not sparse_results:
            return []

        fused = self.fusion.fuse(
            dense_results, sparse_results, top_k=None, trace=trace
        )
        filtered = self._apply_metadata_filters(fused, processed.filters)
        return filtered[:top_k]

    def _candidate_limits(self, top_k: int) -> tuple[int, int]:
        """读取配置的双路候选数量；缺省时至少满足最终 Top-K。"""
        retrieval = getattr(self.settings, "retrieval", None)
        dense_top_k = getattr(retrieval, "dense_top_k", top_k)
        sparse_top_k = getattr(retrieval, "sparse_top_k", top_k)
        return max(top_k, dense_top_k), max(top_k, sparse_top_k)

    def _hard_filters(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        """仅把配置为硬约束的字段前置给支持过滤的 Dense 向量库。"""
        retrieval = getattr(self.settings, "retrieval", None)
        hard_filter_names = set(getattr(retrieval, "hard_filters", []) or [])
        pre_filters = {
            key: value for key, value in filters.items() if key in hard_filter_names
        }
        return pre_filters or None

    def _retrieve_in_parallel(
        self,
        dense_query: str,
        sparse_terms: dict[str, float],
        dense_top_k: int,
        sparse_top_k: int,
        pre_filters: dict[str, Any] | None,
        trace: Any,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        """并行执行可用的召回路径，并将单路异常降级为空结果。"""
        tasks = {}
        if dense_query:
            tasks["dense"] = lambda: self.dense_retriever.retrieve(
                dense_query, top_k=dense_top_k, filters=pre_filters, trace=trace
            )
        if sparse_terms:
            tasks["sparse"] = lambda: self.sparse_retriever.retrieve(
                sparse_terms, top_k=sparse_top_k, trace=trace
            )
        if not tasks:
            return [], []

        results: dict[str, list[RetrievalResult]] = {"dense": [], "sparse": []}
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {name: executor.submit(task) for name, task in tasks.items()}
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception:  # 单路失败不阻断另一条召回路径
                    results[name] = []
        return results["dense"], results["sparse"]

    @staticmethod
    def _apply_metadata_filters(
        candidates: list[RetrievalResult],
        filters: dict[str, Any],
    ) -> list[RetrievalResult]:
        """做融合后的兜底过滤；候选缺少字段时宽松保留。

        单个过滤值按相等判断，列表/集合值按 OR 判断。metadata 本身的列表
        也支持与过滤值相交，例如 ``tags``。
        """
        if not filters:
            return list(candidates)

        return [
            candidate
            for candidate in candidates
            if all(
                HybridSearch._metadata_matches(candidate.metadata, key, expected)
                for key, expected in filters.items()
            )
        ]

    @staticmethod
    def _metadata_matches(metadata: dict[str, Any], key: str, expected: Any) -> bool:
        if key not in metadata or metadata[key] is None:
            return True

        actual = metadata[key]
        expected_values = (
            set(expected)
            if isinstance(expected, (list, tuple, set))
            else {expected}
        )
        actual_values = (
            set(actual) if isinstance(actual, (list, tuple, set)) else {actual}
        )
        return bool(actual_values & expected_values)
