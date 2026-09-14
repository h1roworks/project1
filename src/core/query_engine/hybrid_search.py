"""D5：编排 Dense、Sparse 与 RRF 的混合检索。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from time import perf_counter
from typing import Any

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import Fusion
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.sparse_retriever import SparseRetriever
from core.trace import TraceContext
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
        trace: TraceContext | None = None,
        on_stage: Callable[[str, list[RetrievalResult]], None] | None = None,
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

        started = perf_counter()
        processed = self.query_processor.process(query, filters=filters, trace=trace)
        _record_trace(
            trace,
            "query_processing",
            started,
            method=getattr(processed, "method", "rule"),
            provider="local",
            details={
                "keyword_count": len(processed.keywords),
                "sparse_term_count": len(processed.sparse_terms),
                "filter_keys": sorted(processed.filters),
            },
        )
        dense_top_k, sparse_top_k = self._candidate_limits(top_k)
        pre_filters = self._hard_filters(processed.filters)

        dense_results, sparse_results, retrieval_info = self._retrieve_in_parallel(
            dense_query=processed.dense_query,
            sparse_terms=processed.sparse_terms,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            pre_filters=pre_filters,
            trace=trace,
        )
        _record_retrieval_trace(
            trace, "dense_retrieval", "dense", self.dense_retriever,
            dense_results, dense_top_k, retrieval_info["dense"],
        )
        _record_retrieval_trace(
            trace, "sparse_retrieval", "bm25", self.sparse_retriever,
            sparse_results, sparse_top_k, retrieval_info["sparse"],
        )
        self._notify_stage(on_stage, "dense", dense_results)
        self._notify_stage(on_stage, "sparse", sparse_results)
        if not dense_results and not sparse_results:
            _record_elapsed_trace(
                trace,
                "fusion",
                0.0,
                method="rrf",
                provider=getattr(self.fusion, "name", type(self.fusion).__name__),
                details={
                    "dense_result_count": 0,
                    "sparse_result_count": 0,
                    "result_count": 0,
                    "skipped": True,
                },
            )
            return []

        started = perf_counter()
        fused = self.fusion.fuse(
            dense_results, sparse_results, top_k=None, trace=trace
        )
        _record_trace(
            trace,
            "fusion",
            started,
            method="rrf",
            provider=getattr(self.fusion, "name", type(self.fusion).__name__),
            details={
                "dense_result_count": len(dense_results),
                "sparse_result_count": len(sparse_results),
                "result_count": len(fused),
            },
        )
        self._notify_stage(on_stage, "fusion", fused)
        filtered = self._apply_metadata_filters(fused, processed.filters)
        return filtered[:top_k]

    @staticmethod
    def _notify_stage(
        on_stage: Callable[[str, list[RetrievalResult]], None] | None,
        stage: str,
        results: list[RetrievalResult],
    ) -> None:
        """向可选观察者报告阶段结果，观察代码本身不能影响检索。"""
        if on_stage is None:
            return
        try:
            on_stage(stage, list(results))
        except Exception:
            # CLI / tracing 的展示错误不应破坏主查询链路。
            pass

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
        trace: TraceContext | None,
    ) -> tuple[
        list[RetrievalResult],
        list[RetrievalResult],
        dict[str, dict[str, Any]],
    ]:
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
            return [], [], {
                "dense": {"elapsed_ms": 0.0, "skipped": True},
                "sparse": {"elapsed_ms": 0.0, "skipped": True},
            }

        results: dict[str, list[RetrievalResult]] = {"dense": [], "sparse": []}
        info: dict[str, dict[str, Any]] = {
            "dense": {"elapsed_ms": 0.0, "skipped": "dense" not in tasks},
            "sparse": {"elapsed_ms": 0.0, "skipped": "sparse" not in tasks},
        }

        def timed(task: Callable[[], list[RetrievalResult]]) -> tuple[
            list[RetrievalResult], float, str | None
        ]:
            started = perf_counter()
            try:
                return task(), (perf_counter() - started) * 1000, None
            except Exception as exc:  # 单路失败不阻断另一条召回路径
                return [], (perf_counter() - started) * 1000, str(exc)

        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {name: executor.submit(timed, task) for name, task in tasks.items()}
            for name, future in futures.items():
                try:
                    results[name], info[name]["elapsed_ms"], error = future.result()
                    if error:
                        info[name]["error"] = error
                except Exception:  # 单路失败不阻断另一条召回路径
                    results[name] = []
                    info[name]["error"] = "retrieval task failed unexpectedly"
        return results["dense"], results["sparse"], info

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


def _record_retrieval_trace(
    trace: TraceContext | None,
    stage_name: str,
    method: str,
    retriever: Any,
    results: list[RetrievalResult],
    top_k: int,
    info: dict[str, Any],
) -> None:
    """把并行检索线程测得的耗时写回主 trace。"""
    details = {
        "result_count": len(results),
        "top_k": top_k,
        "skipped": bool(info.get("skipped", False)),
    }
    if info.get("error"):
        details["error"] = info["error"]
    _record_elapsed_trace(
        trace,
        stage_name,
        float(info["elapsed_ms"]),
        method=method,
        provider=getattr(retriever, "name", type(retriever).__name__),
        details=details,
    )


def _record_trace(
    trace: TraceContext | None,
    stage_name: str,
    started: float,
    *,
    method: str,
    provider: str,
    details: dict[str, Any],
) -> None:
    _record_elapsed_trace(
        trace,
        stage_name,
        (perf_counter() - started) * 1000,
        method=method,
        provider=provider,
        details=details,
    )


def _record_elapsed_trace(
    trace: TraceContext | None,
    stage_name: str,
    elapsed_ms: float,
    *,
    method: str,
    provider: str,
    details: dict[str, Any],
) -> None:
    """追踪记录不可反过来影响检索主流程。"""
    if trace is None or not callable(getattr(trace, "record_stage", None)):
        return
    try:
        trace.record_stage(
            stage_name,
            elapsed_ms,
            method=method,
            provider=provider,
            details=details,
        )
    except Exception:
        pass
