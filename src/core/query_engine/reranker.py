"""D6：Core 层重排编排与稳定回退。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from time import perf_counter
from typing import Any

from core.trace import TraceContext
from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker, RerankCandidate
from libs.reranker.reranker_factory import RerankerFactory


class Reranker:
    """将融合候选交给可插拔后端精排，失败时回退到原融合顺序。"""

    name = "reranker"

    def __init__(
        self,
        settings: Any,
        reranker: BaseReranker | None = None,
    ) -> None:
        self.settings = settings
        self.backend = reranker or RerankerFactory.create(settings.rerank)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        trace: TraceContext | None = None,
    ) -> list[RetrievalResult]:
        """精排融合候选；后端异常或超时时安全返回原融合排序。

        回退结果的 metadata 会写入 ``rerank_fallback=True`` 与错误原因，
        便于后续 MCP 响应和观测系统识别这次精排未成功。
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if not isinstance(candidates, list):
            raise TypeError("candidates 必须是列表")
        started = perf_counter()
        backend_name = getattr(self.backend, "backend", "unknown")
        if not candidates:
            _record_rerank_trace(trace, started, backend_name, 0, 0, skipped=True)
            return []

        pool, tail = self._split_top_m(candidates)
        try:
            reranked = self._run_with_timeout(query, pool, trace)
            results = self._merge_reranked(reranked, pool, tail, backend_name)
            _record_rerank_trace(
                trace,
                started,
                backend_name,
                len(candidates),
                len(results),
                fallback=False,
                before_ids=[item.chunk_id for item in candidates],
                after_ids=[item.chunk_id for item in results],
            )
            return results
        except Exception as exc:  # 任何后端/超时问题均保留 D5 的融合排序
            results = self._fallback(candidates, backend_name, exc)
            _record_rerank_trace(
                trace,
                started,
                backend_name,
                len(candidates),
                len(results),
                fallback=True,
                error=str(exc),
                before_ids=[item.chunk_id for item in candidates],
                after_ids=[item.chunk_id for item in results],
            )
            return results

    def _split_top_m(
        self,
        candidates: list[RetrievalResult],
    ) -> tuple[list[RetrievalResult], list[RetrievalResult]]:
        top_m = getattr(getattr(self.settings, "rerank", None), "top_m", 0)
        if isinstance(top_m, int) and not isinstance(top_m, bool) and top_m > 0:
            return candidates[:top_m], candidates[top_m:]
        return list(candidates), []

    def _run_with_timeout(
        self,
        query: str,
        candidates: list[RetrievalResult],
        trace: TraceContext | None,
    ) -> list[RerankCandidate]:
        rerank_candidates = [
            RerankCandidate(
                id=item.chunk_id,
                text=item.text,
                score=item.score,
                metadata=dict(item.metadata),
            )
            for item in candidates
        ]
        timeout_seconds = getattr(
            getattr(self.settings, "rerank", None), "timeout_seconds", 0
        )
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            return self.backend.rerank(query, rerank_candidates, trace=trace)

        # 不使用 ``with ThreadPoolExecutor``：超时后 context manager 会等待任务
        # 结束，反而失去超时保护的意义。
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.backend.rerank, query, rerank_candidates, trace)
        try:
            return future.result(timeout=timeout_seconds)
        except TimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"Reranker 在 {timeout_seconds} 秒内未完成") from exc
        finally:
            # 超时时不等待仍在运行的外部调用；正常情况则等待并释放线程。
            executor.shutdown(wait=not future.running(), cancel_futures=True)

    @staticmethod
    def _merge_reranked(
        reranked: list[RerankCandidate],
        pool: list[RetrievalResult],
        tail: list[RetrievalResult],
        backend_name: str,
    ) -> list[RetrievalResult]:
        """把后端候选映射回完整结果，并保留未参与/未返回的候选。"""
        originals = {item.chunk_id: item for item in pool}
        ordered: list[RetrievalResult] = []
        seen: set[str] = set()

        for item in reranked:
            original = originals.get(item.id)
            if original is None or item.id in seen:
                continue
            metadata = dict(original.metadata)
            metadata.update(item.metadata)
            metadata.update({"rerank_backend": backend_name, "rerank_fallback": False})
            ordered.append(
                RetrievalResult(
                    chunk_id=item.id,
                    score=float(item.score),
                    text=item.text,
                    metadata=metadata,
                )
            )
            seen.add(item.id)

        for original in [*pool, *tail]:
            if original.chunk_id in seen:
                continue
            metadata = dict(original.metadata)
            metadata.update({"rerank_backend": backend_name, "rerank_fallback": False})
            ordered.append(
                RetrievalResult(
                    chunk_id=original.chunk_id,
                    score=original.score,
                    text=original.text,
                    metadata=metadata,
                )
            )
            seen.add(original.chunk_id)
        return ordered

    @staticmethod
    def _fallback(
        candidates: list[RetrievalResult],
        backend_name: str,
        error: Exception,
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id=item.chunk_id,
                score=item.score,
                text=item.text,
                metadata={
                    **item.metadata,
                    "rerank_backend": backend_name,
                    "rerank_fallback": True,
                    "rerank_error": str(error),
                },
            )
            for item in candidates
        ]


def _record_rerank_trace(
    trace: TraceContext | None,
    started: float,
    backend_name: str,
    candidate_count: int,
    result_count: int,
    *,
    fallback: bool = False,
    skipped: bool = False,
    error: str | None = None,
    before_ids: list[str] | None = None,
    after_ids: list[str] | None = None,
) -> None:
    """记录 rerank 的结果；观察失败不应改变原有回退语义。"""
    if trace is None or not callable(getattr(trace, "record_stage", None)):
        return
    details: dict[str, Any] = {
        "candidate_count": candidate_count,
        "result_count": result_count,
        "fallback": fallback,
        "skipped": skipped,
        "before_ids": before_ids or [],
        "after_ids": after_ids or [],
    }
    if error:
        details["error"] = error
    try:
        trace.record_stage(
            "rerank",
            (perf_counter() - started) * 1000,
            method="rerank",
            provider=backend_name,
            details=details,
        )
    except Exception:
        pass
