"""D4：使用 Reciprocal Rank Fusion（RRF）融合双路检索结果。"""

from __future__ import annotations

from typing import Any

from core.types import RetrievalResult

_DEFAULT_RRF_K = 60


class Fusion:
    """将 Dense 与 Sparse 的候选排名融合为一份统一排名。

    RRF 不比较两条路径原始 ``score`` 的数值，而是只看每条路径中的名次。
    这很适合本项目：Dense 是余弦相似度，Sparse 是 BM25 分数，二者的数值
    范围没有可直接比较的共同尺度。

    Args:
        k: RRF 平滑常数。数值越大，排名前后差异越平缓；DEV_SPEC 与配置
            文件默认使用 60。
    """

    name = "rrf_fusion"

    def __init__(self, k: int = _DEFAULT_RRF_K) -> None:
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("RRF 的 k 必须是正整数")
        self.k = k

    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int | None = None,
        trace: Any = None,
    ) -> list[RetrievalResult]:
        """按 RRF 分数返回去重后的统一排序结果。

        一个 chunk 同时被两条路径命中时会累加两份 RRF 分数；只被一条路径
        命中时仍会保留。输入列表内重复的 chunk 只以第一次出现的名次计分。
        在完全同分时按最佳名次、再按 ``chunk_id`` 排序，保证结果可复现。
        """
        if top_k is not None and (
            not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0
        ):
            raise ValueError("top_k 必须是正整数或 None")

        scores: dict[str, float] = {}
        best_ranks: dict[str, int] = {}
        result_by_id: dict[str, RetrievalResult] = {}

        self._add_ranked_results(
            dense_results, scores, best_ranks, result_by_id
        )
        self._add_ranked_results(
            sparse_results, scores, best_ranks, result_by_id
        )

        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: (-scores[chunk_id], best_ranks[chunk_id], chunk_id),
        )
        if top_k is not None:
            ranked_ids = ranked_ids[:top_k]

        return [
            RetrievalResult(
                chunk_id=chunk_id,
                score=scores[chunk_id],
                text=result_by_id[chunk_id].text,
                metadata=result_by_id[chunk_id].metadata,
            )
            for chunk_id in ranked_ids
        ]

    def _add_ranked_results(
        self,
        results: list[RetrievalResult],
        scores: dict[str, float],
        best_ranks: dict[str, int],
        result_by_id: dict[str, RetrievalResult],
    ) -> None:
        """把一条检索路径的 RRF 贡献累加到共享结果中。"""
        seen: set[str] = set()
        for rank, result in enumerate(results, start=1):
            if result.chunk_id in seen:
                continue
            seen.add(result.chunk_id)

            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (
                self.k + rank
            )
            best_ranks[result.chunk_id] = min(
                best_ranks.get(result.chunk_id, rank), rank
            )
            # 两条路径应指向同一 chunk；保留 Dense 路径优先记录即可。
            result_by_id.setdefault(result.chunk_id, result)
