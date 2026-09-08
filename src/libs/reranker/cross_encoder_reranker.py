"""Cross-Encoder 重排实现（本地/托管模型，占位可跑）。

对 Top-M 候选按（query, text）对打分并降序重排。默认 scorer 懒加载
``sentence_transformers.CrossEncoder``；测试中注入 mock scorer 保证确定性。
任何打分失败/超时都抛出 ``RerankError``（可读的回退信号），供 Core 层 D6
捕获后回退融合排名。
"""

from __future__ import annotations

from typing import Any

from libs.reranker.base_reranker import BaseReranker, RerankCandidate
from libs.reranker.reranker_factory import RerankerFactory


class RerankError(RuntimeError):
    """Cross-Encoder 重排失败的可读错误（D6 fallback 的触发信号）。"""


class CrossEncoderReranker(BaseReranker):
    backend = "cross_encoder"

    def __init__(self, settings: Any, scorer: Any = None) -> None:
        """初始化 Cross-Encoder 重排器。

        Args:
            settings: RerankSettings（top_m / model / timeout_seconds）。
            scorer: 可注入的打分器 ``callable(query, texts) -> list[float]``；
                None 时懒加载 sentence-transformers 的 CrossEncoder。
        """
        self.settings = settings
        self._scorer = scorer
        self._model: Any = None

    def _get_scorer(self) -> Any:
        if self._scorer is None:
            self._scorer = self._build_default_scorer()
        return self._scorer

    def _build_default_scorer(self) -> Any:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankError(
                "Cross-Encoder 依赖未安装（pip install sentence-transformers）"
            ) from exc
        self._model = CrossEncoder(self.settings.model)
        return lambda query, texts: self._model.predict([[query, t] for t in texts])

    def _score(self, query: str, texts: list[str]) -> list[float]:
        scores = self._get_scorer()(query, texts)
        if len(scores) != len(texts):
            raise RerankError("scorer 返回的分数数量与候选数不一致")
        return [float(s) for s in scores]

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any = None,
    ) -> list[RerankCandidate]:
        if not isinstance(query, str) or not query:
            raise ValueError("query 必须是非空字符串")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidates 必须是非空 list")
        if len(candidates) <= 1:
            return list(candidates)

        pool = candidates
        top_m = self.settings.top_m or 0
        if 0 < top_m < len(candidates):
            # 先按现有分数取 Top-M 候选，再交给 Cross-Encoder 精排
            pool = sorted(candidates, key=lambda c: c.score, reverse=True)[:top_m]

        try:
            scores = self._score(query, [c.text for c in pool])
        except RerankError:
            raise
        except Exception as exc:  # noqa: BLE001 - 打分失败/超时统一包装为回退信号
            raise RerankError(f"Cross-Encoder 打分失败: {exc}") from exc

        ordered = sorted(zip(scores, pool), key=lambda x: x[0], reverse=True)
        return [
            RerankCandidate(
                id=c.id, text=c.text, score=s, metadata=dict(c.metadata)
            )
            for s, c in ordered
        ]


RerankerFactory.register("cross_encoder", CrossEncoderReranker)
