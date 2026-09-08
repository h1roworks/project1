"""B7.8: Cross-Encoder Reranker 测试（注入 mock scorer，保证确定性）。

覆盖工厂路由、按分数重排、Top-M 限制、单候选短路、打分失败/数量不符的
回退信号与输入校验。
"""

import pytest

from core.settings import RerankSettings
from libs.reranker.base_reranker import RerankCandidate
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker, RerankError
from libs.reranker.reranker_factory import RerankerFactory


class MockScorer:
    def __init__(self, scores: list[float], exc: Exception | None = None) -> None:
        self.scores = scores
        self.exc = exc
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        if self.exc:
            raise self.exc
        return self.scores


def make_settings(**overrides) -> RerankSettings:
    base = {
        "provider": "cross_encoder",
        "enabled": True,
        "top_m": 10,
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }
    base.update(overrides)
    return RerankSettings(**base)


def make_candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", text="苹果怎么种植", score=0.3),
        RerankCandidate(id="b", text="香蕉的产地", score=0.9),
        RerankCandidate(id="c", text="橙子的营养", score=0.5),
    ]


def make_reranker(scores, exc=None) -> CrossEncoderReranker:
    return CrossEncoderReranker(make_settings(), scorer=MockScorer(scores, exc))


# ---------- 工厂路由 ----------

def test_factory_creates_cross_encoder() -> None:
    r = RerankerFactory.create(make_settings())
    assert isinstance(r, CrossEncoderReranker)


# ---------- 重排逻辑 ----------

def test_rerank_sorts_by_scorer_scores() -> None:
    r = make_reranker([0.5, 0.9, 0.1])  # 对应 a/b/c 的文本顺序
    result = r.rerank("水果种植", make_candidates())
    assert [c.id for c in result] == ["b", "a", "c"]
    assert [c.score for c in result] == [0.9, 0.5, 0.1]
    assert r._scorer.calls[0][0] == "水果种植"  # query 透传给 scorer


def test_rerank_passes_texts_in_candidate_order() -> None:
    r = make_reranker([0.0, 0.0, 0.0])
    r.rerank("q", make_candidates())
    assert r._scorer.calls[0][1] == ["苹果怎么种植", "香蕉的产地", "橙子的营养"]


def test_top_m_limits_scored_pool() -> None:
    r = CrossEncoderReranker(
        make_settings(top_m=2), scorer=MockScorer([0.1, 0.9])
    )
    result = r.rerank("q", make_candidates())
    # 只对原始分数前 2 名（b、c）打分，a 被排除
    assert [c.id for c in result] == ["c", "b"]
    assert len(r._scorer.calls[0][1]) == 2


def test_single_candidate_returns_as_is() -> None:
    r = make_reranker([0.5])
    cands = [make_candidates()[0]]
    assert r.rerank("q", cands) == cands


def test_returns_new_candidates_not_mutating_original() -> None:
    r = make_reranker([0.5, 0.9, 0.1])
    original = make_candidates()
    r.rerank("q", original)
    assert original[0].score == 0.3  # 原对象分数未被修改


# ---------- 失败回退信号 ----------

def test_scorer_failure_raises_fallback_signal() -> None:
    r = make_reranker([], exc=TimeoutError("scorer timed out"))
    with pytest.raises(RerankError, match="打分失败"):
        r.rerank("q", make_candidates())


def test_scorer_wrong_count_raises() -> None:
    r = make_reranker([0.1])  # 3 个候选只返回 1 个分数
    with pytest.raises(RerankError, match="数量"):
        r.rerank("q", make_candidates())


# ---------- 输入校验 ----------

def test_invalid_query_raises() -> None:
    r = make_reranker([0.5, 0.9, 0.1])
    with pytest.raises(ValueError, match="query"):
        r.rerank("", make_candidates())


def test_invalid_candidates_raise() -> None:
    r = make_reranker([0.5, 0.9, 0.1])
    with pytest.raises(ValueError, match="candidates"):
        r.rerank("q", [])
