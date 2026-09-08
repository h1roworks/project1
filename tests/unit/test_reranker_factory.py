"""B5: Reranker 抽象接口与工厂测试（含 None 回退）。"""

import pytest

from core.settings import RerankSettings
from libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankCandidate
from libs.reranker.reranker_factory import RerankerFactory


class FakeReranker(BaseReranker):
    backend = "fake"

    def __init__(self, settings: RerankSettings) -> None:
        self.settings = settings

    def rerank(self, query, candidates, trace=None) -> list[RerankCandidate]:
        # 按 score 降序精排
        return sorted(candidates, key=lambda c: c.score, reverse=True)


@pytest.fixture(autouse=True)
def _register_fake() -> None:
    RerankerFactory.register("fake", FakeReranker)
    yield
    RerankerFactory._registry.pop("fake", None)


def make_settings(**overrides) -> RerankSettings:
    base = {"provider": "none", "enabled": False, "top_m": 10}
    base.update(overrides)
    return RerankSettings(**base)


def make_candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", text="aa", score=0.3),
        RerankCandidate(id="b", text="bb", score=0.9),
        RerankCandidate(id="c", text="cc", score=0.5),
    ]


# ---------- NoneReranker ----------

def test_none_reranker_keeps_order() -> None:
    reranker = RerankerFactory.create(make_settings(provider="none", enabled=True))
    assert isinstance(reranker, NoneReranker)
    cands = make_candidates()
    assert [c.id for c in reranker.rerank("q", cands)] == ["a", "b", "c"]


def test_disabled_reranker_returns_none_even_for_registered_backend() -> None:
    reranker = RerankerFactory.create(make_settings(provider="fake", enabled=False))
    assert isinstance(reranker, NoneReranker)


# ---------- 工厂路由 ----------

def test_factory_routes_to_registered_backend() -> None:
    reranker = RerankerFactory.create(make_settings(provider="fake", enabled=True))
    assert isinstance(reranker, FakeReranker)


def test_factory_passes_settings() -> None:
    reranker = RerankerFactory.create(make_settings(provider="fake", enabled=True, top_m=5))
    assert reranker.settings.top_m == 5


def test_fake_reranker_sorts_by_score() -> None:
    reranker = RerankerFactory.create(make_settings(provider="fake", enabled=True))
    result = reranker.rerank("q", make_candidates())
    assert [c.id for c in result] == ["b", "c", "a"]


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="未知的 Reranker backend"):
        RerankerFactory.create(make_settings(provider="nonexistent", enabled=True))
