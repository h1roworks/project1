"""B2: Embedding 抽象接口与工厂测试。

用 Fake Embedding 验证工厂路由与批量 embed 契约。
"""

import pytest

from core.settings import EmbeddingSettings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    provider = "fake"

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings

    def embed(self, texts, trace=None) -> list[list[float]]:
        # 稳定向量：维度 = settings.dimensions，值来自文本内容 hash（保持确定性）
        return [[float(len(t))] * self.settings.dimensions for t in texts]


@pytest.fixture(autouse=True)
def _register_fake() -> None:
    EmbeddingFactory.register("fake", FakeEmbedding)
    yield
    EmbeddingFactory._registry.pop("fake", None)


def make_settings(**overrides) -> EmbeddingSettings:
    base = {"provider": "fake", "model": "fake-embed", "dimensions": 4}
    base.update(overrides)
    return EmbeddingSettings(**base)


def test_factory_returns_impl_instance() -> None:
    emb = EmbeddingFactory.create(make_settings())
    assert isinstance(emb, FakeEmbedding)
    assert isinstance(emb, BaseEmbedding)


def test_embed_returns_expected_shape() -> None:
    emb = EmbeddingFactory.create(make_settings(dimensions=4))
    vectors = emb.embed(["a", "longer text", "c"])
    assert len(vectors) == 3
    assert all(len(v) == 4 for v in vectors)


def test_embed_deterministic() -> None:
    emb = EmbeddingFactory.create(make_settings())
    assert emb.embed(["hello"]) == emb.embed(["hello"])


def test_empty_texts() -> None:
    emb = EmbeddingFactory.create(make_settings())
    assert emb.embed([]) == []


def test_empty_provider_raises() -> None:
    with pytest.raises(ValueError, match="provider 未配置"):
        EmbeddingFactory.create(make_settings(provider=""))


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="未知的 Embedding provider"):
        EmbeddingFactory.create(make_settings(provider="nonexistent"))
