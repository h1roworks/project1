"""B7.4: Ollama Embedding 测试（注入 Fake client，不触网）。

覆盖批量 embed、默认 host、空/非法/超长输入行为、连接失败包装与工厂路由。
"""

import pytest

from core.settings import EmbeddingSettings
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.ollama_embedding import OllamaEmbedding


class FakeOllamaClient:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc
        self.calls: list[dict] = []

    def embed(self, model, input, **kwargs) -> dict:
        self.calls.append({"model": model, "input": list(input)})
        if self.exc:
            raise self.exc
        return {"embeddings": [[float(len(t)), 1.0] for t in input]}


def make_settings(**overrides) -> EmbeddingSettings:
    base = {
        "provider": "ollama",
        "model": "nomic-embed-text",
        "api_key": "",
        "base_url": "http://localhost:11434",
        "dimensions": 2,
        "batch_size": 32,
        "max_input_chars": 0,
    }
    base.update(overrides)
    return EmbeddingSettings(**base)


def test_embed_success() -> None:
    fake = FakeOllamaClient()
    emb = OllamaEmbedding(make_settings(), client=fake)
    vectors = emb.embed(["a", "bb"])
    assert vectors == [[1.0, 1.0], [2.0, 1.0]]  # 维度 2，来自输入长度
    call = fake.calls[0]
    assert call["model"] == "nomic-embed-text"
    assert call["input"] == ["a", "bb"]


def test_default_host_when_base_url_empty() -> None:
    emb = OllamaEmbedding(make_settings(base_url=""), client=FakeOllamaClient())
    assert emb._host == "http://localhost:11434"


def test_batch_multiple_texts_single_call() -> None:
    fake = FakeOllamaClient()
    emb = OllamaEmbedding(make_settings(), client=fake)
    vectors = emb.embed(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    assert len(fake.calls) == 1  # 一次批量调用
    assert len(fake.calls[0]["input"]) == 5


def test_empty_input_returns_empty() -> None:
    emb = OllamaEmbedding(make_settings(), client=FakeOllamaClient())
    assert emb.embed([]) == []


def test_invalid_texts_raise() -> None:
    emb = OllamaEmbedding(make_settings(), client=FakeOllamaClient())
    with pytest.raises(ValueError, match="list"):
        emb.embed("oops")
    with pytest.raises(ValueError, match="字符串"):
        emb.embed(["ok", None])


def test_oversize_input_truncated_by_config() -> None:
    fake = FakeOllamaClient()
    emb = OllamaEmbedding(make_settings(max_input_chars=5), client=fake)
    emb.embed(["hello world long text"])
    assert fake.calls[0]["input"] == ["hello"]


def test_connection_failure_readable() -> None:
    emb = OllamaEmbedding(
        make_settings(),
        client=FakeOllamaClient(exc=ConnectionError("connection refused")),
    )
    with pytest.raises(RuntimeError, match="Ollama Embedding 调用失败") as exc_info:
        emb.embed(["a"])
    assert "ConnectionError" in str(exc_info.value)
    # 错误信息不泄露敏感配置
    assert "api_key" not in str(exc_info.value)


def test_factory_creates() -> None:
    emb = EmbeddingFactory.create(make_settings())
    assert isinstance(emb, OllamaEmbedding)
