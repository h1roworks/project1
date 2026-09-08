"""B7.3: OpenAI / Azure / DashScope Embedding 测试（httpx.MockTransport mock HTTP）。

覆盖工厂路由、请求体正确性（model/input）、批量化、Azure deployment_name、
空/非法/超长输入行为、认证失败与缺失配置的可读错误。
"""

import json

import httpx
import pytest

from core.settings import EmbeddingSettings
from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.dashscope_embedding import DashScopeEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import OpenAIEmbedding


def make_settings(**overrides) -> EmbeddingSettings:
    base = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": "test-key",
        "base_url": "https://api.openai.com/v1",
        "dimensions": 3,
        "batch_size": 32,
        "max_input_chars": 0,
        "azure_endpoint": "",
        "api_version": "",
        "deployment_name": "",
    }
    base.update(overrides)
    return EmbeddingSettings(**base)


def make_mock_client(
    captured: dict,
    status: int = 200,
    payload: dict | None = None,
) -> httpx.Client:
    """MockTransport：按请求 input 条数生成等长 embedding，并记录请求体。"""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.setdefault("bodies", []).append(body)
        n = len(body["input"])
        data = [
            {
                "object": "embedding",
                "embedding": [float(captured.setdefault("counter", -1) + 1 + i), 0.0, 0.0],
                "index": i,
            }
            for i in range(n)
        ]
        captured["counter"] = captured["counter"] + n
        resp = payload
        if resp is None:
            resp = {
                "object": "list",
                "data": data,
                "model": body["model"],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            }
        return httpx.Response(status, json=resp, request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------- 工厂路由 ----------

@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("openai", OpenAIEmbedding),
        ("azure", AzureEmbedding),
        ("dashscope", DashScopeEmbedding),
    ],
)
def test_factory_routes_embedding_provider(provider, expected) -> None:
    emb = EmbeddingFactory.create(make_settings(provider=provider))
    assert isinstance(emb, expected)
    assert isinstance(emb, BaseEmbedding)


def test_dashscope_default_base_url() -> None:
    """config 默认 provider 为 dashscope，OpenAI 兼容 base_url 必须正确。"""
    assert (
        DashScopeEmbedding._default_base_url
        == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


# ---------- OpenAI 调用 ----------

def test_openai_embed_success() -> None:
    captured: dict = {}
    emb = OpenAIEmbedding(make_settings(), http_client=make_mock_client(captured))
    vectors = emb.embed(["a", "bb"])
    assert len(vectors) == 2
    assert vectors[0] == [0.0, 0.0, 0.0]  # index=0
    assert vectors[1] == [1.0, 0.0, 0.0]  # index=1
    body = captured["bodies"][0]
    assert body["model"] == "text-embedding-3-small"
    assert body["input"] == ["a", "bb"]


def test_embed_respects_batch_size() -> None:
    captured: dict = {}
    emb = OpenAIEmbedding(
        make_settings(batch_size=2), http_client=make_mock_client(captured)
    )
    vectors = emb.embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert len(captured["bodies"]) == 2  # [a,b] + [c] 两次请求


def test_embed_order_preserved_across_batches() -> None:
    captured: dict = {}
    emb = OpenAIEmbedding(
        make_settings(batch_size=2), http_client=make_mock_client(captured)
    )
    vectors = emb.embed(["a", "b", "c"])
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]  # 与输入顺序一致


# ---------- Azure ----------

def test_azure_uses_deployment_name() -> None:
    captured: dict = {}
    emb = AzureEmbedding(
        make_settings(
            provider="azure",
            model="",
            base_url="",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-06-01",
            deployment_name="emb-deploy",
        ),
        http_client=make_mock_client(captured),
    )
    emb.embed(["a"])
    assert captured["bodies"][0]["model"] == "emb-deploy"


def test_azure_missing_config_readable() -> None:
    emb = AzureEmbedding(make_settings(provider="azure", base_url=""))
    with pytest.raises(RuntimeError, match="azure_endpoint"):
        emb.embed(["a"])

    emb = AzureEmbedding(
        make_settings(
            provider="azure",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="",
        )
    )
    with pytest.raises(RuntimeError, match="api_version"):
        emb.embed(["a"])


# ---------- 输入行为 ----------

def test_empty_input_returns_empty() -> None:
    emb = OpenAIEmbedding(make_settings(), http_client=make_mock_client({}))
    assert emb.embed([]) == []


def test_invalid_texts_raise() -> None:
    emb = OpenAIEmbedding(make_settings(), http_client=make_mock_client({}))
    with pytest.raises(ValueError, match="list"):
        emb.embed("not-a-list")
    with pytest.raises(ValueError, match="字符串"):
        emb.embed(["ok", 123])


def test_oversize_input_truncated_by_config() -> None:
    captured: dict = {}
    emb = OpenAIEmbedding(
        make_settings(max_input_chars=5), http_client=make_mock_client(captured)
    )
    emb.embed(["hello world long text"])
    assert captured["bodies"][0]["input"] == ["hello"]


def test_no_truncation_when_disabled() -> None:
    captured: dict = {}
    emb = OpenAIEmbedding(
        make_settings(max_input_chars=0), http_client=make_mock_client(captured)
    )
    emb.embed(["hello world"])
    assert captured["bodies"][0]["input"] == ["hello world"]


# ---------- 错误包装 ----------

def test_api_error_wrapped_readable() -> None:
    emb = OpenAIEmbedding(
        make_settings(),
        http_client=make_mock_client(
            captured={},
            status=401,
            payload={"error": {"message": "bad key", "type": "invalid_request_error"}},
        ),
    )
    with pytest.raises(RuntimeError, match="provider=openai") as exc_info:
        emb.embed(["a"])
    assert "AuthenticationError" in str(exc_info.value)
