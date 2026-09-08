"""B7.1: OpenAI/Azure/DeepSeek/DashScope LLM 冒烟测试。

使用 httpx.MockTransport mock HTTP，不触发真实网络。覆盖工厂路由、正常对话、
输入校验与 API 错误的可读包装。
"""

import json

import httpx
import pytest

from core.settings import LLMSettings
from libs.llm.azure_llm import AzureLLM
from libs.llm.dashscope_llm import DashScopeLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import OpenAILLM

CHAT_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-test",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "你好，测试回复"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
}


def make_settings(**overrides) -> LLMSettings:
    base = {
        "provider": "openai",
        "model": "test-model",
        "api_key": "test-key",
        "base_url": "",
        "temperature": 0.2,
        "max_tokens": 128,
    }
    base.update(overrides)
    return LLMSettings(**base)


def make_mock_client(
    status: int = 200,
    payload: dict | None = None,
    captured: dict | None = None,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured["body"] = json.loads(request.content)
        return httpx.Response(
            status,
            json=payload if payload is not None else CHAT_RESPONSE,
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------- 工厂路由 ----------


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("openai", OpenAILLM),
        ("azure", AzureLLM),
        ("deepseek", DeepSeekLLM),
        ("dashscope", DashScopeLLM),
    ],
)
def test_factory_routes_to_provider(provider, expected) -> None:
    llm = LLMFactory.create(make_settings(provider=provider))
    assert isinstance(llm, expected)


def test_dashscope_default_base_url() -> None:
    assert DashScopeLLM._default_base_url == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


# ---------- 正常对话（mock HTTP） ----------


def test_openai_chat_success() -> None:
    captured: dict = {}
    llm = OpenAILLM(
        make_settings(provider="openai", model="gpt-4o"),
        http_client=make_mock_client(captured=captured),
    )
    resp = llm.chat([{"role": "user", "content": "你好"}])
    assert resp.content == "你好，测试回复"
    assert resp.model == "gpt-test"
    assert resp.usage["total_tokens"] == 12
    # 请求体透传了 model/temperature/max_tokens
    body = captured["body"]
    assert body["model"] == "gpt-4o"
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 128


def test_azure_chat_uses_deployment_name() -> None:
    captured: dict = {}
    llm = AzureLLM(
        make_settings(
            provider="azure",
            model="",
            deployment_name="my-deploy",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-06-01",
        ),
        http_client=make_mock_client(captured=captured),
    )
    resp = llm.chat([{"role": "user", "content": "hi"}])
    assert resp.content == "你好，测试回复"
    assert captured["body"]["model"] == "my-deploy"


# ---------- 输入校验 ----------


@pytest.mark.parametrize(
    "bad", ["not-a-list", [], [{"role": "user"}], [{"content": "x"}]]
)
def test_chat_rejects_invalid_messages(bad) -> None:
    llm = OpenAILLM(make_settings(), http_client=make_mock_client())
    with pytest.raises(ValueError, match="messages|消息"):
        llm.chat(bad)


# ---------- API 错误包装 ----------


def test_api_error_wrapped_with_provider_and_type() -> None:
    llm = OpenAILLM(
        make_settings(),
        http_client=make_mock_client(
            status=401,
            payload={"error": {"message": "bad key", "type": "invalid_request_error"}},
        ),
    )
    with pytest.raises(RuntimeError, match="provider=openai") as exc_info:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "AuthenticationError" in str(exc_info.value)


def test_azure_missing_config_readable() -> None:
    llm = AzureLLM(make_settings(provider="azure", azure_endpoint="", api_version=""))
    with pytest.raises(RuntimeError, match="azure_endpoint"):
        llm.chat([{"role": "user", "content": "hi"}])
