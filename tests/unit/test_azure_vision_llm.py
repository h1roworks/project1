"""B9: Azure Vision LLM 测试（httpx.MockTransport mock HTTP，不走真实 API）。

覆盖正常调用（字节/路径两种输入）、deployment_name、图片压缩、认证失败与
缺失配置的可读错误。
"""

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from core.settings import VisionLLMSettings
from libs.llm.azure_vision_llm import AzureVisionLLM
from libs.llm.llm_factory import LLMFactory

CHAT_RESPONSE = {
    "id": "chatcmpl-vision-test",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "这是测试图片"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def make_settings(**overrides) -> VisionLLMSettings:
    base = {
        "provider": "azure",
        "model": "gpt-4o",
        "api_key": "test-key",
        "base_url": "",
        "azure_endpoint": "https://test.openai.azure.com/",
        "api_version": "2024-06-01",
        "deployment_name": "gpt-4o-deploy",
        "max_tokens": 128,
        "max_image_size": 64,
    }
    base.update(overrides)
    return VisionLLMSettings(**base)


def make_mock_client(
    captured: dict,
    status: int = 200,
    payload: dict | None = None,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            status,
            json=payload if payload is not None else CHAT_RESPONSE,
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def make_image(size: tuple[int, int], color: str = "red", fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format=fmt)
    return buf.getvalue()


def _image_url_of(captured: dict) -> str:
    content = captured["body"]["messages"][0]["content"]
    return content[1]["image_url"]["url"]


def test_chat_with_image_from_bytes() -> None:
    captured: dict = {}
    llm = AzureVisionLLM(make_settings(), http_client=make_mock_client(captured))
    resp = llm.chat_with_image("描述图片", make_image((8, 8)))
    assert resp.content == "这是测试图片"
    assert resp.model == "gpt-4o"
    assert _image_url_of(captured).startswith("data:image/png;base64,")


def test_chat_with_image_from_path(tmp_path) -> None:
    path = tmp_path / "tiny.png"
    path.write_bytes(make_image((8, 8)))
    captured: dict = {}
    llm = AzureVisionLLM(make_settings(), http_client=make_mock_client(captured))
    resp = llm.chat_with_image("描述图片", str(path))
    assert resp.content == "这是测试图片"


def test_chat_uses_deployment_name() -> None:
    captured: dict = {}
    llm = AzureVisionLLM(make_settings(), http_client=make_mock_client(captured))
    llm.chat_with_image("描述图片", make_image((8, 8)))
    assert captured["body"]["model"] == "gpt-4o-deploy"


def test_oversize_image_is_compressed() -> None:
    captured: dict = {}
    llm = AzureVisionLLM(
        make_settings(max_image_size=16),
        http_client=make_mock_client(captured),
    )
    llm.chat_with_image("描述图片", make_image((200, 100)))
    url = _image_url_of(captured)
    data = base64.b64decode(url.split(",", 1)[1])
    img = Image.open(io.BytesIO(data))
    assert max(img.size) <= 16


def test_small_image_not_resized() -> None:
    raw = make_image((8, 8))
    captured: dict = {}
    llm = AzureVisionLLM(
        make_settings(max_image_size=64), http_client=make_mock_client(captured)
    )
    llm.chat_with_image("描述图片", raw)
    url = _image_url_of(captured)
    data = base64.b64decode(url.split(",", 1)[1])
    assert Image.open(io.BytesIO(data)).size == (8, 8)


def test_api_error_wrapped_readable() -> None:
    llm = AzureVisionLLM(
        make_settings(),
        http_client=make_mock_client(
            captured={},
            status=401,
            payload={"error": {"message": "bad key", "type": "invalid_request_error"}},
        ),
    )
    with pytest.raises(RuntimeError, match="provider=azure") as exc_info:
        llm.chat_with_image("描述图片", make_image((8, 8)))
    assert "AuthenticationError" in str(exc_info.value)


def test_missing_azure_endpoint_readable() -> None:
    llm = AzureVisionLLM(make_settings(azure_endpoint="", base_url=""))
    with pytest.raises(RuntimeError, match="azure_endpoint"):
        llm.chat_with_image("描述图片", make_image((8, 8)))


def test_factory_creates_azure_vision() -> None:
    llm = LLMFactory.create_vision_llm(make_settings())
    assert isinstance(llm, AzureVisionLLM)
