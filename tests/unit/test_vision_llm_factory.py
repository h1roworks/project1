"""B8: Vision LLM 抽象接口与工厂测试。"""

import pytest

from core.settings import VisionLLMSettings
from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM
from libs.llm.llm_factory import LLMFactory


class FakeVisionLLM(BaseVisionLLM):
    provider = "fake_vision"

    def __init__(self, settings: VisionLLMSettings) -> None:
        self.settings = settings

    def chat_with_image(self, text, image_path, trace=None, **kwargs) -> ChatResponse:
        return ChatResponse(content=f"vision:{text}")


@pytest.fixture(autouse=True)
def _register_fake() -> None:
    LLMFactory.register_vision("fake_vision", FakeVisionLLM)
    yield
    LLMFactory._vision_registry.pop("fake_vision", None)


def make_settings(**overrides) -> VisionLLMSettings:
    base = {"provider": "fake_vision", "model": "fake-vl"}
    base.update(overrides)
    return VisionLLMSettings(**base)


def test_base_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseVisionLLM()  # type: ignore[abstract]


def test_factory_returns_impl_instance() -> None:
    v = LLMFactory.create_vision_llm(make_settings())
    assert isinstance(v, FakeVisionLLM)
    assert isinstance(v, BaseVisionLLM)


def test_chat_with_image_returns_structured() -> None:
    v = LLMFactory.create_vision_llm(make_settings())
    resp = v.chat_with_image("描述图片", "dummy.png")
    assert isinstance(resp, ChatResponse)
    assert resp.content == "vision:描述图片"


def test_empty_provider_raises() -> None:
    with pytest.raises(ValueError, match="Vision LLM provider 未配置"):
        LLMFactory.create_vision_llm(make_settings(provider=""))


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="未知的 Vision LLM provider"):
        LLMFactory.create_vision_llm(make_settings(provider="nonexistent"))


def test_factory_creates_dashscope_vision() -> None:
    """config 默认 provider 为 dashscope，OpenAI 兼容 Vision 必须可创建。"""
    from libs.llm.openai_vision_llm import OpenAIVisionLLM

    v = LLMFactory.create_vision_llm(
        VisionLLMSettings(provider="dashscope", model="qwen-vl-plus")
    )
    assert isinstance(v, OpenAIVisionLLM)
