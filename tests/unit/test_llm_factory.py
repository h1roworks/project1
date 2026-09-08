"""B1: LLM 抽象接口与工厂测试。

用 Fake LLM 验证工厂路由逻辑，不依赖任何真实后端。
"""

import pytest

from core.settings import LLMSettings
from libs.llm.base_llm import BaseLLM, ChatResponse
from libs.llm.llm_factory import LLMFactory


class FakeLLM(BaseLLM):
    provider = "fake"

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def chat(self, messages, trace=None, **kwargs) -> ChatResponse:
        return ChatResponse(content="fake-reply", model=self.settings.model)


@pytest.fixture(autouse=True)
def _register_fake() -> None:
    LLMFactory.register("fake", FakeLLM)
    yield
    LLMFactory._registry.pop("fake", None)


def make_settings(**overrides) -> LLMSettings:
    base = {"provider": "fake", "model": "fake-model"}
    base.update(overrides)
    return LLMSettings(**base)


def test_factory_returns_impl_instance() -> None:
    llm = LLMFactory.create(make_settings())
    assert isinstance(llm, FakeLLM)
    assert isinstance(llm, BaseLLM)


def test_factory_passes_settings_through() -> None:
    llm = LLMFactory.create(make_settings(model="qwen-plus", temperature=0.3))
    assert llm.settings.model == "qwen-plus"
    assert llm.settings.temperature == 0.3


def test_chat_returns_structured_response() -> None:
    llm = LLMFactory.create(make_settings())
    resp = llm.chat([{"role": "user", "content": "hi"}])
    assert isinstance(resp, ChatResponse)
    assert resp.content == "fake-reply"
    assert resp.model == "fake-model"


def test_empty_provider_raises() -> None:
    with pytest.raises(ValueError, match="provider 未配置"):
        LLMFactory.create(make_settings(provider=""))


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="未知的 LLM provider"):
        LLMFactory.create(make_settings(provider="nonexistent"))
